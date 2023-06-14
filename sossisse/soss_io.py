import hashlib
import shutil
from datetime import datetime

import yaml
from astraeus import xarrayIO as xrio
from etienne_tools import printc
import numpy as np
from astropy.io import fits
from astropy.table import Table
import glob
from tqdm import tqdm
import os

def yaml_to_html(params):

    keys= params.keys()

    out_keys = []
    out_txt = []

    for key in keys:
        tmp = params[key]
        #print(type(tmp))
        if type(tmp) == bool:
            out_keys = np.append(out_keys,key)
            out_txt = np.append(out_txt,'{}<br>\n'.format(str(tmp)))
        if type(tmp) == str:
            out_keys = np.append(out_keys,key)
            out_txt = np.append(out_txt,'{}<br>\n'.format(tmp))
        if type(tmp) == list:

            if type(tmp[0]) == str:
                tmp = '<br>'.join(np.array(tmp,dtype = 'U999'))
                out_keys = np.append(out_keys,key)
                out_txt = np.append(out_txt,'{}<br>\n'.format(tmp))
            else:
                tmp = ', '.join(np.array(tmp,dtype = 'U999'))
                out_keys = np.append(out_keys,key)
                out_txt = np.append(out_txt,'[{}]<br>\n'.format(tmp))

    out_html = ''
    out_html +=    """
    <!DOCTYPE html>
    <html>
    <body>
    """
    out_html +="<h1>{}</h1>\n".format(params['object'])
    out_html +="<h1>{}</h1>\n".format(str(datetime.now()))

    png_files = glob.glob(params['PLOT_PATH']+'/*.png')
    out_html +="<h2>{}</h2>\n".format('All plots')

    for png_file in png_files:
        out_html += '<img src="{0}" alt="{0}" width = "600"><br><br><br>'.format(png_file.split('/')[-1])

    csv_files = glob.glob(params['PLOT_PATH']+'/*.csv')


    out_html+='<br><br><br>'
    out_html +="<h2>{}</h2>\n".format('All CSVs')

    for csv_file in csv_files:
        out_html += '<a href="{0}">{0}</a><br>'.format(csv_file.split('/')[-1])

    out_html+='<br><br><br>'
    out_html +="<h2>{}</h2>\n".format('All inputs from yaml file')

    for i in range(len(out_keys)):
        out_html +="<p> <b>{} </b></p>\n".format(out_keys[i])
        out_html +=out_txt[i]

    out_html +="""
    </body>
    </html>
    """
    f = open(params['PLOT_PATH']+"/index.html", "w")
    f.write(out_html)
    f.close()

    outdir = params['object']+'_'+datetime.now().isoformat('_').replace(':','-').split('.')[0]+'_'+params['checksum']
    cmd = 'rsync -av -e "ssh  -oPort=5822"  {}/* artigau@venus.astro.umontreal.ca:/home/artigau/www/sossisse/{}'.format(
        params['PLOT_PATH'],outdir)

    os.system(cmd)

def to_eureka_fmt(flux, flux_error, wave, time, outfile):
    # flux and flux_error should be of shape time x wavelength
    # wavelength should be in descending order (and in microns)
    # time in BJD_TBD
    # outfile is the name you want to save your file as

    outdata = xrio.makeDataset()
    outdata['optspec'] = (['time', 'x'], flux[:, ::-1])
    outdata['optspec'].attrs['flux_units'] = 'e-/s'
    outdata['optspec'].attrs['time_units'] = 'BJD_TBD'

    outdata['opterr'] = (['time', 'x'], flux_error[:, ::-1])
    outdata['opterr'].attrs['flux_units'] = 'e-/s'
    outdata['opterr'].attrs['time_units'] = 'BJD_TBD'

    outdata['wave_1d'] = (['x'], wave[::-1])
    outdata['wave_1d'].attrs['wave_units'] = 'micron'

    outdata.coords['time'] = time

    xrio.writeXR(outfile + '.h5', outdata)

# =====================================================================================================================
# Define functions
# =====================================================================================================================
def sossisson_to_eureka(params):
    """
    Convert a sossisson output file in the FITS format to the standard Eureka! (h5) format.

    Parameters
    ----------
    param_file: str
                Name of the config file.

    Returns
    -------
    0

    """

    # Get the tag for unique ID
    params = mk_tag(params)

    for trace_order in params['trace_orders']:
        printc("Processing order {}...".format(trace_order), 'number')

        # Get file names
        fname = "{}/spectra_ord{}{}.fits".format(params['FITS_PATH'], trace_order, params['tag2'])  # "extract1d" specs
        fnames_time = params['files']  # "raw" data files to get time stamps

        # Define the file name for the output
        outfname = fname[:-len(".fits")]

        printc("Opening sossisson output...", 'info')

        ffile = fits.open(fname)
        flux, flux_error, wave = ffile["RELFLUX"].data, ffile["RELFLUX_ERROR"].data, ffile["WAVELENGTH"].data
        wave = wave[0, :]  # assuming wavelength arrays are the same for all integrations

        printc("Sorting in decreasing wavelength order...", 'info')
        isort = np.argsort(wave)[::-1]
        flux, flux_error, wave = flux[:, isort], flux_error[:, isort], wave[isort]

        printc("Reading jw file(s) to get time array...", 'info')
        time = np.array([])
        for fname_time_i in fnames_time:
            in_times_tbl = fits.open(fname_time_i)["INT_TIMES"].data
            time = np.concatenate((time, Table(in_times_tbl)["int_mid_BJD_TDB"].data))

        printc("Converting to Eureka! format...", 'info')
        to_eureka_fmt(flux, flux_error, wave, time, outfname)
        printc("Done.", 'info')

    return 0

def clean_doublets():
    """
    we look at *all* files in the folders, and files with the same checksum are replaced with hard links. This can save
    a huge amount of disk space
    """

    files = glob.glob('*/*/*/*.fits')
    cs = []
    for file in tqdm(files, leave=False):
        cs = np.append(cs, get_checksum(file))
    cs = np.array(cs)

    for ucs in tqdm(np.unique(cs), leave=False):
        g = np.where(cs == ucs)[0]
        if len(g) == 1:
            continue

        for gg in g[1:]:
            cmd = 'rm ' + files[gg]
            os.system(cmd)
            cmd = 'ln -L ' + files[g[0]] + ' ' + files[gg]
            os.system(cmd)

    return



def get_checksum(filename, hash_function="md5"):
    """Generate checksum for file baed on hash function (MD5 or SHA256).

    Args:
        filename (str): Path to file that will have the checksum generated.
        hash_function (str):  Hash function name - supports MD5 or SHA256

    Returns:
        str`: Checksum based on Hash function of choice.

    Raises:
        Exception: Invalid hash function is entered.

    """
    hash_function = hash_function.lower()

    with open(filename, "rb") as f:
        bytes = f.read()  # read file as bytes
        if hash_function == "md5":
            readable_hash = hashlib.md5(bytes).hexdigest()
        elif hash_function == "sha256":
            readable_hash = hashlib.sha256(bytes).hexdigest()
        else:
            raise ValueError("{} is an invalid hash function. Please Enter MD5 or SHA256")

    return readable_hash

def mk_tag(params):
    cds_tag = ''
    if "cds_id" in params.keys():
        cds_tag = "_cds-{}-{}".format(params["cds_id"][0], params["cds_id"][1])

    wlc_domain_tag = ''
    if 'wlc domain' in params.keys():
        wlc_domain_tag = "_wlcdomain-{}-{}um".format(params["wlc domain"][0], params["wlc domain"][1])

    tag = ("{}{}_ootmed{}_bkg{}_1fpolyord{}_fitrot{}_fitzp{}_fitddy{}_it1-{}-it4-{}"
           .format(cds_tag, wlc_domain_tag, int(params['ootmed']), int(params['do_bkg']), params["degree_1f_corr"],
                   int(params["fit_rotation"]), int(params["zero_point_offset"]), int(params["ddy"]),
                   params['it'][0], params['it'][3]))

    params['tag'] = tag

    tag2 = tag + "_it2-{}-it3-{}_remoottrend{}".format(params['it'][1], params['it'][2], int(params['remove_trend']))
    tag2 = tag2 + "_transit-base-polyord-{}".format(params['transit_baseline_polyord'] if params['remove_trend'] else
                                                    "None")
    params['tag2'] = tag2

    return params

def load_yaml_default(silent = False):
    default_param_file = os.path.join(os.path.dirname(__file__),'data','defaults.yaml')
    with open(default_param_file, "r") as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.FullLoader)
        if not silent:
            printc("Read of default parameters successful", 'info')

    return params

def load_yaml_params(param_file, force=False, do_time_link = False, silent = False):
    with open(param_file, "r") as yamlfile:
        params = yaml.load(yamlfile, Loader=yaml.FullLoader)


        if not silent:
            printc("Read of parameters successful", 'info')

    params = dict(params)
    params2 = dict(params)
    for key in params.keys():
        key_tmp = str(key)
        if ' ' in key:
            printc('rename "{}" for "{}" in {}'.format(key_tmp, key_tmp.replace(' ','_'), param_file),'bad3')
            params2[key_tmp.replace(' ','_')] = params2[key_tmp]
            del params2[key_tmp]
            key_tmp = key_tmp.replace(' ','_')
        if '-' in key:
            printc('rename "{}" for "{}" in {}'.format(key_tmp, key_tmp.replace('-','_'), param_file),'bad3')
            params2[key_tmp.replace('-','_')] = params2[key_tmp]
            del params2[key_tmp]
            key_tmp = key_tmp.replace(' ','_')

    params = dict(params2)

    defaults = dict(load_yaml_default(silent = silent))
    for key in defaults.keys():
        if key not in params.keys():
            if not silent:
                printc('Parameter "{}" is not given in the yaml, we use the default'.format(key), 'bad1')
                printc('\t{} : {}'.format(key, defaults[key]), 'bad1')
            params[key] = defaults[key]

    params['SOSSIOPATH'] = os.getenv('SOSSIOPATH')
    params['checksum'] = get_checksum(param_file)[0:8]
    params['whoami'] = os.getlogin()

    # parameters for the linear system. Will be filled along the way with more
    # parameters
    params['output_names'] = ['amplitude']
    params['output_units'] = ['flux']
    params['output_factor'] = [1.0]

    # we check that SOSSIOPATH exists
    if not os.path.isdir(params['SOSSIOPATH']):
        err_string = 'Path {} does not exit, we stop!'.format(params['SOSSIOPATH'])
        os.mkdir(params['SOSSIOPATH'])
        raise ValueError(err_string)

    params['MODEPATH'] = params['SOSSIOPATH'] + params['mode'] + '/'
    # we check that the MODE path exists
    if not os.path.isdir(params['MODEPATH']):
        err_string = 'Path {} does not exit, we stop!'.format(params['MODEPATH'])
        os.mkdir(params['MODEPATH'])
        raise ValueError(err_string)

    params['CALIBPATH'] = params['MODEPATH'] + 'calibrations'
    if not os.path.isdir(params['CALIBPATH']):
        err_string ='Path {} does not exit, we stop!'.format(params['CALIBPATH'])
        raise ValueError(err_string)

    params['OBJECTPATH'] = params['MODEPATH'] + params['object'] + '/' + params['checksum']
    # we check that the OBJECTPATH path exists
    if not os.path.isdir(params['OBJECTPATH']):
        printc('We create {} that does not exit'.format(params['OBJECTPATH']), 'bad3')
        os.mkdir(params['OBJECTPATH'])

    if do_time_link:
        time = datetime.now().isoformat('_')
        cmd = 'ln -s {} {}'.format( params['checksum'], params['MODEPATH'] + params['object'] + '/' + time)
        os.system(cmd)
        printc(cmd,'bad1')


    params['RAWPATH'] = params['MODEPATH'] + params['object'] + '/rawdata'

    # we check that the OBJECTPATH path exists
    if not os.path.isdir(params['RAWPATH']):
        err_string = 'Path {} does not exit, we stop!'.format(params['RAWPATH'])
        raise ValueError(err_string)

    params['TEMP_PATH'] = params['OBJECTPATH'] + '/temporary'
    # we check that the OBJECTPATH path exists
    if not os.path.isdir(params['TEMP_PATH']):
        if not silent:
            printc('We create {} that does not exit'.format(params['TEMP_PATH']), 'bad1')
        os.mkdir(params['TEMP_PATH'])

    # file that will be created with the diff between in and out of transit
    params['file_temporary_in_vs_out'] = params['TEMP_PATH'] + '/temporary_in_vs_out.fits'

    params['PLOT_PATH'] = params['OBJECTPATH'] + '/plots'
    # we check that the PLOT_PATH path exists
    if not os.path.isdir(params['PLOT_PATH']):
        if not silent:
            printc('We create {} that does not exit'.format(params['PLOT_PATH']), 'bad1')
        os.mkdir(params['PLOT_PATH'])

    params['CSV_PATH'] = params['OBJECTPATH'] + '/csvs'
    # we check that the OBJECTPATH path exists
    if not os.path.isdir(params['CSV_PATH']):
        if not silent:
            printc('We create {} that does not exit'.format(params['CSV_PATH']), 'bad1')
        os.mkdir(params['CSV_PATH'])

    params['FITS_PATH'] = params['OBJECTPATH'] + '/fits'
    # we check that the OBJECTPATH path exists
    if not os.path.isdir(params['FITS_PATH']):
        if not silent:
            printc('We create {} that does not exit'.format(params['FITS_PATH']), 'bad1')
        os.mkdir(params['FITS_PATH'])

    files = []
    for file in params['files']:
        files.append(params['RAWPATH'] + '/' + file)
    params['files'] = files

    for file in params['files']:
        if os.path.isfile(file):
            if not silent:
                printc('File {} exists'.format(file), 'info')
        else:
            err_string = 'We have a problem... {} does not exist!'.format(file)
            raise ValueError(err_string)

    if not params['fit_pca']:
        params['n_pca'] = 0

    params['do_bkg'] = params['bkgfile'] != ''
    if params['do_bkg']:
        params['bkgfile'] = params['CALIBPATH'] + '/' + params['bkgfile']

    if params['bkgfile'] != '':
        if params['CALIBPATH'] not in params['bkgfile']:
            params['bkgfile'] = params['CALIBPATH'] + '/' + params['bkgfile']
        params['bgnd'] = fits.getdata(params['bkgfile'])

    if params['flatfile'] != '':
        if params['CALIBPATH'] not in params['flatfile']:
            params['flatfile'] = params['CALIBPATH'] + '/' + params['flatfile']
        params['flat'] = fits.getdata(params['flatfile'])

    if params['pos_file'] != '':
        if params['CALIBPATH'] not in params['pos_file']:
            params['pos_file'] = params['CALIBPATH'] + '/' + params['pos_file']
        # params['pos'] = Table.read(params['pos_file'])

    if force:
        if not silent:
            printc('We have force = True as an input, we re-create temporary files if they exist.', 'bad1')
        params['allow_temporary'] = False

    # get the tag for unique ID
    params = mk_tag(params)

    # we show or not the plots at the end
    params['show_plots'] = params['whoami'] in params['user_show_plot']

    if 'suffix' not in params.keys():
        params['suffix'] = ''

    # we will *not* mask order 0 if this is not a SOSS file
    if params['mode'] != 'SOSS':
        if not silent:
            printc('We will not mask order 0, this is a {} dataset'.format(params['mode']), 'bad1')
        params['mask_order_0'] = False

    # if mode == PRISM, we should not have params['recenter_trace_position'] = True
    if (params['mode'] == 'PRISM')*(params['recenter_trace_position'] == True):
        printc('With the PRISM mode, we set "recenter_trace_position" = False','bad3')
        params['recenter_trace_position'] = False

    if not silent:
        printc('We copy config file to {}'.format(params['CSV_PATH']), 'info')
    # os.path.copy(param_file,params['CSV_PATH'])

    if params['zero_point_offset'] and params['quadratic_term']:
        raise ValueError('We have *both "zero point offset" and "quadratic term" set to True, this is conceptually '
                         'wrong!')

    outname = params['CSV_PATH'] + '/' + param_file.split('/')[-1]
    if not os.path.exists(outname):
        shutil.copyfile(param_file, outname)

    return params

