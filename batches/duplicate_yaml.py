import yaml

"""
batch 1
-rw-r--r--  1 eartigau  staff  828446400 28 oct  2022 jw01201101001_04101_00001-seg001_nis_rateints.fits
-rw-r--r--  1 eartigau  staff  440464320 28 oct  2022 jw01201101001_04101_00001-seg002_nis_rateints.fits
batch 2
-rw-r--r--  1 eartigau  staff  828452160 19 jui  2023 jw01201105001_04101_00001-seg001_nis_rateints.fits
-rw-r--r--  1 eartigau  staff  440467200 19 jui  2023 jw01201105001_04101_00001-seg002_nis_rateints.fits
batch 3
-rw-r--r--  1 eartigau  staff  828452160 25 jui  2023 jw01201104001_04101_00001-seg001_nis_rateints.fits
-rw-r--r--  1 eartigau  staff  440467200 25 jui  2023 jw01201104001_04101_00001-seg002_nis_rateints.fits
batch 4
-rw-r--r--  1 eartigau  staff  828452160 10 jul  2023 jw01201102001_04101_00001-seg001_nis_rateints.fits
-rw-r--r--  1 eartigau  staff  440467200 10 jul  2023 jw01201102001_04101_00001-seg002_nis_rateints.fits
batch 5
-rw-r--r--  1 eartigau  staff  828452160 24 jul  2023 jw01201103001_04101_00001-seg001_nis_rateints.fits
-rw-r--r--  1 eartigau  staff  440467200 24 jul  2023 jw01201103001_04101_00001-seg002_nis_rateints.fits
"""

reference_file = 'config_t1f.yaml'

params = yaml.load(open(reference_file), Loader=yaml.FullLoader)

for batch in [1,2,3,4,5,6,7,8,9,10]:

    if batch == 1:
        params['files'] = ['jw01201101001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201101001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [67, 73, 96, 102]
        params['suffix'] = 'visit1'

    elif batch == 2:
        params['files'] = ['jw01201105001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201105001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [69, 77, 100, 106]
        params['suffix'] = 'visit2'

    elif batch == 3:
        params['files'] = ['jw01201104001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201104001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [46, 54, 75, 83]
        params['suffix'] = 'visit3'

    elif batch == 4:
        params['files'] = ['jw01201102001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201102001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [66, 72, 85, 101]
        params['suffix'] = 'visit4'

    elif batch == 5:
        params['files'] = ['jw01201103001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201103001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [54, 62, 83, 91]
        params['suffix'] = 'visit5'

    elif batch == 6:
        params['files'] = ['jw01201105001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201105001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [69, 77, 100, 106]
        params['suffix'] = 'visit2'
        params['percentile_bgnd'] = 10
    elif batch == 7:
        params['files'] = ['jw01201105001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201105001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [69, 77, 100, 106]
        params['suffix'] = 'visit2'
        params['percentile_bgnd'] = 20

    elif batch == 8:
        params['files'] = ['jw01201105001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201105001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [69, 77, 100, 106]
        params['suffix'] = 'visit2'
        params['percentile_bgnd'] = 30

    elif batch == 9:
        params['files'] = ['jw01201105001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201105001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [69, 77, 100, 106]
        params['suffix'] = 'visit2'
        params['percentile_bgnd'] = 20
        params['trace_width_extraction'] = 30

    elif batch == 10:
        params['files'] = ['jw01201105001_04101_00001-seg001_nis_rateints.fits',
                           'jw01201105001_04101_00001-seg002_nis_rateints.fits']
        params['it'] = [69, 77, 100, 106]
        params['suffix'] = 'visit2'
        params['percentile_bgnd'] = 20
        params['trace_width_extraction'] = 20


    outname = '{}_batch{}.yaml'.format(reference_file.split('.yaml')[0], batch)

    print('Writing {}'.format(outname))
    with open(outname, 'w') as f:
        yaml.dump(params, f)
