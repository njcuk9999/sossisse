import numpy as np
import statsmodels.api as sm
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from typing import Tuple


def lowpassfilter(input_vect, width=101):
    # Computes a low-pass filter of an input vector. This is done while properly handling
    # NaN values, but at the same time being reasonably fast.
    # Algorithm:
    #
    # provide an input vector of an arbtrary length and compute a running NaN median over a
    # box of a given length (width value). The running median is NOT computed at every pixel
    # but at steps of 1/4th of the width value. This provides a vector of points where
    # the nan-median has been computed (ymed) and mean position along the input vector (xmed)
    # of valid (non-NaN) pixels. This xmed/ymed combination is then used in a spline to
    # recover a vector for all pixel positions within the input vector.
    #
    # When there are no valid pixel in a 'width' domain, the value is skipped in the creation
    # of xmed and ymed, and the domain is splined over.

    # indices along input vector
    index = np.arange(len(input_vect))

    # placeholders for x and y position along vector
    xmed = []
    ymed = []

    # loop through the lenght of the input vector
    for i in np.arange(-width // 2, len(input_vect) + width // 2, width // 4):

        # if we are at the start or end of vector, we go 'off the edge' and
        # define a box that goes beyond it. It will lead to an effectively
        # smaller 'width' value, but will provide a consistent result at edges.
        low_bound = i
        high_bound = i + int(width)

        if low_bound < 0:
            low_bound = 0
        if high_bound > (len(input_vect) - 1):
            high_bound = (len(input_vect) - 1)

        pixval = index[low_bound:high_bound]

        if len(pixval) < 3:
            continue

        # if no finite value, skip
        if np.max(np.isfinite(input_vect[pixval])) == 0:
            continue

        # mean position along vector and NaN median value of
        # points at those positions
        xmed.append(np.nanmean(pixval))
        ymed.append(np.nanmedian(input_vect[pixval]))

    xmed = np.array(xmed, dtype=float)
    ymed = np.array(ymed, dtype=float)

    # we need at least 3 valid points to return a
    # low-passed vector.
    if len(xmed) < 3:
        return np.zeros_like(input_vect) + np.nan

    if len(xmed) != len(np.unique(xmed)):
        xmed2 = np.unique(xmed)
        ymed2 = np.zeros_like(xmed2)
        for i in range(len(xmed2)):
            ymed2[i] = np.mean(ymed[xmed == xmed2[i]])
        xmed = xmed2
        ymed = ymed2

    # splining the vector
    spline = ius(xmed, ymed, k=2, ext=3)
    lowpass = spline(np.arange(len(input_vect)))

    return lowpass


def robust_polyfit(xvector: np.ndarray, yvector: np.ndarray, degree: int,
                   nsigcut: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    A robust polyfit function that iteratively fits a polynomial to the data until
    the dispersion of values is accounted for by a weight vector. This is
    equivalent to a soft-edged sigma-clipping

    :param xvector: np.ndarray, the x array to pass to np.polyval
    :param yvector: np.ndarray, the y array to pass to np.polyval
    :param degree: int, the degree of polynomial fit passed to np.polyval
    :param nsigcut: float, the threshold sigma above which a point is considered
    and outlier
    :return: a tuple containing the polynomial fit (as a NumPy array)
    and a boolean mask of the good values (p>50% of valid)
    """
    # Initialize the fit to None
    fit = None
    # Create an array of weights, initialized to 1 for all values
    weight = np.ones_like(xvector)
    # Pre-compute the odd_cut value
    odd_cut = np.exp(-.5 * nsigcut ** 2)
    # Initialize an array of weights from the previous iteration,
    # set to 0 for all values
    weight_before = np.zeros_like(weight)
    # Set the maximum number of iterations and initialize the iteration counter
    nite_max = 20
    count = 0
    # remove nans
    keep = np.isfinite(xvector) & np.isfinite(yvector)
    # Enter a loop that will iterate until either the maximum difference
    # between the current and previous weights
    # becomes smaller than a certain threshold, or until the maximum number of
    # iterations is reached
    while (np.max(abs(weight - weight_before)) > 1e-9) and (count < nite_max):
        # Calculate the polynomial fit using the x- and y-values, and the
        # given degree, weighting the fit by the weights. Weights are computed
        # from the dispersion to the fit and the sigmax
        fit = np.polyfit(xvector[keep], yvector[keep], degree, w=weight[keep])
        # Calculate the residuals of the polynomial fit by subtracting the
        # result of np.polyval from the original y-values
        res = yvector - np.polyval(fit, xvector)
        # Calculate the new sigma values as the median absolute deviation of
        # the residuals
        sig = np.nanmedian(np.abs(res))
        # Calculate the odds of being part of the "valid" values
        num = np.exp(-0.5 * (res / sig) ** 2) * (1 - odd_cut)
        # Calculate the odds of being an outlier
        den = odd_cut + num
        # Update the weights from the previous iteration
        weight_before = np.array(weight)
        # Calculate the new weights as the odds ratio that is fed back to
        # the fit
        weight = num / den
        # Increment the iteration counter
        count += 1
    # Set the mask of good values to be those for which there is a 50%
    # likelihood of being valid
    keep = np.array(weight > 0.5)
    # return the fit and keep vectors
    return fit, keep


def odd_ratio_mean(value, err, odd_ratio=1e-4, nmax=10):
    #
    # Provide values and corresponding errors and compute a
    # weighted mean
    #
    #
    # odd_bad -> probability that the point is bad
    #
    # nmax -> number of iterations
    keep = np.isfinite(value) * np.isfinite(err)

    if np.sum(keep) == 0:
        return np.nan, np.nan

    value = value[keep]
    err = err[keep]

    guess = np.nanmedian(value)
    odd_good = 1.0

    nite = 0
    while nite < nmax:
        nsig = (value - guess) / err
        gg = np.exp(-0.5 * nsig ** 2)
        odd_bad = odd_ratio / (gg + odd_ratio)
        odd_good = 1 - odd_bad

        w = odd_good / err ** 2

        guess = np.nansum(value * w) / np.nansum(w)
        nite += 1

    bulk_error = np.sqrt(1 / np.nansum(odd_good / err ** 2))

    return guess, bulk_error


def sigma(im):
    p1 = (1 - 0.682689492137086) / 2
    p2 = 1 - p1
    return (np.nanpercentile(im, p2 * 100) - np.nanpercentile(im, p1 * 100)) / 2


def lin_mini_errors(y0, yerr0, sample):
    y = y0.ravel()

    if sample.shape[1] == y.shape[0]:
        x = np.array(sample.T)
    else:
        x = np.array(sample)

    errs = yerr0.ravel()  # np.array(err[i]).ravel()

    # weights should be inverse of *square* error
    res_wls = sm.WLS(y, x, weights=1.0 / errs ** 2, missing='drop').fit()

    amps = res_wls.params
    errs = res_wls.bse

    recon = np.zeros_like(y0)
    for i in range(x.shape[1]):
        recon += amps[i] * x[:, i].reshape(y0.shape)

    return amps, errs, recon


def linear_minimization(vector, sample, mm, v, sz_sample, case, recon, amps):
    # raise ValueError(emsg.format(func_name))
    # ​
    # vector of N elements
    # sample: matrix N * M each M column is adjusted in amplitude to minimize
    # the chi2 according to the input vector
    # output: vector of length M gives the amplitude of each column
    #
    if case == 1:
        # fill-in the co-variance matrix
        for i in range(sz_sample[0]):
            for j in range(i, sz_sample[0]):
                mm[i, j] = np.sum(sample[i, :] * sample[j, :])
                # we know the matrix is symetric, we fill the other half
                # of the diagonal directly
                mm[j, i] = mm[i, j]
            # dot-product of vector with sample columns
            v[i] = np.sum(vector * sample[i, :])
        # if the matrix cannot we inverted because the determinant is zero,
        # then we return a NaN for all outputs
        if np.linalg.det(mm) == 0:
            amps = np.zeros(sz_sample[0]) + np.nan
            recon = np.zeros_like(v)
            return amps, recon

        # invert coveriance matrix
        inv = np.linalg.inv(mm)
        # retrieve amplitudes
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]

        # reconstruction of the best-fit from the input sample and derived
        # amplitudes
        for i in range(sz_sample[0]):
            recon += amps[i] * sample[i, :]
        return amps, recon

    if case == 2:
        # same as for case 1 but with axis flipped
        for i in range(sz_sample[1]):
            for j in range(i, sz_sample[1]):
                mm[i, j] = np.sum(sample[:, i] * sample[:, j])
                mm[j, i] = mm[i, j]
            v[i] = np.sum(vector * sample[:, i])

        if np.linalg.det(mm) == 0:
            return amps, recon

        inv = np.linalg.inv(mm)
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]

        for i in range(sz_sample[1]):
            recon += amps[i] * sample[:, i]
        return amps, recon


def lin_mini(vector, sample):
    # wrapper function that sets everything for the @jit later
    # In particular, we avoid the np.zeros that are not handled
    # by numba

    # size of input vectors and sample to be adjusted
    sz_sample = sample.shape  # 1d vector of length N
    sz_vector = vector.shape  # 2d matrix that is N x M or M x N

    case = -1
    # define which way the sample is flipped relative to the input vector
    if sz_vector[0] == sz_sample[0]:
        case = 2
    elif sz_vector[0] == sz_sample[1]:
        case = 1
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        print(emsg)

    # we check if there are NaNs in the vector or the sample
    # if there are NaNs, we'll fit the rest of the domain
    isnan = (np.sum(np.isnan(vector)) != 0) or (np.sum(np.isnan(sample)) != 0)

    # to avoid ambiguity regarding the existence of the variables
    mm, v, recon, amps, keep = None, None, None, None, None

    if case == 1:

        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=0))
            # redefine the input vector to avoid NaNs
            vector = vector[keep]
            sample = sample[:, keep]

            sz_sample = sample.shape

        # matrix of covariances
        mm = np.zeros([sz_sample[0], sz_sample[0]])
        # cross-terms of vector and columns of sample
        v = np.zeros(sz_sample[0])
        # reconstructed amplitudes
        amps = np.zeros(sz_sample[0])
        # reconstruted fit
        recon = np.zeros(sz_sample[1])

    if case == 2:
        # same as for case 1, but with axis flipped
        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=1))
            vector = vector[keep]
            sample = sample[keep, :]

            sz_sample = sample.shape

        mm = np.zeros([sz_sample[1], sz_sample[1]])
        v = np.zeros(sz_sample[1])
        amps = np.zeros(sz_sample[1])
        recon = np.zeros(sz_sample[0])

    # pass all variables and pre-formatted vectors to the @jit part of the code
    amp_out, recon_out = linear_minimization(vector, sample, mm, v, sz_sample, case,
                                             recon, amps)

    # if we had NaNs in the first place, we create a reconstructed vector
    # that has the same size as the input vector, but pad with NaNs values
    # for which we cannot derive a value
    if isnan:
        recon_out2 = np.zeros_like(keep) + np.nan
        recon_out2[keep] = recon_out
        recon_out = recon_out2

    return amp_out, recon_out
