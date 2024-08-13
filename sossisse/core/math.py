#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Define basic math functionality

Created on 2024-08-13

@author: cook
"""
from typing import Tuple, Union

import numpy as np
import statsmodels.api as sm
# noinspection PyPep8Naming
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import erf

from sossisse.core import base

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.core.math'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define functions
# =============================================================================
def lowpassfilter(input_vect: np.ndarray, width: int = 101) -> np.ndarray:
    """
    Computes a low-pass filter of an input vector. This is done while
    properly handling NaN values, but at the same time being reasonably fast.

    Algorithm:

    provide an input vector of an arbtrary length and compute a running NaN
    median over a box of a given length (width value). The running median is
    NOT computed at every pixel but at steps of 1/4th of the width value.
    This provides a vector of points where the nan-median has been computed
    (ymed) and mean position along the input vector (xmed) of valid (non-NaN)
    pixels. This xmed/ymed combination is then used in a spline to recover a
    vector for all pixel positions within the input vector.

    When there are no valid pixel in a 'width' domain, the value is skipped in
    the creation of xmed and ymed, and the domain is splined over.

    :param input_vect:
    :param width:
    :return:
    """
    # indices along input vector
    index = np.arange(len(input_vect))
    # placeholders for x and y position along vector
    xmed = []
    ymed = []
    # loop through the length of the input vector
    for it in np.arange(-width // 2, len(input_vect) + width // 2, width // 4):
        # if we are at the start or end of vector, we go 'off the edge' and
        # define a box that goes beyond it. It will lead to an effectively
        # smaller 'width' value, but will provide a consistent result at edges.
        low_bound = it
        high_bound = it + int(width)
        # deal with the edges
        if low_bound < 0:
            low_bound = 0
        if high_bound > (len(input_vect) - 1):
            high_bound = (len(input_vect) - 1)
        # get the pixels in the bounds
        pixval = index[low_bound:high_bound]
        # if we have less that 3 pixels, skip
        if len(pixval) < 3:
            continue
        # if no finite value, skip
        if np.max(np.isfinite(input_vect[pixval])) == 0:
            continue
        # mean position along vector and NaN median value of
        # points at those positions
        xmed.append(np.nanmean(pixval))
        ymed.append(np.nanmedian(input_vect[pixval]))
    # convert to numpy arrays
    xmed = np.array(xmed, dtype=float)
    ymed = np.array(ymed, dtype=float)
    # we need at least 3 valid points to return a
    # low-passed vector.
    if len(xmed) < 3:
        return np.zeros_like(input_vect) + np.nan
    # if we have duplicate x values, we average the y values
    if len(xmed) != len(np.unique(xmed)):
        xmed2 = np.unique(xmed)
        ymed2 = np.zeros_like(xmed2)
        for i in range(len(xmed2)):
            ymed2[i] = np.mean(ymed[xmed == xmed2[i]])
        xmed = xmed2
        ymed = ymed2
    # spline the vector
    spline = ius(xmed, ymed, k=2, ext=3)
    lowpass = spline(np.arange(len(input_vect)))
    # return the low pass
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
    # Enter a loop that will iterate until either the maximum difference
    # between the current and previous weights
    # becomes smaller than a certain threshold, or until the maximum number of
    # iterations is reached
    while (np.max(abs(weight - weight_before)) > 1e-9) and (count < nite_max):
        # Calculate the polynomial fit using the x- and y-values, and the
        # given degree, weighting the fit by the weights. Weights are computed
        # from the dispersion to the fit and the sigmax
        fit = np.polyfit(xvector, yvector, degree, w=weight)
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


def odd_ratio_mean(value: np.ndarray, error: np.ndarray,
                   odd_ratio: float = 2e-4, nmax: int = 10,
                   conv_cut=1e-2) -> Tuple[float, float]:
    """
    Provide values and corresponding errors and compute a weighted mean

    :param value: np.array (1D), value array
    :param error: np.array (1D), uncertainties for value array
    :param odd_ratio: float, the probability that the point is bad
                    Recommended value in Artigau et al. 2021 : f0 = 0.002
    :param nmax: int, maximum number of iterations to pass through
    :param conv_cut: float, the convergence cut criteria - how precise we have
                     to get

    :return: tuple, 1. the weighted mean, 2. the error on weighted mean
    """
    # deal with NaNs in value or error
    keep = np.isfinite(value) & np.isfinite(error)
    # deal with no finite values
    if np.sum(keep) == 0:
        return np.nan, np.nan
    # remove NaNs from arrays
    value, error = value[keep], error[keep]
    # work out some values to speed up loop
    error2 = error ** 2
    # placeholders for the "while" below
    guess_prev = np.inf
    # the 'guess' must be started as close as we possibly can to the actual
    # value. Starting beyond ~3 sigma (or whatever the odd_ratio implies)
    # would lead to the rejection of pretty much all points and would
    # completely mess the convergence of the loop
    guess = np.nanmedian(value)
    bulk_error = 1.0
    ite = 0
    # loop around until we do all required iterations
    while (np.abs(guess - guess_prev) / bulk_error > conv_cut) and (ite < nmax):
        # store the previous guess
        guess_prev = float(guess)
        # model points as gaussian weighted by likelihood of being a valid point
        # nearly but not exactly one for low-sigma values
        gfit = (1 - odd_ratio) * np.exp(-0.5 * ((value - guess) ** 2 / error2))
        # find the probability that a point is bad
        odd_bad = odd_ratio / (gfit + odd_ratio)
        # find the probability that a point is good
        odd_good = 1 - odd_bad
        # calculate the weights based on the probability of being good
        weights = odd_good / error2
        # update the guess based on the weights
        if np.sum(np.isfinite(weights)) == 0:
            guess = np.nan
        else:
            guess = np.nansum(value * weights) / np.nansum(weights)
            # work out the bulk error
            bulk_error = np.sqrt(1.0 / np.nansum(odd_good / error2))
        # keep track of the number of iterations
        ite += 1

    # return the guess and bulk error
    return guess, bulk_error


def normal_fraction(sigma: Union[float, np.ndarray] = 1.0
                    ) -> Union[float, np.ndarray]:
    """
    Return the expected fraction of population inside a range
    (Assuming data is normally distributed)

    :param sigma: the number of sigma away from the median to be
    :return:
    """
    # set function name
    # _ = display_func('normal_fraction', __NAME__)
    # return error function
    return erf(sigma / np.sqrt(2.0))


def estimate_sigma(tmp: np.ndarray, sigma=1.0) -> float:
    """
    Return a robust estimate of N sigma away from the mean

    :param tmp: np.array (1D) - the data to estimate N sigma of
    :param sigma: int, number of sigma away from mean (default is 1)

    :return: the sigma value
    """
    # get formal definition of N sigma
    sig1 = normal_fraction(sigma)
    # get the 1 sigma as a percentile
    p1 = (1 - (1 - sig1) / 2) * 100
    # work out the lower and upper percentiles for 1 sigma
    upper = np.nanpercentile(tmp, p1)
    lower = np.nanpercentile(tmp, 100 - p1)
    # return the mean of these two bounds
    return (upper - lower) / 2.0


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


def linear_minimization_internal(vector, sample, mm, v, sz_sample, case,
                                 recon, amps):
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


def linear_minimization(vector: np.ndarray, sample: np.ndarray,
                        no_recon: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    wrapper function that sets everything for the @jit later
    In particular, we avoid the np.zeros that are not handled
    by numba, size of input vectors and sample to be adjusted

    :param vector: 2d matrix that is (N x M) or (M x N)
    :param sample: 1d vector of length N
    :param no_recon: bool, if True does not calculate recon
    :return:
    """
    # set function name
    func_name = __NAME__ + 'linear_minimization()'
    # get sample and vector shapes
    sz_sample = sample.shape  # 1d vector of length N
    sz_vector = vector.shape  # 2d matrix that is N x M or M x N
    # define which way the sample is flipped relative to the input vector
    if sz_vector[0] == sz_sample[0]:
        case = 2
    elif sz_vector[0] == sz_sample[1]:
        case = 1
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        print(emsg)
        raise ValueError(emsg.format(func_name))
    # ----------------------------------------------------------------------
    # Part A) we deal with NaNs
    # ----------------------------------------------------------------------
    # set up keep vector
    keep = None
    # we check if there are NaNs in the vector or the sample
    # if there are NaNs, we'll fit the rest of the domain
    isnan = (np.sum(np.isnan(vector)) != 0) or (np.sum(np.isnan(sample)) != 0)
    # ----------------------------------------------------------------------
    # case 1: sample is not flipped relative to the input vector
    if case == 1:
        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=0))
            # redefine the input vector to avoid NaNs
            vector = vector[keep]
            sample = sample[:, keep]
            # re-find shapes
            sz_sample = sample.shape  # 1d vector of length N
        # matrix of covariances
        mm = np.zeros([sz_sample[0], sz_sample[0]])
        # cross-terms of vector and columns of sample
        vec = np.zeros(sz_sample[0])
        # reconstructed amplitudes
        amps = np.zeros(sz_sample[0])
        # reconstruted fit
        recon = np.zeros(sz_sample[1])
    # ----------------------------------------------------------------------
    # case 2: sample is flipped relative to the input vector
    elif case == 2:
        # same as for case 1, but with axis flipped
        if isnan:
            # we create a mask of non-NaN
            keep = np.isfinite(vector) * np.isfinite(np.sum(sample, axis=1))
            vector = vector[keep]
            sample = sample[keep, :]
            # re-find shapes
            sz_sample = sample.shape  # 1d vector of length N
        mm = np.zeros([sz_sample[1], sz_sample[1]])
        vec = np.zeros(sz_sample[1])
        amps = np.zeros(sz_sample[1])
        recon = np.zeros(sz_sample[0])
    # ----------------------------------------------------------------------
    # should not get here (so just repeat the raise from earlier)
    else:
        emsg = ('Neither vector[0]==sample[0] nor vector[0]==sample[1] '
                '(function = {0})')
        raise ValueError(emsg.format(func_name))

    # ----------------------------------------------------------------------
    # Part B) pass to optimized linear minimization
    # ----------------------------------------------------------------------
    # pass all variables and pre-formatted vectors to the @jit part of the code
    amp_out, recon_out = lin_mini(vector, sample, mm, vec, sz_sample,
                                  case, recon, amps, no_recon=no_recon)
    # ----------------------------------------------------------------------
    # if we had NaNs in the first place, we create a reconstructed vector
    # that has the same size as the input vector, but pad with NaNs values
    # for which we cannot derive a value
    if isnan:
        recon_out2 = np.zeros_like(keep) + np.nan
        recon_out2[keep] = recon_out
        recon_out = recon_out2

    return amp_out, recon_out


def lin_mini(vector: np.ndarray, sample: np.ndarray, mm: np.ndarray,
             v: np.ndarray, sz_sample: Tuple[int], case: int,
             recon: np.ndarray, amps: np.ndarray,
             no_recon: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear minimization of sample with vector

    Used internally in math.genearl.linear_minimization - you probably should
    use the linear_minization function instead of this directly

    :param vector: vector of N elements
    :param sample: sample: matrix N * M each M column is adjusted in
                   amplitude to minimize the chi2 according to the input vector
    :param mm: zero filled vector for filling size = M
    :param v: zero filled vector for filling size = M
    :param sz_sample: tuple the shape of the sample (N, M)
    :param case: int, if case = 1 then vector.shape[0] = sample.shape[1],
                 if case = 2 vector.shape[0] = sample.shape[0]
    :param recon: zero filled vector size = N, recon output
    :param amps: zero filled vector size = M, amplitudes output
    :param no_recon: boolean if True does not calculate recon
                     (output = input for recon)

    :returns: amps, recon
    """
    # do not set function name here -- cannot use functions here
    # case 1
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
        if not no_recon:
            for i in range(sz_sample[0]):
                recon += amps[i] * sample[i, :]
        return amps, recon
    # same as for case 1 but with axis flipped
    if case == 2:
        # fill-in the co-variance matrix
        for i in range(sz_sample[1]):
            for j in range(i, sz_sample[1]):
                mm[i, j] = np.sum(sample[:, i] * sample[:, j])
                # we know the matrix is symetric, we fill the other half
                # of the diagonal directly
                mm[j, i] = mm[i, j]
            # dot-product of vector with sample columns
            v[i] = np.sum(vector * sample[:, i])
        # if the matrix cannot we inverted because the determinant is zero,
        # then we return a NaN for all outputs
        if np.linalg.det(mm) == 0:
            return amps, recon
        # invert coveriance matrix
        inv = np.linalg.inv(mm)
        # retrieve amplitudes
        for i in range(len(v)):
            for j in range(len(v)):
                amps[i] += inv[i, j] * v[j]
        # reconstruction of the best-fit from the input sample and derived
        # amplitudes
        if not no_recon:
            for i in range(sz_sample[1]):
                recon += amps[i] * sample[:, i]
        return amps, recon


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # print 'Hello World!'
    print("Hello World!")

# =============================================================================
# End of code
# =============================================================================
