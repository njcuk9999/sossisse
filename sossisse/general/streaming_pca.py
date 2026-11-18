#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fit SOSSISSE's weighted EMPCA through bounded DataCube reads."""

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from wpca.empca import orthonormalize, random_orthonormal, solve_weighted

from sossisse.core import base
from sossisse.general.datacube import DataCube


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.streaming_pca'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def streaming_pca(
        datacube: DataCube, med: np.ndarray, trace_mask: np.ndarray,
        baseline_domain: slice | NDArray[Any], n_components: int,
        *, max_iter: int = 100, random_state: int | None = None
        ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stream the weighted EMPCA fit and SOSSISSE component reconstruction.

    :param datacube: bounded FLUX/ERR products in integration order
    :param med: reference detector image subtracted from each frame
    :param trace_mask: Boolean detector mask defining the fitted trace
    :param baseline_domain: integration slice, mask or ordered indices
    :param n_components: number of weighted components to fit
    :param max_iter: number of EM updates, matching wpca's default
    :param random_state: optional seed with the same meaning as wpca's seed
    :return: normalized component images and relative explained variances
    :raises ValueError: for mismatched shapes, empty baseline or invalid rank
    :raises numpy.linalg.LinAlgError: if a weighted solve is singular
    """
    shape = datacube.shape[1:]
    if med.shape != shape or trace_mask.shape != shape:
        raise ValueError('PCA median and trace mask must match the detector')
    selected = np.arange(len(datacube))[baseline_domain]
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError('PCA requires a nonempty one-dimensional baseline')
    if n_components < 1:
        raise ValueError('PCA requires a positive component count')
    nanmask = np.ones(shape)
    nanmask[~trace_mask] = np.nan

    def batches() -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Yield median-subtracted flux and errors in baseline order.

        :return: selected-row offset and two bounded flattened arrays
        """
        offset = 0
        while offset < selected.size:
            start = int(selected[offset])
            stop = offset + 1
            while (stop < selected.size
                   and stop - offset < datacube.max_integrations
                   and selected[stop] == start + stop - offset):
                stop += 1
            arrays = datacube.read_integrations(start, start + stop - offset)
            flux = arrays.pop('FLUX')
            error = arrays.pop('ERR')
            del arrays
            flux -= med
            yield offset, flux.reshape(len(flux), med.size), error.reshape(
                len(error), med.size)
            del flux, error
            offset = stop

    def prepare(flux: np.ndarray, error: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the legacy trace and finite-data masks.

        :param flux: flattened median-subtracted frames inside this batch
        :param error: matching uncertainties
        :return: masked data and inverse-error weights
        """
        data = flux * nanmask.ravel()
        bad = ~(np.isfinite(data) & np.isfinite(error))
        weights = 1 / error
        weights[bad] = 0
        data[bad] = 0
        return data, weights

    # wpca centers with inverse errors, but uses their squares in EM.
    counts = np.zeros(med.size, dtype=np.int64)
    weighted_sum = np.zeros(med.size)
    weight_sum = np.zeros(med.size)
    finite_weights = np.ones(med.size, dtype=bool)
    for _, flux, error in batches():
        with np.errstate(divide='ignore', invalid='ignore'):
            data, weights = prepare(flux, error)
            counts += np.count_nonzero(weights, axis=0)
            finite_weights &= np.all(np.isfinite(weights), axis=0)
            weighted_sum += np.einsum('ij,ij->j', data, weights)
            weight_sum += weights.sum(axis=0)
        del flux, error, data, weights
    valid = np.flatnonzero(counts / selected.size > 0.95)
    if valid.size < n_components:
        raise ValueError('Not enough pixels meet the PCA coverage threshold')
    if not np.all(finite_weights[valid]):
        raise ValueError('PCA weights must be finite on fitted pixels')
    mean = weighted_sum[valid] / weight_sum[valid]
    basis = random_orthonormal(n_components, valid.size,
                               random_state=random_state)
    scores = np.empty((selected.size, n_components))

    def centered_batches() -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        """Select and center the fitted detector pixels.

        :return: selected-row offset, centered data and inverse errors
        """
        for offset, flux, error in batches():
            with np.errstate(divide='ignore', invalid='ignore'):
                data, weights = prepare(flux, error)
            yield offset, data[:, valid] - mean, weights[:, valid]
            del flux, error, data, weights

    for _ in tqdm(range(max_iter), desc='Weighted PCA', unit='iteration'):
        # Component zero needs no deflation, so reuse the E-step reads.
        numerator = np.zeros(valid.size)
        denominator = np.zeros(valid.size)
        for offset, data, weights in centered_batches():
            for index in range(len(data)):
                scores[offset + index] = solve_weighted(
                    basis.T, data[index], weights[index])
            coeff = scores[offset:offset + len(data), :1]
            squared = weights ** 2
            numerator += (coeff.T @ (squared * data))[0]
            denominator += (coeff.T @ (squared * coeff))[0]
            del data, weights, squared
        basis[0] = numerator / denominator
        basis[:1] = orthonormalize(basis[:1])

        # Each component depends on the updated, orthonormalized prefix.
        for component in range(1, n_components):
            numerator = np.zeros(valid.size)
            denominator = np.zeros(valid.size)
            for offset, data, weights in centered_batches():
                batch_scores = scores[offset:offset + len(data)]
                residual = data - batch_scores[:, :component] @ basis[
                    :component]
                coeff = batch_scores[:, component:component + 1]
                squared = weights ** 2
                numerator += (coeff.T @ (squared * residual))[0]
                denominator += (coeff.T @ (squared * coeff))[0]
                del data, weights, residual, squared
            basis[component] = numerator / denominator
            basis[:component + 1] = orthonormalize(basis[:component + 1])

    # Collect final scores and variance alongside the legacy image synthesis.
    variance_sum = np.zeros(valid.size)
    variance_weights = np.zeros(valid.size)
    pcas = np.zeros((n_components,) + shape)
    for offset, flux, error in batches():
        with np.errstate(divide='ignore', invalid='ignore'):
            data, weights = prepare(flux, error)
        data = data[:, valid]
        weights = weights[:, valid]
        centered = data - mean
        variance_sum += ((centered * weights) ** 2).sum(axis=0)
        variance_weights += (weights ** 2).sum(axis=0)
        for index, frame in enumerate(centered):
            scores[offset + index] = solve_weighted(
                basis.T, frame, weights[index])
            # SOSSISSE projects uncentered data, not the EM scores, here.
            error_squared = error[index, valid] ** 2
            denominator = np.nansum(1 / error_squared)
            image = flux[index].reshape(shape)
            image[~np.isfinite(image)] = 0
            for component in range(n_components):
                amplitude = np.nansum(basis[component] * data[index]
                                      / error_squared) / denominator
                pcas[component] += amplitude * image
            del image
        del flux, error, data, weights, centered, frame
    ratios = (scores ** 2).sum(axis=0) / selected.size
    ratios /= np.sum(variance_sum / variance_weights)
    ratios /= ratios[0]
    reference_power = np.nansum(med ** 2 * nanmask)
    for image in pcas:
        for column in range(shape[1]):
            image[:, column] -= np.nanmedian(image[:, column])
        image /= np.sqrt(np.nansum(image ** 2 * nanmask) / reference_power)
    return pcas, ratios


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================