#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare full-cube NumPy reductions with sossisse.general.blockmath

Reports timing, peak allocated memory and fractional differences.

Usage:
    python comparison_block_math.py --path {DIR} --operation nanmedian
    python comparison_block_math.py --operation nanpercentile --percentile 16

Created on 2026-11-05

@author: cook
"""
import argparse
import glob
import os
import time
import tracemalloc
import warnings
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from sossisse.general import blockmath

# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'comparison_block_math.py'
# default input directory
PATH = '/scratch3/jwst/misc/large_rate_ints_test'
# default glob pattern for the input segments
PATTERN = '*.fits'
# default FITS extension holding the science cube
EXTENSION = 1
# reductions supported by both NumPy and blockmath
OPERATIONS = ['nanmean', 'nanmedian', 'nansum', 'nanpercentile']
# default reduction to compare
OPERATION = 'nanpercentile'
# default percentile (only used for nanpercentile)
PERCENTILE = 50.0
# default blockmath memory budget in GiB
MAX_MEMORY_GB = 32.0
# bytes per GiB (used for memory reporting)
GIB = 1024 ** 3


# =============================================================================
# Define functions
# =============================================================================
def get_args() -> argparse.Namespace:
    """
    Parse the command-line arguments

    :return: argparse.Namespace, the parsed arguments
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--path', type=str, default=PATH,
                        help='Directory containing the input segments')
    parser.add_argument('--pattern', type=str, default=PATTERN,
                        help='Glob pattern selecting input segments')
    parser.add_argument('--extension', type=int, default=EXTENSION,
                        help='FITS extension number of the science cube')
    parser.add_argument('--operation', type=str, default=OPERATION,
                        choices=OPERATIONS, help='Reduction to compare')
    parser.add_argument('--percentile', type=float, default=PERCENTILE,
                        help='Percentile in [0, 100] (nanpercentile only)')
    parser.add_argument('--max-memory-gb', type=float, default=MAX_MEMORY_GB,
                        help='blockmath memory budget in GiB')
    parser.add_argument('--no-plot', action='store_true',
                        help='Do not display the difference image')
    return parser.parse_args()


def get_filenames(path: str, pattern: str) -> List[str]:
    """
    Find the input segments

    :param path: str, directory to search
    :param pattern: str, glob pattern for the input segments

    :raises FileNotFoundError: if no files match
    :return: list of str, the sorted filenames (sorted to fix segment order)
    """
    filenames = sorted(glob.glob(os.path.join(path, pattern)))
    if len(filenames) == 0:
        emsg = 'No files found matching {0}'
        raise FileNotFoundError(emsg.format(os.path.join(path, pattern)))
    return filenames


def profile(func: Callable[[], np.ndarray]) -> Tuple[np.ndarray, float,
                                                       float]:
    """
    Run a function, measuring wall time and peak allocated memory

    tracemalloc tracks NumPy allocations, but not memory held by Python
    before tracing starts, so this is the peak *additional* allocation.

    :param func: callable with no arguments returning the result array

    :return: tuple, 1. the result, 2. wall time in s, 3. peak memory in GiB
    """
    tracemalloc.start()
    start = time.time()
    result = func()
    end = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end - start, peak / GIB


def old_way(filenames: List[str], extension: int, operation: str,
            percentile: float) -> np.ndarray:
    """
    Reduce the full cube in memory with NumPy

    Cast to float64 to match blockmath (raw rateints are float32).

    :param filenames: list of str, the input segments in time order
    :param extension: int, FITS extension number of the science cube
    :param operation: str, the NumPy reduction name
    :param percentile: float, percentile in [0, 100] (nanpercentile only)

    :return: np.ndarray, the reduced 2D image
    """
    cube = [fits.getdata(filename, ext=extension) for filename in filenames]
    cube = np.vstack(cube).astype(np.float64)
    math_func = getattr(np, operation)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if operation == 'nanpercentile':
            return math_func(cube, percentile, axis=0)
        return math_func(cube, axis=0)


def new_way(filenames: List[str], extension: int, operation: str,
            percentile: float, max_memory_gb: float) -> np.ndarray:
    """
    Reduce the cube in memory-bounded tiles with blockmath

    :param filenames: list of str, the input segments in time order
    :param extension: int, FITS extension number of the science cube
    :param operation: str, the blockmath reduction name
    :param percentile: float, percentile in [0, 100] (nanpercentile only)
    :param max_memory_gb: float, blockmath memory budget in GiB

    :return: np.ndarray, the reduced 2D image
    """
    shape, reader = blockmath.sample_reader(filenames, extension=extension)
    bm_kwargs = dict(axis=0, max_memory_gb=max_memory_gb)
    if operation == 'nanpercentile':
        bm_kwargs['q'] = percentile
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return blockmath.block_math(operation, shape, reader, **bm_kwargs)


def main():
    """
    Compare the old (full cube) and new (blockmath) reductions

    :return: None, prints timing/memory/deviations and optionally plots
    """
    args = get_args()
    filenames = get_filenames(args.path, args.pattern)
    operation = args.operation
    percentile = args.percentile
    if operation == 'nanpercentile':
        if not 0 <= percentile <= 100:
            emsg = '--percentile must be in [0, 100], got {0}'
            raise ValueError(emsg.format(percentile))
        label = f'{operation} (q={percentile})'
    else:
        label = operation
    print(f'Found {len(filenames)} files in {args.path}')
    # ---------------------------------------------------------------------
    # old way: full cube in memory
    print(f'Old {label}')
    old_args = [filenames, args.extension, operation, percentile]
    old_outs = profile(lambda: old_way(*old_args))
    result_old, old_time, old_memory = old_outs
    # ---------------------------------------------------------------------
    # new way: blockmath tiles
    print(f'New {label}')
    new_args = [filenames, args.extension, operation, percentile,
                args.max_memory_gb]
    new_outs = profile(lambda: new_way(*new_args))
    result_new, new_time, new_memory = new_outs
    # ---------------------------------------------------------------------
    print(f'Old timing: {old_time:.3f} s, peak memory: {old_memory:.3f} GiB')
    print(f'New timing: {new_time:.3f} s, peak memory: {new_memory:.3f} GiB')
    # ---------------------------------------------------------------------
    diff = result_old - result_new
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        frac = diff / result_old
        print('Max deviation: ', np.nanmax(np.abs(frac)))
        print('Median deviation: ', np.nanmedian(np.abs(frac)))
        print('Mean deviation: ', np.nanmean(np.abs(frac)))
    # ---------------------------------------------------------------------
    if args.no_plot:
        return
    plt.imshow(frac, origin='lower', aspect='auto', interpolation='None')
    plt.title(f'{label} fractional difference diff/result')
    plt.colorbar()
    plt.show()


# =============================================================================
# Start of code
# =============================================================================
if __name__ == "__main__":
    main()

# =============================================================================
# End of code
# =============================================================================