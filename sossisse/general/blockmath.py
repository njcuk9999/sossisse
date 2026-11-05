#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute exact, memory-budgeted reductions through a storage-free reader.

``block_math`` knows only a three-axis shape and a callback, not filenames,
file formats or named data products. ``sample_reader`` is an optional FITS
reference adapter for testing; only that factory imports Astropy FITS I/O.
"""

from bisect import bisect_right
from collections.abc import Callable, Sequence
import itertools
import math
from numbers import Real
from os import PathLike
import resource
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import psutil
from tqdm import tqdm

from aperocore.core import drs_log
from sossisse.core import base


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.blockmath'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# get the logger
WLOG = drs_log.wlog
# number of reader calls per block (sets progress bar resolution)
READ_STEPS = 100
# total memory as a multiple of the float64 block (block + NumPy scratch);
# measured with NumPy 2.3 as about 2.1, 2.3, 4.4 and 1.0, plus a margin
MEMORY_FACTORS = dict(nansum=3.0, nanmean=3.0, nanmedian=5.0,
                      nanpercentile=2.0)


# =============================================================================
# Define classes
# =============================================================================


# =============================================================================
# Define functions
# =============================================================================
def block_math(
        operation: str, source_shape: tuple[int, int, int],
        reader: Callable[[NDArray[np.intp], tuple[slice, ...]],
                         NDArray[np.float64]],
        axis: int | tuple[int, ...] | None = 0,
        selection: slice | Sequence[int] | NDArray[Any] | None = None,
        max_memory_gb: float = 8.0, *,
        q: float | Sequence[float] | NDArray[Any] | None = None,
        keepdims: bool = False, method: str = 'linear'
        ) -> NDArray[np.float64]:
    """Reduce complete selected domains while tiling only unreduced axes.

    Every tile retains all selected samples along each reduced axis, in
    their original selected order. NumPy performs the actual reduction;
    partial means, medians or percentiles are never combined. This preserves
    exact mathematical results, NaN/infinity semantics and duplicate weights
    rather than introducing a streaming or approximate statistic.

    Reader contract
    ---------------
    The engine calls ``reader(indices, axis_slices)`` with exactly two
    positional arguments. ``indices`` is a one-dimensional ``np.intp`` array
    of absolute source-axis-0 coordinates, already normalized to
    ``0 <= index < source_shape[0]``. Preserve their order and duplicates;
    do not sort, deduplicate or apply ``selection`` again. ``axis_slices``
    is exactly two slices, for source axes 1 and 2, respectively. Their
    integer start/stop bounds are explicit, nonnegative and within the
    source shape; stops are exclusive and steps are None (unit stride).

    Return an owned float64 NumPy array with shape
    ``(len(indices), axis_slices[0].stop - axis_slices[0].start,
    axis_slices[1].stop - axis_slices[1].start)``. For example, a source of
    shape ``(6, 4, 5)``, indices ``[4, 0, 4]`` and slices ``(slice(1, 3),
    slice(0, 2))`` require a ``(3, 2, 2)`` tile in that exact index order.
    Axis-0 selection applies before reduction, regardless of ``axis``.
    Each block may be requested in several consecutive axis-0 pieces (up to
    ``READ_STEPS`` calls per block); the engine copies each piece into its
    own block array. The engine makes no calls for empty sources or
    selections and does not cache returned tiles. Do not mutate callback
    arguments,
    retain tiles or return arrays backed by closed storage handles. Read
    only the requested data with bounded temporary allocations; allocating
    the full source inside a reader defeats the memory estimate.

    A testing-only FITS adapter can be bound once and reused::

        shape, reader = sample_reader(files, extension='SCI')
        result = block_math('nanmedian', shape, reader, axis=0)

    The estimate includes final output, selection/index metadata and the
    block plus NumPy scratch, as ``MEMORY_FACTORS`` times the float64 block
    size. These factors were measured with NumPy 2.3 plus a safety margin.
    It is not a hard RSS cap: Python overhead, allocator caching and library
    internals remain outside this estimate. If the measured peak process
    memory during the call exceeds ``max_memory_gb``, a warning is logged.
    Readers must bound their own I/O and avoid
    retaining tiles; their backend-specific read limits are not engine state.

    :param operation: nansum, nanmean, nanmedian or nanpercentile
    :param source_shape: nonnegative lengths of source axes (0, 1, 2);
                         zero lengths are allowed for empty-source behavior
    :param reader: storage-independent callback following the reader contract
                   above; source access and ownership belong to the callback
    :param axis: reduced axes in the selected 3-D source; negative axes,
                 tuples (including empty) and None follow NumPy
    :param selection: axis-0 slice, Boolean mask of source axis-0 length, or
                      integer vector; preserve order and duplicates, allow
                      negative indices and empty selections
    :param max_memory_gb: positive finite real allocation budget in GiB
                         (1024**3 bytes), not gigabits; default 8.0 GiB;
                         includes output; not a Boolean
    :param q: required scalar or 1-D percentiles in [0, 100]; vector
              quantiles prepend an output axis, as in NumPy
    :param keepdims: retain reduced axes with length one
    :param method: NumPy percentile interpolation method, default linear
    :return: float64 result, including zero-dimensional scalar arrays;
             empty shapes and warnings follow NumPy without reader calls
    :raises ValueError: for invalid shape, budget, operation, selection,
                        duplicate axes, percentile options or reader tile
    :raises TypeError: if reader is not callable
    :raises IndexError: for out-of-range indices or axes
    :raises MemoryError: before reader calls if output plus the smallest
                         complete reduction tile cannot fit the budget
    :raises Exception: propagate reader exceptions without wrapping them
    """
    # Dispatch only known non-mutating reductions, never arbitrary callables.
    reducers = dict(nansum=np.nansum, nanmean=np.nanmean,
                    nanmedian=np.nanmedian, nanpercentile=np.nanpercentile)
    if operation not in reducers:
        raise ValueError(f'Unsupported block_math operation: {operation}')
    # Shape validation needs no source access and permits empty managers.
    if (not isinstance(source_shape, tuple) or len(source_shape) != 3
            or any(isinstance(length, (bool, np.bool_))
                   or not isinstance(length, (int, np.integer))
                   or length < 0 for length in source_shape)):
        raise ValueError('source_shape must be three nonnegative integers')
    # Built-in lengths avoid overflow from NumPy integer shape arithmetic.
    count, axis1_length, axis2_length = (int(length)
                                        for length in source_shape)
    if not callable(reader):
        raise TypeError('reader must be callable')
    # Reusable callers need the same strict budget contract as adapters.
    if (isinstance(max_memory_gb, (bool, np.bool_))
            or not isinstance(max_memory_gb, Real)):
        raise ValueError('max_memory_gb must be positive and finite')
    try:
        memory = float(max_memory_gb)
    except OverflowError as error:
        raise ValueError('max_memory_gb is too large') from error
    if not math.isfinite(memory) or memory <= 0:
        raise ValueError('max_memory_gb must be positive and finite')
    # Normalize negative axes and reject duplicates before planning any reads.
    axes = _math_axes(axis)
    # Integer-ratio conversion avoids overflow or rounding a large GiB value.
    numerator, denominator = memory.as_integer_ratio()
    budget = numerator * 1024 ** 3 // denominator
    # Reserve conservative source/index metadata before allocating a map.
    if 64 * count > budget:
        raise MemoryError('Math budget cannot hold selection metadata')
    indices = _math_indices(selection, count, budget // 64 - count)
    metadata_bytes = 64 * (count + indices.size)
    quantiles: NDArray[np.float64] | None = None
    if operation == 'nanpercentile':
        if q is None:
            raise ValueError('nanpercentile requires q')
        # Bound quantile conversion and validation masks before making copies.
        quantile_count = 1
        if isinstance(q, np.ndarray):
            quantile_count = q.size
        elif isinstance(q, Sequence):
            quantile_count = len(q)
        if metadata_bytes + 32 * quantile_count > budget:
            raise MemoryError('Math budget cannot hold quantile metadata')
        quantiles = np.asarray(q, dtype=np.float64)
        if (quantiles.ndim > 1 or not np.all(np.isfinite(quantiles))
                or np.any((quantiles < 0) | (quantiles > 100))):
            raise ValueError('q must be scalar or 1-D in [0, 100]')
        # Let NumPy validate its method names without reading source samples.
        method_kwargs: dict[str, Any] = dict(method=method)
        np.nanpercentile(np.ones(1), 50., **method_kwargs)
        metadata_bytes += 32 * quantiles.size
    elif q is not None or method != 'linear':
        raise ValueError('q/method are only valid for nanpercentile')
    # Selection changes axis-0 length, independently of the reduction axes.
    selected_shape = (indices.size, axis1_length, axis2_length)
    output_shape = tuple(1 if index in axes else length
                         for index, length in enumerate(selected_shape)
                         if keepdims or index not in axes)
    # NumPy places a vector quantile dimension before all surviving axes.
    if quantiles is not None and quantiles.ndim == 1:
        output_shape = (quantiles.size,) + output_shape
    output_bytes = 8 * math.prod(output_shape)
    factor = MEMORY_FACTORS[operation]
    tile_shape = list(selected_shape)

    def estimated_bytes() -> int:
        """Estimate simultaneous final, tile and reduction storage.

        :return: bytes for the current candidate tile and fixed allocations
        """
        # Reduced axes produce one output location even with keepdims=True.
        tile_output = math.prod(size for index, size in enumerate(tile_shape)
                                if index not in axes)
        if quantiles is not None and quantiles.ndim == 1:
            tile_output *= quantiles.size
        return math.ceil(metadata_bytes + output_bytes
                         + 8 * factor * math.prod(tile_shape)
                         + 16 * tile_output)

    # Shrink only surviving axes: splitting a reduced axis would require
    # combining partial statistics and would lose the exact NumPy behavior.
    fixed_bytes = metadata_bytes + output_bytes
    while estimated_bytes() > budget:
        candidates = [index for index in range(3)
                      if index not in axes and tile_shape[index] > 1]
        if not candidates:
            raise MemoryError(
                'Math budget cannot hold final output plus one complete '
                'reduced domain and its NumPy workspace; increase '
                'max_memory_gb')
        largest = max(candidates, key=lambda index: tile_shape[index])
        length = tile_shape[largest]
        # The estimate is linear in each surviving axis length.
        tile_shape[largest] = 1
        per_unit = estimated_bytes() - fixed_bytes
        fit = max(1, (budget - fixed_bytes) // per_unit)
        # Equal-sized blocks with the fewest splits that fit the budget.
        n_split = math.ceil(length / fit)
        tile_shape[largest] = math.ceil(length / n_split)
    reduce_kwargs: dict[str, Any] = dict(axis=axes, keepdims=keepdims)
    if quantiles is not None:
        reduce_kwargs.update(q=quantiles, method=method)
    reducer = reducers[operation]
    # Empty reductions have operation-specific values, shapes and warnings.
    # Calling NumPy directly also avoids zero-step tile loops or reader I/O.
    if math.prod(selected_shape) == 0:
        return np.asarray(reducer(np.empty(selected_shape), **reduce_kwargs),
                          dtype=np.float64)
    start_rss = psutil.Process().memory_info().rss
    start_peak = _peak_rss_bytes()
    output = np.empty(output_shape, dtype=np.float64)
    # This Cartesian grid covers unreduced coordinates without overlap;
    # reduced axes have one full-length tile, so each output is computed once.
    tile_starts = [range(0, length, size)
                   for length, size in zip(selected_shape, tile_shape)]
    n_blocks = math.prod(len(starts) for starts in tile_starts)
    # Axis-0 tiles partition the selection, so this counts every read slice.
    n_reads = (selected_shape[0] * len(tile_starts[1])
               * len(tile_starts[2]))
    tile_text = ' x '.join(str(size) for size in tile_shape)
    msg = ('Calculating the {0} on {1} block(s) of up to {2} '
           '({3} axis-0 slices to read)')
    margs = [operation, n_blocks, tile_text, n_reads]
    WLOG(None, '', msg.format(*margs))
    blocks = enumerate(itertools.product(*tile_starts))
    with tqdm(total=n_reads, unit='slice') as progress:
        for block_number, starts in blocks:
            progress.set_description(f'Block {block_number + 1}/{n_blocks}')
            slices = tuple(slice(start, min(start + size, length))
                           for start, size, length in
                           zip(starts, tile_shape, selected_shape))
            expected_shape = tuple(part.stop - part.start for part in slices)
            block_indices = indices[slices[0]]
            tile = np.empty(expected_shape, dtype=np.float64)
            step = max(1, math.ceil(block_indices.size / READ_STEPS))
            for start in range(0, block_indices.size, step):
                stop = min(start + step, block_indices.size)
                # The callback owns storage access; selected axis-0 order is
                # explicit, while axes 1 and 2 remain simple slices.
                piece = reader(block_indices[start:stop], slices[1:])
                piece_shape = (stop - start,) + expected_shape[1:]
                if (not isinstance(piece, np.ndarray)
                        or piece.dtype != np.dtype(np.float64)
                        or piece.shape != piece_shape):
                    raise ValueError('reader must return a float64 tile '
                                     'with the requested shape')
                tile[start:stop] = piece
                del piece
                progress.update(stop - start)
            reduced = reducer(tile, **reduce_kwargs)
            # Drop reduced coordinates, or address their singleton axes
            # with full slices when keepdims retains them in the output.
            target = tuple(slice(None) if index in axes else part
                           for index, part in enumerate(slices)
                           if keepdims or index not in axes)
            if quantiles is not None and quantiles.ndim == 1:
                target = (slice(None),) + target
            output[target] = reduced
            # Release both temporaries before the next reader allocation.
            del tile, reduced
    # The process peak only moves if this call set a new high-water mark,
    # so an earlier larger peak can hide an overrun (no false warnings).
    end_peak = _peak_rss_bytes()
    used = end_peak - start_rss
    if end_peak > start_peak and used > budget:
        wmsg = ('block_math {0}: peak memory increase {1:.2f} GiB exceeded '
                'max_memory_gb={2:.2f} GiB; MEMORY_FACTORS may be too low '
                'for this NumPy version')
        wargs = [operation, used / 1024 ** 3, memory]
        WLOG(None, 'warning', wmsg.format(*wargs))
    return output


def sample_reader(
        filenames: Sequence[str | PathLike[str]], extension: int | str = 0
        ) -> tuple[tuple[int, int, int],
                   Callable[[NDArray[np.intp], tuple[slice, ...]],
                            NDArray[np.float64]]]:
    """Bind a simple FITS reference reader for tests, not production I/O.

    The selected images form one logical three-axis source, concatenated
    along axis 0 in supplied filename order. No combined array is created.
    Only headers are inspected by the factory; axis-0 lengths may differ,
    but axes 1 and 2 must agree. Header identifiers do not affect ordering.
    There are no assumptions about other extensions or named data products.

    The returned callback implements the ``block_math`` reader contract.
    It allocates one owned float64 output tile and reads one requested
    axis-0 section at a time, preserving arbitrary order and duplicates.
    Each section opens and closes its file, including on failure. No open
    handles or image arrays are retained between calls. This deliberately
    simple implementation favors clarity over production throughput.
    Files must remain unchanged while the returned reader is in use.

    Usage::

        shape, reader = sample_reader(['first.fits', 'second.fits'])
        result = block_math('nanmean', shape, reader, axis=0)

    :param filenames: nonempty sequence of FITS paths in concatenation order
    :param extension: explicit image extension index or name, default 0
    :return: combined source shape and reusable ``reader(indices,
             axis_slices)`` callback; indices and slices must already satisfy
             the engine contract; empty indices return an empty float64 tile
    :raises ValueError: for an empty file sequence, non-3-D or empty images,
                        non-image extensions or mismatched axes 1 and 2
    :raises OSError: propagate file access and FITS read failures
    :raises KeyError: if a named extension is missing
    :raises IndexError: if an extension index is missing
    :raises ImportError: if the optional Astropy dependency is unavailable
    """
    from astropy.io import fits

    image_types = (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
    paths = tuple(filenames)
    if not paths:
        raise ValueError('filenames must be nonempty')
    metadata: list[tuple[str | PathLike[str], int, int]] = []
    stops: list[int] = []
    trailing_shape: tuple[int, int] | None = None
    total = 0
    for filename in paths:
        with fits.open(filename, memmap=False) as hdus:
            extension_index = int(hdus.index_of(extension))
            image = hdus[extension_index]
            if not isinstance(image, image_types):
                raise ValueError('sample_reader requires image extensions')
            header = image.header
            if header.get('NAXIS') != 3:
                raise ValueError('sample_reader requires 3-D images')
            lengths: list[int] = []
            for keyword in ('NAXIS3', 'NAXIS2', 'NAXIS1'):
                length = header[keyword]
                if (isinstance(length, (bool, np.bool_))
                        or not isinstance(length, (int, np.integer))
                        or length <= 0):
                    raise ValueError('sample_reader requires nonempty images')
                lengths.append(int(length))
            image_shape = (lengths[0], lengths[1], lengths[2])
            if trailing_shape is None:
                trailing_shape = image_shape[1:]
            elif image_shape[1:] != trailing_shape:
                raise ValueError('sample_reader axes 1 and 2 must match')
        metadata.append((filename, extension_index, total))
        total += image_shape[0]
        stops.append(total)
    assert trailing_shape is not None
    source_shape = (total, trailing_shape[0], trailing_shape[1])

    def read_tile(indices: NDArray[np.intp], axis_slices: tuple[slice, ...]
                  ) -> NDArray[np.float64]:
        """Read a bounded tile using normalized engine coordinates.

        :param indices: valid absolute axis-0 indices, ordered with duplicates
        :param axis_slices: two explicit bounded unit-stride slices
        :return: owned float64 tile, including for empty indices
        :raises OSError: propagate file access and FITS section read failures
        """
        tile_shape = (len(indices),
                      axis_slices[0].stop - axis_slices[0].start,
                      axis_slices[1].stop - axis_slices[1].start)
        tile = np.empty(tile_shape, dtype=np.float64)
        for output_index, source_index in enumerate(indices):
            file_index = bisect_right(stops, int(source_index))
            filename, extension_index, start = metadata[file_index]
            local_index = int(source_index) - start
            with fits.open(filename, memmap=False) as hdus:
                image = hdus[extension_index]
                if not isinstance(image, image_types):
                    raise ValueError('sample_reader requires image extensions')
                section_key = (local_index,) + axis_slices
                tile[output_index] = image.section[section_key]
        return tile

    return source_shape, read_tile


# =============================================================================
# Define worker functions
# =============================================================================
def _peak_rss_bytes() -> int:
    """Return the peak resident memory of this process so far.

    :return: int, the high-water mark in bytes (ru_maxrss is KiB on Linux)
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return int(peak)
    return int(peak) * 1024


def _math_axes(axis: int | tuple[int, ...] | None) -> tuple[int, ...]:
    """Validate and normalize reduced axes of a three-dimensional source.

    :param axis: integer, tuple of integers or None for all axes
    :return: nonnegative unique axes, retaining the supplied order
    :raises ValueError: for Boolean/noninteger or duplicate axes
    :raises IndexError: for axes outside [-3, 2]
    """
    if axis is None:
        return (0, 1, 2)
    raw_axes = axis if isinstance(axis, tuple) else (axis,)
    normalized: list[int] = []
    for value in raw_axes:
        # Booleans are integer subclasses, but never meaningful axis names.
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))):
            raise ValueError('axis must contain integers')
        if not -3 <= value < 3:
            raise np.exceptions.AxisError(value, ndim=3)
        normalized.append(int(value) % 3)
    # Normalize first so aliases such as (0, -3) count as duplicates.
    if len(set(normalized)) != len(normalized):
        raise ValueError('duplicate axes are not allowed')
    return tuple(normalized)


def _math_indices(selection: slice | Sequence[int] | NDArray[Any] | None,
                  count: int, max_indices: int) -> NDArray[np.intp]:
    """Validate axis-0 selection without reading source samples.

    :param selection: None, slice, 1-D Boolean mask or integer vector
    :param count: active source axis-0 length
    :param max_indices: maximum selected count after reserving math metadata
    :return: owned axis-0 indices with NumPy order/duplicates preserved
    :raises ValueError: for malformed selection or wrong Boolean length
    :raises IndexError: for out-of-range integer indices
    :raises MemoryError: before allocating an oversized selection map
    """
    def check_size(size: int) -> None:
        """Bound selected metadata before allocating an owned index map.

        :param size: proposed selected axis-0 length
        :return: None if the selection metadata fits
        :raises MemoryError: if the selection exceeds its reserved allowance
        """
        if size > max_indices:
            raise MemoryError('Math budget cannot hold selection metadata')

    if selection is None:
        check_size(count)
        return np.arange(count, dtype=np.intp)
    if isinstance(selection, slice):
        # slice.indices preserves clipping, reverse order and zero-step errors.
        bounds = selection.indices(count)
        check_size(len(range(*bounds)))
        return np.arange(*bounds, dtype=np.intp)
    # A Python sequence needs a temporary array before dtype validation.
    if isinstance(selection, Sequence):
        check_size(len(selection))
    values = np.asarray(selection)
    if values.ndim != 1:
        raise ValueError('selection must be a one-dimensional vector')
    if values.dtype.kind == 'b':
        if values.size != count:
            raise ValueError('Boolean selection must match axis-0 length')
        check_size(int(np.count_nonzero(values)))
        return np.flatnonzero(values)
    # An empty Python vector conventionally selects no axis-0 samples.
    if values.size == 0:
        return np.empty(0, dtype=np.intp)
    if values.dtype.kind not in 'iu':
        raise ValueError('selection must contain Boolean or integer values')
    if np.any(values >= count) or np.any(values < -count):
        raise IndexError('selection axis-0 index is out of bounds')
    check_size(values.size)
    # Copy before translating negatives so the caller's selection is unchanged.
    indices = values.astype(np.intp, copy=True)
    indices[indices < 0] += count
    return indices


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================