#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manage bounded, aligned SCI/ERR/DQ FITS products on disk."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile
from typing import cast
from typing import Self
from typing import Any

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits

from sossisse.core import base
from sossisse.general import blockmath


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.datacube'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__


# =============================================================================
# Define classes
# =============================================================================
@dataclass(frozen=True)
class _RawFile:
    """Store immutable raw-segment metadata without retaining image arrays.

    :param filename: source path, retained in its original path type
    :param start: positive, inclusive first raw integration identifier
    :param end: inclusive last identifier; ``end - start + 1 == shape[0]``
    :param shape: NumPy axis lengths, ``(integrations, y, x)`` or
                  ``(integrations, reads, y, x)`` for CDS data
    :param extensions: one SCI index for CDS/synthetic errors, otherwise
                       SCI, ERR and DQ indices in that order
    :param bounds: clipped detector-column bounds ``(start, stop)``, with
                   an exclusive stop and a nonempty selected width
    """

    # Accept both string filenames and pathlib paths without opening data.
    filename: str | os.PathLike[str]
    # These are inclusive integration identifiers from the primary header.
    start: int
    end: int
    # Preserve the raw NumPy axis order, including reads for CDS inputs.
    shape: tuple[int, ...]
    # Store numeric SCI/ERR/DQ HDU indices resolved during header inspection.
    extensions: tuple[int, ...]
    # Selected detector columns use Python's exclusive upper bound.
    bounds: tuple[int, int]


class DataCube:
    """Own active SCI/ERR/DQ product filenames, never a resident cube.

    ``current_infiles`` contains active product paths as strings. Acquisition
    ``filenames`` and inclusive ``integration_ranges`` are header-sorted and
    populated only by normalization; reopening products leaves them empty.
    ``flag_cds`` records whether normalization converted raw read axes.
    ``shape`` is ``(selected binned integrations, y, cropped x)``, initially
    ``(0, 0, 0)``. ``max_integrations`` bounds product sizes and explicit
    reads.

    :param current_infiles: ordered normalized product paths to validate and
                            commit, or None to create an empty manager
    :param max_integrations: positive integer integration limit; not a Boolean
    :param max_math_memory_gb: positive finite estimated math budget in GiB
                              (1024**3 bytes), not gigabits; includes final
                              output; default 8.0 GiB
    :raises ValueError: if the limit or supplied product stage is invalid
    :raises OSError: if an initial product cannot be opened
    """

    def __init__(self,
                 current_infiles: Iterable[str | os.PathLike[str]] | None
                 = None,
                 max_integrations: int | np.integer[Any] = 1000,
                 max_math_memory_gb: float = 8.0
                 ) -> None:
        """Initialize metadata and optionally reopen a complete product stage.

        :param current_infiles: ordered normalized paths, or None for an
                                empty manager; images are not loaded
        :param max_integrations: positive, non-Boolean integration limit
        :param max_math_memory_gb: positive finite math budget in GiB
                  (1024**3 bytes), not gigabits; default 8.0 GiB;
                  includes final output and estimated scratch
        :return: None; initialize state and commit supplied valid products
        :raises ValueError: if the limit, products or coverage are invalid
        :raises OSError: if supplied products cannot be opened
        """
        # Reject bools/floats rather than silently changing the memory budget.
        self.max_integrations: int = _positive_integer(max_integrations,
                                                       'max_integrations')
        # Validate before coercion so Boolean/string configuration is rejected.
        if (isinstance(max_math_memory_gb, (bool, np.bool_))
                or not isinstance(max_math_memory_gb, Real)):
            raise ValueError('max_math_memory_gb must be positive and finite')
        # A mathematically finite integer may overflow the float storage type.
        try:
            memory = float(max_math_memory_gb)
        except OverflowError as error:
            raise ValueError('max_math_memory_gb is too large') from error
        if not math.isfinite(memory) or memory <= 0:
            raise ValueError('max_math_memory_gb must be positive and finite')
        self.max_math_memory_gb: float = memory
        # An empty manager is populated by normalize, not a reader closure.
        self.current_infiles: list[str] = []
        # Acquisition metadata remains independent of the current stage.
        self.filenames: tuple[str | os.PathLike[str], ...] = ()
        self.integration_ranges: tuple[tuple[int, int], ...] = ()
        self.flag_cds: bool = False
        self.shape: tuple[int, int, int] = (0, 0, 0)
        # Only small metadata records remain resident between reads.
        self._products: list[_Product] = []
        if current_infiles is not None:
            # Validate the entire list before publishing any filenames.
            products = [_inspect_product(path, self.max_integrations)
                        for path in current_infiles]
            self.commit_products(products)

    def __len__(self) -> int:
        """Return the number of selected, binned integrations.

        :return: nonnegative integration count, zero before products are
                 committed; this is not the number of source files
        """
        # Length counts selected integrations rather than source files.
        return self.shape[0]

    @staticmethod
    def input_n_integrations(filenames: Iterable[str | os.PathLike[str]],
                             bin_size: int | np.integer[Any] = 1
                             ) -> int:
        """Count complete per-segment bins using raw headers only.

        :param filenames: raw segment paths; chronological order is inferred
        :param bin_size: positive, non-Boolean raw integrations per bin
        :return: nonnegative total of complete bins; each segment's trailing
                 incomplete bin is discarded without reading image data
        :raises ValueError: if bin size, segment metadata, shapes or temporal
                            coverage are invalid, or no paths are supplied
        :raises OSError: if a raw file cannot be opened
        """
        # Match normalization's per-file tail-discarding rule exactly.
        bin_size = _positive_integer(bin_size, 'bin_size')
        records = _raw_metadata(filenames)
        return sum(record.shape[0] // bin_size for record in records)

    def normalize(self, filenames: Iterable[str | os.PathLike[str]],
                  product_prefix: str | os.PathLike[str],
                  bin_size: int | np.integer[Any] = 1,
                  pixel_bounds: Callable[
                      [str | os.PathLike[str]],
                        tuple[int | np.integer[Any], int | np.integer[Any]]
                        | list[int | np.integer[Any]]] | None = None,
                  from_cds: Callable[
                      [NDArray[np.float64]],
                      tuple[NDArray[np.generic], NDArray[np.generic],
                            NDArray[np.generic]]] | None = None,
                  synthetic_errors: bool = False,
                  integration_slice: tuple[int | np.integer[Any],
                                           int | np.integer[Any]]
                  | list[int | np.integer[Any]] | None = None,
                  reuse: bool = False,
                  conversion_signature: object | Callable[[], object] = None
                  ) -> Self:
        """Write initial products, preserving segment and jump boundaries.

        :param filenames: raw segment paths, sorted by inclusive header IDs
        :param product_prefix: detector/jump-specific path used to derive
                               product filenames; its parent is created
        :param bin_size: positive raw integrations per bin; discard incomplete
                         tails independently for each segment
        :param pixel_bounds: optional filename callback returning two integer
                             column bounds; clip using Python slice rules
        :param from_cds: required for 4-D inputs; convert float64 binned SCI
                         shaped ``(1, reads, y, cropped x)`` to SCI/ERR/DQ
                         tuple arrays each shaped ``(1, y, cropped x)``;
                         retain this legacy instrument interface and map it
                         to CUBE/ERR/DQ at the normalized product boundary;
                         numeric/Boolean outputs are converted to float64
        :param synthetic_errors: for non-CDS inputs, derive ERR as
                                 ``sqrt(abs(SCI))`` and DQ as zeros
        :param integration_slice: nonempty half-open concatenated BINNED
                                  indices, or None to select all complete bins
        :param reuse: skip writes only for a complete, valid expected set
        :param conversion_signature: JSON-serializable CDS settings, or a
                                     zero-argument callback producing them;
                                     evaluated only for CDS, otherwise ignored
        :return: this manager after all products validate; update active paths,
                 shape and acquisition metadata, never retaining image arrays
        :raises ValueError: if metadata, binning, crop, selection, conversion
                            shapes or product alignment are invalid, or CDS
                            conversion is required but no callable is supplied
        :raises TypeError: if CDS settings cannot be JSON-serialized
        :raises OSError: if reading, writing or replacing a FITS product fails
        :raises Exception: propagates pixel-bound and CDS callback failures;
                           failed normalization does not publish active paths
        """
        # Validate raw metadata before allocating any image blocks.
        bin_size = _positive_integer(bin_size, 'bin_size')
        records = _raw_metadata(filenames, pixel_bounds, synthetic_errors)
        flag_cds = len(records[0].shape) == 4
        if flag_cds and not callable(from_cds):
            raise ValueError('CDS requires from_cds callable')
        # Inspect CDS settings only when conversion actually consumes them.
        if flag_cds and callable(conversion_signature):
            conversion_signature = conversion_signature()
        if not flag_cds:
            conversion_signature = None
        # Start binning afresh per source segment; tails never carry over.
        counts = [record.shape[0] // bin_size for record in records]
        selected = _selection(integration_slice, sum(counts))
        # Build every expected name/header before considering cache reuse.
        products: list[_Product] = []
        tasks: list[tuple[_RawFile, int, int]] = []
        offset = 0
        for record, count in zip(records, counts):
            # Intersect the jump with this segment in segment-local bins.
            local_start = max(0, selected[0] - offset)
            local_stop = min(count, selected[1] - offset)
            for start in range(local_start, local_stop,
                               self.max_integrations):
                # Neither source nor jump boundaries can be crossed here.
                stop = min(start + self.max_integrations, local_stop)
                product_args = [record, start, stop, offset, bin_size,
                                selected, product_prefix, synthetic_errors,
                                from_cds, conversion_signature]
                products.append(_initial_product(*product_args))
                tasks.append((record, start, stop))
            # Skipped segments still contribute their complete-bin count.
            offset += count
        if not (reuse and self.products_valid(products)):
            # Read at most one bounded output block from one raw segment.
            for product, task in zip(products, tasks):
                record, start, stop = task
                arrays: dict[str, NDArray[np.float64]] = dict()
                arrays['CUBE'] = np.empty(product.shape, dtype=np.float64)
                arrays['ERR'] = np.empty(product.shape, dtype=np.float64)
                arrays['DQ'] = np.empty(product.shape, dtype=np.float64)
                with fits.open(record.filename, memmap=False,
                               lazy_load_hdus=True) as hdul:
                    for target, bin_index in enumerate(range(start, stop)):
                        bin_args = [hdul, record, bin_index, bin_size,
                                    self.max_integrations, from_cds,
                                    synthetic_errors]
                        converted = _read_raw_bin(*bin_args)
                        expected = (1,) + product.shape[1:]
                        if (len(converted) != 3
                                or any(array.shape != expected
                                       for array in converted)):
                            raise ValueError('Inconsistent converted shapes')
                        # Copy only this integration into the bounded product.
                        science, error, quality = converted
                        arrays['CUBE'][target] = science[0]
                        arrays['ERR'][target] = error[0]
                        arrays['DQ'][target] = quality[0]
                self.write_product(product, arrays)
        # Failed reads/writes/validation never publish a partial active list.
        self.commit_products(products)
        self.filenames = tuple(record.filename for record in records)
        self.integration_ranges = tuple((record.start, record.end)
                                        for record in records)
        self.flag_cds = flag_cds
        return self

    def plan_stage(self, stage: str, signature: object) -> list['_Product']:
        """Plan deterministic outputs without changing active filenames.

        The signature identifies calibration/settings. The complete input
        list prevents reusing global reductions from another selection.

        :param stage: FITS stage label also embedded in the output basename
        :param signature: opaque JSON-serializable calibration/settings
                  payload; a callable is not evaluated here
        :return: ordered metadata for outputs with unchanged 3-D block shapes;
             empty if no products are active; no files or state change
        :raises TypeError: if the signature cannot be JSON-serialized
        :raises ValueError: if serialization detects a circular payload or
                    the resulting basename is invalid
        """
        # Fingerprint the entire stage, not only an individual source block.
        parents = [product.header['PRODUCT'] for product in self._products]
        identity = _fingerprint([stage, signature, parents])
        products = []
        for product in self._products:
            # Retain original segment/crop/bin/jump provenance in every stage.
            header = product.header.copy()
            header['STAGE'] = stage
            header['PARENT'] = header['PRODUCT']
            header['PRODUCT'] = _fingerprint([identity, header['PARENT']])
            # Keep the instrument's detector/jump context in the filename.
            source = Path(product.filename)
            name = f'{source.stem}_{stage}_{identity[:16]}.fits'
            filename = str(source.with_name(name))
            products.append(_Product(filename, product.shape, header))
        return products

    def products_valid(self, products: Sequence['_Product']) -> bool:
        """Require all expected products to match provenance and layout.

        :param products: expected product metadata in stage order
        :return: True only for a nonempty set of readable, bounded float64
                 SCI/ERR/DQ products matching expected headers and shapes;
                 missing, truncated or malformed caches return False
        """
        try:
            for product in products:
                # Also reject missing, truncated and oversized products.
                actual = _inspect_product(product.filename,
                                          self.max_integrations)
                _match_product(actual, product)
            return bool(products)
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            # Partial/malformed caches are misses, never active stages.
            return False

    def write_product(self, product: '_Product',
                      arrays: dict[str, NDArray[Any]]) -> None:
        """Atomically write a bounded floating SCI/ERR/DQ FITS block.

        :param product: destination path, expected 3-D shape and FITS header
        :param arrays: dictionary with exactly CUBE, ERR and DQ keys, each
                   array matching ``product.shape``; store CUBE as SCI;
                   convert numeric/Boolean values to float64, keeping NaNs
                   without changing the dictionary or its input arrays
        :return: None; create parent directories and replace the destination
                 atomically, removing the temporary file even on failure
        :raises ValueError: if payload keys, named array shapes or integration
                    limit fail, or values cannot be converted to float
        :raises TypeError: if values cannot be converted to floating point
        :raises OSError: if directory creation, FITS writing or rename fails
        """
        # Reject misalignment before replacing any successful product file.
        expected_keys = {'CUBE', 'ERR', 'DQ'}
        if set(arrays) != expected_keys:
            missing = sorted(expected_keys.difference(arrays))
            unexpected = sorted(set(arrays).difference(expected_keys))
            raise ValueError('Product requires exactly CUBE/ERR/DQ keys; '
                             f'missing: {missing}; unexpected: {unexpected}')
        for name in ('CUBE', 'ERR', 'DQ'):
            if arrays[name].shape != product.shape:
                raise ValueError(f'Product {name} shape {arrays[name].shape} '
                                 f'does not match expected {product.shape}')
        if product.shape[0] > self.max_integrations:
            raise ValueError('Product exceeds max_integrations')
        # Primary metadata describes the identity of all three extensions.
        hdus = fits.HDUList([fits.PrimaryHDU(header=product.header)])
        for name, extension in (('CUBE', 'SCI'), ('ERR', 'ERR'),
                                ('DQ', 'DQ')):
            # Floating DQ preserves the previous representation and NaNs.
            hdus.append(fits.ImageHDU(np.asarray(arrays[name], dtype=float),
                                      name=extension))
        destination = Path(product.filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A same-directory temporary file permits an atomic final rename.
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent,
                                                suffix='.fits')
        os.close(descriptor)
        try:
            # Never expose a partly written file under its final name.
            hdus.writeto(temporary, overwrite=True)
            os.replace(temporary, destination)
        finally:
            # Failed writes leave no orphan temporary files for reuse.
            if os.path.exists(temporary):
                os.unlink(temporary)

    def commit_products(self, products: Iterable['_Product']) -> None:
        """Publish a complete aligned stage or leave active files unchanged.

        :param products: ordered expected metadata for an entire selection;
                         consume into an owned list before validation
        :return: None; publish active filenames, shape and product metadata
                 only after validating every file and integration boundary;
                 acquisition metadata is not changed
        :raises ValueError: if products are empty, invalid, incomplete,
                            unordered or inconsistent, or an existing stage's
                            block boundaries, shape or alignment would change
        """
        # Own the list so later caller mutations cannot change the plan.
        products = list(products)
        if not products or not self.products_valid(products):
            raise ValueError('Incomplete or invalid product stage')
        first = products[0]
        if (first.header['GSTART'] != first.header['SELSTART']
                or products[-1].header['GSTOP'] != first.header['SELSTOP']):
            raise ValueError('Incomplete product selection coverage')
        expected_start = first.header['GSTART']
        for index, product in enumerate(products):
            # Global half-open coordinates must be contiguous and ordered.
            if (product.header['GSTART'] != expected_start
                    or product.shape[1:] != first.shape[1:]
                    or product.header['SELSTART'] != first.header['SELSTART']
                    or product.header['SELSTOP'] != first.header['SELSTOP']
                    or product.header['STAGE'] != first.header['STAGE']):
                raise ValueError('Product stage has inconsistent alignment')
            expected_start = product.header['GSTOP']
            if self._products:
                # Science stages may change values, not integration identity.
                if len(products) != len(self._products):
                    raise ValueError('Product stage changed block boundaries')
                previous = self._products[index]
                for key in _alignment_keys():
                    if product.header[key] != previous.header[key]:
                        raise ValueError('Product stage changed alignment')
                if product.shape != previous.shape:
                    raise ValueError('Product stage changed shape')
        # Publish state only after validating the entire stage above.
        count = sum(product.shape[0] for product in products)
        self.shape = (count, first.shape[1], first.shape[2])
        self._products = products
        self.current_infiles = [product.filename for product in products]

    def block_math(
            self, operation: str, data_key: str = 'CUBE',
            axis: int | tuple[int, ...] | None = 0,
            selection: slice | Sequence[int] | NDArray[Any] | None = None,
            *, q: float | Sequence[float] | NDArray[Any] | None = None,
            keepdims: bool = False, method: str = 'linear'
            ) -> NDArray[np.float64]:
        """Delegate exact reductions to the storage-independent memory engine.

        The blockmath engine owns validation, memory planning and numerical
        reduction. This adapter supplies the active shape and configured
        budget, validates the payload key, and binds it into a tile reader.
        The reader retains product metadata, extension mapping and bounded
        FITS access here; none of those details enter the numerical engine.

        Each tile holds *all* selected samples along every reduced axis.
        In particular, medians/percentiles never combine partial statistics.
        Input is normalized float64 and reduced by the actual NumPy function,
        retaining NaN/infinity semantics and the selected sample ordering.

        The budget includes the final output, selection/index metadata, tile
        input, FITS section temporaries and NumPy workspace
        (blockmath.MEMORY_FACTORS times the block). It is an
        allocation estimate, not a hard process RSS cap: Python, FITS headers,
        allocator caching and library internals are not bounded here. No
        streaming/approximate fallback is used for oversized reduced domains.

        :param operation: nansum, nanmean, nanmedian or nanpercentile
        :param data_key: CUBE (FITS SCI), ERR or DQ
        :param axis: reduced axes in the selected 3-D cube; negative axes,
                     tuples (including empty) and None follow NumPy
        :param selection: integration slice, 1-D Boolean mask of cube length,
                          or integer vector; preserve order and duplicates,
                          allow negative indices and empty selections
        :param q: required scalar or 1-D percentiles in [0, 100]; vector
                  quantiles prepend an output axis, as in NumPy
        :param keepdims: retain reduced axes with length one
        :param method: NumPy percentile interpolation method
        :return: float64 result, including a zero-dimensional scalar array;
                 empty selection behavior/warnings follow NumPy
        :raises ValueError: for unknown operation/key, malformed selection,
                            duplicate axes or invalid percentile options
        :raises IndexError: for out-of-range indices or axes
        :raises MemoryError: before pixel reads if output plus the smallest
                             complete reduction tile cannot fit the budget
        :raises OSError: if an active product cannot be read
        """
        # Resolve payload names at the storage boundary, not in the engine.
        if data_key not in ('CUBE', 'ERR', 'DQ'):
            raise ValueError('data_key must be CUBE, ERR or DQ')

        def read_tile(indices: NDArray[np.intp], spatial: tuple[slice, ...]
                      ) -> NDArray[np.float64]:
            """Read the bound payload without exposing storage to the engine.

            :param indices: validated active indices, preserving duplicates
            :param spatial: half-open row and column slices
            :return: float64 tile in selected time order
            :raises OSError: if an active product section cannot be read
            """
            # Resolve the method at call time, preserving reader overrides.
            return self._math_tile(indices, spatial, data_key)

        # The engine sees only active dimensions and a closed reader callback.
        math_kwargs: dict[str, Any] = dict(
            axis=axis, selection=selection,
            max_memory_gb=self.max_math_memory_gb, q=q,
            keepdims=keepdims, method=method)
        return blockmath.block_math(operation, self.shape, read_tile,
                                    **math_kwargs)

    def _math_tile(self, indices: NDArray[np.intp],
                   spatial: tuple[slice, ...], data_key: str
                   ) -> NDArray[np.float64]:
        """Read one output tile in selected order using bounded FITS sections.

        :param indices: validated active integration indices, duplicates kept
        :param spatial: detector row/column slices for this output tile
        :param data_key: validated CUBE, ERR or DQ payload name
        :return: owned float64 tile; raw reads never exceed max_integrations
        :raises OSError: if a product section cannot be read
        """
        # Product edges map chunk-local indices without a full-cube gather.
        edges = np.cumsum([product.shape[0] for product in self._products])
        shape = (indices.size,) + tuple(part.stop - part.start
                                       for part in spatial)
        tile = np.empty(shape, dtype=np.float64)
        extension = 'SCI' if data_key == 'CUBE' else data_key
        position = 0
        while position < indices.size:
            index = int(indices[position])
            product_index = int(np.searchsorted(edges, index, side='right'))
            offset = 0 if product_index == 0 else int(edges[product_index - 1])
            product = self._products[product_index]
            # Consecutive selected samples share one bounded raw section.
            stop = position + 1
            while (stop < indices.size
                   and stop - position < self.max_integrations
                   and indices[stop] == index + stop - position
                   and indices[stop] < edges[product_index]):
                stop += 1
            local = index - offset
            section = (slice(local, local + stop - position),) + spatial
            with fits.open(product.filename, memmap=False,
                           lazy_load_hdus=True) as hdul:
                image = cast(fits.ImageHDU, hdul[extension])
                tile[position:stop] = image.section[section]
            position = stop
        return tile

    def nanmean(self, data_key: str = 'CUBE',
                axis: int | tuple[int, ...] | None = 0,
                selection: slice | Sequence[int] | NDArray[Any] | None = None,
                *, keepdims: bool = False) -> NDArray[np.float64]:
        """Return exact NumPy nanmean; see block_math for options and errors.

        :param data_key: CUBE, ERR or DQ
        :param axis: reduced axes, default integration axis
        :param selection: integration mask, slice or ordered index vector
        :param keepdims: retain reduced axes
        :return: float64 mean, with NumPy NaN/infinity/empty behavior
        :raises MemoryError: if estimated math allocations exceed the budget
        """
        return self.block_math('nanmean', data_key, axis, selection,
                               keepdims=keepdims)

    def nansum(self, data_key: str = 'CUBE',
               axis: int | tuple[int, ...] | None = 0,
               selection: slice | Sequence[int] | NDArray[Any] | None = None,
               *, keepdims: bool = False) -> NDArray[np.float64]:
        """Return exact NumPy nansum; see block_math for options and errors.

        :param data_key: CUBE, ERR or DQ
        :param axis: reduced axes, default integration axis
        :param selection: integration mask, slice or ordered index vector
        :param keepdims: retain reduced axes
        :return: float64 sum, with NumPy NaN/infinity/empty behavior
        :raises MemoryError: if estimated math allocations exceed the budget
        """
        return self.block_math('nansum', data_key, axis, selection,
                               keepdims=keepdims)

    def nanmedian(self, data_key: str = 'CUBE',
                  axis: int | tuple[int, ...] | None = 0,
                  selection: slice | Sequence[int] | NDArray[Any] | None
                  = None,
                  *, keepdims: bool = False) -> NDArray[np.float64]:
        """Return exact NumPy nanmedian; see block_math for options and errors.

        :param data_key: CUBE, ERR or DQ
        :param axis: reduced axes, default integration axis
        :param selection: integration mask, slice or ordered index vector
        :param keepdims: retain reduced axes
        :return: float64 median, never a median of partial medians
        :raises MemoryError: if estimated math allocations exceed the budget
        """
        return self.block_math('nanmedian', data_key, axis, selection,
                               keepdims=keepdims)

    def nanpercentile(
            self, q: float | Sequence[float] | NDArray[Any],
            data_key: str = 'CUBE', axis: int | tuple[int, ...] | None = 0,
            selection: slice | Sequence[int] | NDArray[Any] | None = None,
            *, keepdims: bool = False, method: str = 'linear'
            ) -> NDArray[np.float64]:
        """Return exact NumPy nanpercentile within estimated math memory.

        :param q: scalar or 1-D percentiles in [0, 100], as in NumPy
        :param data_key: CUBE, ERR or DQ
        :param axis: reduced axes, default integration axis
        :param selection: integration mask, slice or ordered index vector
        :param keepdims: retain reduced axes
        :param method: NumPy interpolation method, default linear
        :return: float64 percentiles with leading vector-quantile axis
        :raises ValueError: if q or method is invalid
        :raises MemoryError: if estimated math allocations exceed the budget
        """
        return self.block_math('nanpercentile', data_key, axis, selection,
                               q=q, keepdims=keepdims, method=method)

    def read_integrations(self, start: int | np.integer[Any] = 0,
                          stop: int | np.integer[Any] | None = None
                          ) -> dict[str, NDArray[np.float64]]:
        """Read an aligned half-open range within the integration limit.

        :param start: zero-based inclusive binned index; not a Boolean
        :param stop: exclusive binned index, or None for at most one bounded
                     block beginning at start and clipped to the selection
        :return: dictionary with exactly CUBE, ERR and DQ keys and newly
             allocated float64 arrays, each shaped
             ``(stop - start, y, cropped x)``; CUBE reads FITS SCI;
             may span product boundaries
        :raises ValueError: if indices are invalid, Boolean, empty, outside
                            the selection or exceed the integration limit
        :raises OSError: if an active product cannot be read
        """
        # An omitted stop requests one block, never the full observation.
        if stop is None:
            stop = cast(int | np.integer[Any],
                        min(start + self.max_integrations, len(self)))
        if (any(isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                for value in (start, stop))
                or not 0 <= start < stop <= len(self)):
            raise ValueError('Integration range is outside the DataCube')
        if stop - start > self.max_integrations:
            raise ValueError('Requested integrations exceed max_integrations')
        # Explicit reads may span products but cannot exceed the limit.
        shape = (stop - start,) + self.shape[1:]
        arrays: dict[str, NDArray[np.float64]] = dict()
        arrays['CUBE'] = np.empty(shape, dtype=np.float64)
        arrays['ERR'] = np.empty(shape, dtype=np.float64)
        arrays['DQ'] = np.empty(shape, dtype=np.float64)
        offset = 0
        for product in self._products:
            local_start = max(0, start - offset)
            local_stop = min(product.shape[0], stop - offset)
            if local_start < local_stop:
                with fits.open(product.filename, memmap=False,
                               lazy_load_hdus=True) as hdul:
                    for name, extension in (('CUBE', 'SCI'), ('ERR', 'ERR'),
                                            ('DQ', 'DQ')):
                        target = offset + local_start - start
                        count = local_stop - local_start
                        # Section reads also handle scaled FITS image types.
                        image = cast(fits.ImageHDU, hdul[extension])
                        arrays[name][target:target + count] = image.section[
                            local_start:local_stop, :, :]
            offset += product.shape[0]
        return arrays

    def iter_integrations(self) -> Iterator[
            tuple[slice, dict[str, NDArray[np.float64]]]]:
        """Yield aligned SCI/ERR/DQ blocks at active product boundaries.

        :return: iterator of ``(slice, arrays)`` pairs, where arrays is a
             dictionary with exactly CUBE, ERR and DQ keys; slices use
             half-open binned indices and each float64 array has shape
                 ``(slice.stop - slice.start, y, cropped x)`` within the limit;
                 an empty manager yields nothing and retains no image blocks
        :raises OSError: if an active product cannot be read while iterating
        :raises ValueError: if active metadata violates the read bounds
        """
        offset = 0
        for product in self._products:
            # Every product obeys both source boundaries and the read limit.
            stop = offset + product.shape[0]
            yield slice(offset, stop), self.read_integrations(offset, stop)
            offset = stop


@dataclass(frozen=True)
class _Product:
    """Store expected product metadata without retaining image arrays.

    :param filename: normalized product destination as a string path
    :param shape: positive ``(binned integrations, y, cropped x)`` lengths
    :param header: FITS provenance, stage identity and half-open selection
                   coordinates; the frozen record does not freeze this header
    """

    filename: str
    shape: tuple[int, int, int]
    header: fits.Header


# =============================================================================
# Define functions
# =============================================================================
def _selection(selection: tuple[int | np.integer[Any], int | np.integer[Any]]
               | list[int | np.integer[Any]] | None,
               total: int) -> tuple[int, int]:
    """Validate half-open concatenated binned integration coordinates.

    :param selection: two non-Boolean integer bounds, or None for
                      ``(0, total)``
    :param total: available complete, concatenated binned integration count
    :return: built-in integer pair with ``0 <= start < stop <= total``
    :raises ValueError: if bounds are malformed, empty or out of range;
                        selecting all of zero available bins is also invalid
    """
    if selection is None:
        selection = (0, total)
    if (not isinstance(selection, (tuple, list)) or len(selection) != 2
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, np.integer))
                   for value in selection)
            or not 0 <= selection[0] < selection[1] <= total):
        raise ValueError('integration_slice must be nonempty and in bounds')
    return cast(tuple[int, int], tuple(int(value) for value in selection))


def _fingerprint(values: object) -> str:
    """Identify source/settings with stable serialization, not Python hash.

    :param values: opaque, potentially nested heterogeneous JSON payload;
                   serialized with the default ``json.dumps`` options
    :return: 64-character hexadecimal SHA-256 digest of the UTF-8
             serialization; mapping insertion order remains significant
    :raises TypeError: if any value is not JSON-serializable
    :raises ValueError: if the payload contains a circular reference
    """
    return hashlib.sha256(json.dumps(values).encode('utf-8')).hexdigest()


def _alignment_keys() -> tuple[str, ...]:
    """Return integration identity keys that science stages must not change.

    :return: ordered FITS keyword tuple describing source, inclusive raw IDs,
             binning, crop and half-open binned selection coordinates
    """
    return ('SOURCE', 'SEGSTART', 'SEGEND', 'BSTART', 'BSTOP', 'GSTART',
            'GSTOP', 'BINNING', 'CROPLO', 'CROPHI', 'SELSTART', 'SELSTOP')


def _initial_product(record: _RawFile, start: int, stop: int, offset: int,
                     bin_size: int, selected: tuple[int, int],
                     product_prefix: str | os.PathLike[str],
                     synthetic_errors: bool,
                     from_cds: Callable[
                         [NDArray[np.float64]],
                         tuple[NDArray[np.generic], NDArray[np.generic],
                               NDArray[np.generic]]] | None,
                     conversion_signature: object) -> _Product:
    """Build deterministic provenance and names for one normalized block.

    :param record: validated source-segment metadata and selected column bounds
    :param start: inclusive, segment-local binned integration index
    :param stop: exclusive, segment-local binned integration index
    :param offset: complete bins preceding this segment in the concatenation
    :param bin_size: positive raw integrations combined per output integration
    :param selected: global half-open binned integration selection
    :param product_prefix: detector/jump-specific path for derived filenames
    :param synthetic_errors: include synthetic-error mode in product identity
    :param from_cds: optional converter; only its qualified name is inspected,
                     never called by this metadata helper
    :param conversion_signature: already-evaluated, JSON-serializable CDS
                                 settings, or None when CDS is not used
    :return: product path, provenance header and fixed 3-D shape
             ``(stop - start, raw y, selected width)``; no image files are made
    :raises OSError: if resolving or stat-ing the source fails
    :raises TypeError: if identity values are not JSON-serializable
    :raises ValueError: if the identity payload is circular or name is invalid
    """
    # Full source identity distinguishes equal basenames and detector files.
    source = str(Path(record.filename).resolve())
    stat = os.stat(source)
    converter = getattr(from_cds, '__qualname__', None)
    identity = _fingerprint([source, stat.st_size, stat.st_mtime_ns,
                             record.start, record.end, record.bounds,
                             bin_size, selected, start, stop, offset,
                             synthetic_errors, converter,
                             conversion_signature, __version__])
    # The prefix already carries instrument-defined detector/jump suffixes.
    prefix = Path(product_prefix)
    name = (f'{prefix.stem}_seg{record.start:06d}-{record.end:06d}'
            f'_int{start:06d}-{stop:06d}_{identity[:16]}.fits')
    header = fits.Header()
    # Raw inclusive IDs and binned half-open indices are both explicit.
    header['DCVER'] = 1
    header['STAGE'] = 'initial'
    header['PRODUCT'] = identity
    header['SOURCE'] = source
    header['SEGSTART'] = record.start
    header['SEGEND'] = record.end
    header['BSTART'] = start
    header['BSTOP'] = stop
    header['GSTART'] = offset + start
    header['GSTOP'] = offset + stop
    header['BINNING'] = bin_size
    header['CROPLO'], header['CROPHI'] = record.bounds
    header['SELSTART'], header['SELSTOP'] = selected
    # Retain exactly the selected bins and cropped spatial geometry.
    shape = (stop - start, record.shape[-2],
             record.bounds[1] - record.bounds[0])
    return _Product(str(prefix.with_name(name)), shape, header)


def _inspect_product(filename: str | os.PathLike[str], limit: int) -> _Product:
    """Validate product provenance, extension alignment and completeness.

    :param filename: normalized FITS product path
    :param limit: positive maximum integrations permitted in this block
    :return: metadata with a string path, copied primary header and positive
             ``(integrations, y, cropped x)`` shape; all SCI/ERR/DQ extensions
             must have this shape and float64 storage, with complete payloads;
             validation reads headers and file sizes, not image arrays
    :raises ValueError: if provenance, dimensions, limits, alignment, extension
                        storage or file completeness checks fail
    :raises KeyError: if a required extension or FITS axis keyword is missing
    :raises OSError: if the file cannot be opened or its size cannot be read
    """
    with fits.open(filename, memmap=False, lazy_load_hdus=True) as hdul:
        header = cast(fits.PrimaryHDU, hdul[0]).header.copy()
        if (header.get('DCVER') != 1 or not header.get('PRODUCT')
                or not header.get('STAGE') or len(hdul) != 4):
            raise ValueError(f'{filename}: invalid normalized product header')
        # Reopened products must carry all required alignment metadata.
        for key in _alignment_keys():
            if key not in header:
                raise ValueError(f'{filename}: missing {key}')
        science_hdu = cast(fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU,
                   hdul['SCI'])
        shape = cast(tuple[int, int, int],
                 _image_shape(science_hdu, filename, (3,)))
        if shape[0] > limit:
            raise ValueError(f'{filename}: product exceeds max_integrations')
        # Validate scalar types too; FITS strings are not integration indices.
        start = _nonnegative_integer(header['BSTART'], 'BSTART')
        stop = _positive_integer(header['BSTOP'], 'BSTOP')
        global_start = _nonnegative_integer(header['GSTART'], 'GSTART')
        global_stop = _positive_integer(header['GSTOP'], 'GSTOP')
        segment_start = _positive_integer(header['SEGSTART'], 'SEGSTART')
        segment_stop = _positive_integer(header['SEGEND'], 'SEGEND')
        selection_start = _nonnegative_integer(header['SELSTART'], 'SELSTART')
        selection_stop = _positive_integer(header['SELSTOP'], 'SELSTOP')
        crop_start = _nonnegative_integer(header['CROPLO'], 'CROPLO')
        crop_stop = _positive_integer(header['CROPHI'], 'CROPHI')
        available = segment_stop - segment_start + 1
        available //= _positive_integer(header['BINNING'], 'BINNING')
        # Reject products crossing their source segment or selected jump.
        if (not 0 <= start < stop <= available
                or stop - start != shape[0]
                or global_stop - global_start != shape[0]
                or not selection_start <= global_start < global_stop
                <= selection_stop
                or crop_stop - crop_start != shape[-1]):
            raise ValueError(f'{filename}: inconsistent product alignment')
        for name in ('SCI', 'ERR', 'DQ'):
            hdu = cast(fits.ImageHDU, hdul[name])
            if (_image_shape(hdu, filename, (3,)) != shape
                    or hdu.header['BITPIX'] != -64):
                raise ValueError(f'{filename}: inconsistent extension shapes')
            # Header axes alone cannot catch a truncated image payload.
            info = hdu.fileinfo()
            if (info is None or os.path.getsize(filename)
                    < info['datLoc'] + info['datSpan']):
                raise ValueError(f'{filename}: truncated product')
    return _Product(str(filename), shape, header)


def _match_product(actual: _Product, expected: _Product) -> None:
    """Require expected provenance and shape, not merely file existence.

    :param actual: inspected on-disk metadata
    :param expected: required shape and header values; extra actual keys are
                     permitted and filenames are not compared
    :return: None; neither record nor file is changed
    :raises ValueError: if the shape or any expected header value differs
    """
    if actual.shape != expected.shape:
        raise ValueError('Product has inconsistent expected shape')
    for key in expected.header:
        if actual.header.get(key) != expected.header[key]:
            raise ValueError(f'Product has inconsistent {key}')


def _positive_integer(value: object, name: str) -> int:
    """Validate counts without silently coercing floats or booleans.

    :param value: candidate count, including untrusted FITS header values
    :param name: field label included in validation errors
    :return: built-in integer at least one, accepting NumPy integer scalars
    :raises ValueError: if the value is Boolean, noninteger or less than one
    """
    # NumPy integer scalars are accepted, but bool is not a valid count.
    if (isinstance(value, bool)
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _nonnegative_integer(value: object, name: str) -> int:
    """Validate zero-based metadata without coercing booleans or floats.

    :param value: candidate coordinate, including untrusted FITS header values
    :param name: field label included in validation errors
    :return: built-in integer at least zero, accepting NumPy integer scalars
    :raises ValueError: if the value is Boolean, noninteger or negative
    """
    if (isinstance(value, bool)
            or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _image_shape(hdu: fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU,
                 filename: str | os.PathLike[str],
                 dimensions: tuple[int, ...]) -> tuple[int, ...]:
    """Recover NumPy image shape entirely from FITS axis headers.

    :param hdu: image HDU whose header is inspected without accessing data
    :param filename: source path used only to label validation errors
    :param dimensions: permitted axis counts, normally ``(3,)`` for products
                       or ``(3, 4)`` for ordinary/CDS raw data
    :return: positive axis lengths in reversed FITS order, hence NumPy
             ``(integrations, y, x)`` or ``(integrations, reads, y, x)``
    :raises ValueError: if the axis count is not permitted or any axis length
                        is missing, Boolean, noninteger or nonpositive
    """
    # Header access never evaluates the image .data property.
    header = hdu.header
    ndim = header.get('NAXIS')
    if ndim not in dimensions:
        raise ValueError(f'{filename}: expected image dimensions {dimensions}')
    # Reverse FITS x,y,... axes to NumPy (...,y,x) ordering.
    shape = tuple(header.get(f'NAXIS{axis}')
                  for axis in range(cast(int, ndim), 0, -1))
    for size in shape:
        _positive_integer(size, f'{filename}: image axis')
    return cast(tuple[int, ...], shape)


def _extension(hdul: fits.HDUList, name: str, fallback: int,
               filename: str | os.PathLike[str]) -> int:
    """Prefer named extensions, retaining legacy positional raw inputs.

    :param hdul: open FITS HDU list; image payloads are not read
    :param name: preferred extension name, usually SCI, ERR or DQ
    :param fallback: legacy nonnegative positional index when name is absent
    :param filename: source path used only to label validation errors
    :return: zero-based HDU index, preferring the named extension
    :raises ValueError: if the name is absent and the fallback is out of bounds
    """
    try:
        return hdul.index_of(name)
    except KeyError:
        if fallback >= len(hdul):
            raise ValueError(f'{filename}: missing {name} extension') from None
        return fallback


def _raw_metadata(filenames: Iterable[str | os.PathLike[str]],
                  pixel_bounds: Callable[
                      [str | os.PathLike[str]],
                      tuple[int | np.integer[Any], int | np.integer[Any]]
                      | list[int | np.integer[Any]]] | None = None,
                  synthetic_errors: bool = True) -> list[_RawFile]:
    """Validate and order raw segments without reading image sections.

    :param filenames: raw segment paths; input iteration order is immaterial
    :param pixel_bounds: optional filename callback returning an integer pair
                         or list of column bounds; clip to the last raw axis
    :param synthetic_errors: permit ordinary SCI-only inputs; otherwise require
                             aligned ERR/DQ; CDS always needs only SCI
    :return: nonempty header-sorted metadata list with contiguous inclusive
             raw integration IDs, matching selected spatial/read geometry and
             uniformly 3-D ordinary or 4-D CDS shapes; retain original paths
    :raises ValueError: if paths are empty, required metadata or extensions are
                        invalid, raw IDs overlap or have gaps, shapes differ,
                        CDS is mixed with ordinary data, or a crop is empty
    :raises OSError: if a source file cannot be opened
    :raises Exception: propagates pixel-bound callback failures
    """
    # Retain metadata only, irrespective of observation length.
    records: list[_RawFile] = []
    for filename in filenames:
        with fits.open(filename, memmap=False, lazy_load_hdus=True) as hdul:
            header = cast(fits.PrimaryHDU, hdul[0]).header
            start = _positive_integer(header.get('INTSTART'),
                                      f'{filename}: INTSTART')
            end = _positive_integer(header.get('INTEND'),
                                    f'{filename}: INTEND')
            science_ext = _extension(hdul, 'SCI', 1, filename)
            science_hdu = cast(
                fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU,
                hdul[science_ext])
            shape = _image_shape(science_hdu, filename, (3, 4))
            # Inclusive header IDs must account for the entire raw time axis.
            if end < start or end - start + 1 != shape[0]:
                raise ValueError(f'{filename}: INTSTART/INTEND count does not '
                                 'match science integration axis')
            extensions = (science_ext,)
            if len(shape) == 3 and not synthetic_errors:
                # Ordinary inputs carry ERR/DQ; CDS and FGS derive them.
                extension_specs = [('ERR', 2), ('DQ', 3)]
                extensions += tuple(
                    _extension(hdul, name, fallback, filename)
                    for name, fallback in extension_specs)
                for extension in extensions[1:]:
                    image_hdu = cast(
                        fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU,
                        hdul[extension])
                    if _image_shape(image_hdu, filename, (3,)) != shape:
                        raise ValueError(f'{filename}: inconsistent '
                                         'extension shapes')
            bounds = (0, shape[-1])
            if pixel_bounds is not None:
                # Resolve instrument-specific crops before geometry checks.
                bounds = pixel_bounds(filename)
                if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                        or any(isinstance(value, bool)
                               or not isinstance(value, (int, np.integer))
                               for value in bounds)):
                    raise ValueError(f'{filename}: invalid pixel_bounds')
                bounds = slice(*bounds).indices(shape[-1])[:2]
                if bounds[0] >= bounds[1]:
                    raise ValueError(f'{filename}: empty pixel_bounds')
            records.append(_RawFile(filename, start, end, shape,
                                    extensions,
                                    cast(tuple[int, int], tuple(bounds))))
    if not records:
        raise ValueError('No input filenames')
    # Header identity, never YAML/name order, defines chronology.
    records.sort(key=lambda record: record.start)
    first = records[0]
    first_shape = first.shape[1:-1] + (first.bounds[1] - first.bounds[0],)
    for index, record in enumerate(records):
        if len(record.shape) != len(first.shape):
            emsg = f'{record.filename}: mixed CDS and non-CDS inputs'
            raise ValueError(emsg)
        width = record.bounds[1] - record.bounds[0]
        selected_shape = record.shape[1:-1] + (width,)
        if selected_shape != first_shape:
            raise ValueError(f'{record.filename}: inconsistent image shapes')
        # Reject missing/duplicated IDs rather than compressing time silently.
        if index and record.start != records[index - 1].end + 1:
            overlaps = record.start <= records[index - 1].end
            problem = 'overlap' if overlaps else 'gap'
            raise ValueError(f'{record.filename}: integration range {problem}')
    return records


def _read_raw_bin(hdul: fits.HDUList, record: _RawFile, bin_index: int,
                  bin_size: int, limit: int,
                  from_cds: Callable[
                      [NDArray[np.float64]],
                      tuple[NDArray[np.generic], NDArray[np.generic],
                            NDArray[np.generic]]] | None,
                  synthetic_errors: bool
                  ) -> tuple[NDArray[np.generic], NDArray[np.generic],
                             NDArray[np.generic]]:
    """Combine one output bin, bounding every raw integration section read.

    :param hdul: open raw FITS HDUs, kept open by the caller during section
                 reads
    :param record: validated source metadata, HDU indices and clipped crop
    :param bin_index: zero-based segment-local index of a complete output bin
    :param bin_size: positive raw integrations per bin; never span segments
    :param limit: positive maximum raw integrations per section read
    :param from_cds: required for 4-D raw inputs; convert float64 binned SCI
                     shaped ``(1, reads, y, cropped x)`` to SCI/ERR/DQ arrays
    :param synthetic_errors: for ordinary SCI-only data, derive
                             ``sqrt(abs(SCI))`` errors and zero flags
    :return: legacy SCI/ERR/DQ tuple arrays shaped ``(1, y, cropped x)`` for a
             valid converter; normalization maps these to CUBE/ERR/DQ keys.
             Ordinary outputs are float64, while CDS output dtypes
             are converter-defined and may include Boolean DQ. For binning,
             sum SCI, add ERR in quadrature and take DQ's NaN-aware maximum;
             without binning preserve values and NaNs
    :raises OSError: if a raw image section cannot be read
    :raises TypeError: if CDS is requested without a callable converter
    :raises Exception: propagates CDS conversion failures; the caller checks
                       the converted array count and shapes before writing
    """
    begin = bin_index * bin_size
    end = begin + bin_size
    # Retain CDS reads and cropped spatial axes while reducing integrations.
    image_shape = record.shape[1:-1] + (record.bounds[1] - record.bounds[0],)
    totals = [np.zeros(image_shape) for _ in record.extensions]
    if len(totals) == 3:
        # NaN allows the first finite flag to win during streamed maxima.
        totals[2].fill(np.nan)
    for start in range(begin, end, limit):
        stop = min(start + limit, end)
        selection = ((slice(start, stop),)
                     + (slice(None),) * (len(record.shape) - 2)
                     + (slice(*record.bounds),))
        for index, extension in enumerate(record.extensions):
            image_hdu = cast(
                fits.PrimaryHDU | fits.ImageHDU | fits.CompImageHDU,
                hdul[extension])
            block = np.asarray(image_hdu.section[selection], dtype=float)
            if bin_size == 1:
                # No reduction: retain original values, including NaNs.
                totals[index] = block[0]
            elif index == 2:
                # Match legacy DQ maximum binning, not bitwise OR.
                reduced = np.nanmax(block, axis=0)
                np.fmax(totals[index], reduced, out=totals[index])
            else:
                # SCI sums; independent ERR values add in quadrature.
                values = block ** 2 if index == 1 else block
                totals[index] += np.nansum(values, axis=0)
    science = totals[0][None]
    if len(record.shape) == 4:
        # CDS conversion follows per-file binning, as in the original loader.
        converter = cast(Callable[
            [NDArray[np.float64]],
            tuple[NDArray[np.generic], NDArray[np.generic],
                  NDArray[np.generic]]], from_cds)
        return converter(science)
    if synthetic_errors:
        # FGS SCI-only inputs derive errors/flags after binning.
        return science, np.sqrt(np.abs(science)), np.zeros_like(science)
    error = np.sqrt(totals[1]) if bin_size > 1 else totals[1]
    return science, error[None], totals[2][None]


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
