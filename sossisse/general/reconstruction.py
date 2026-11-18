#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persist aligned, bounded linear-reconstruction products and manifests."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from astropy.io import fits
from astropy.table import Table

from sossisse.core import base
from sossisse.general.datacube import _Product
from sossisse.general.datacube import _image_shape
from sossisse.general.datacube import _nonnegative_integer
from sossisse.general.datacube import _positive_integer


# =============================================================================
# Define variables
# =============================================================================
__NAME__ = 'sossisse.general.reconstruction'
__version__ = base.__version__
__date__ = base.__date__
__authors__ = base.__authors__
# RECON is amplitude-normalized; RESIDUAL is data minus the fitted model.
# VALID contains the fitting mask (1/NaN); ERR retains input uncertainties.
PLANES = ('RECON', 'RESIDUAL', 'VALID', 'ERR')


# =============================================================================
# Define classes
# =============================================================================
class ReconstructionProducts:
    """Keep reconstruction filenames, never observation-sized image arrays.

    :param filenames: ordered block paths with global half-open coordinates
    :param max_integrations: maximum integrations per block/read
    :param max_math_memory_gb: working budget for spatial-history reads
    :raises ValueError: if blocks are missing, malformed or misaligned
    """

    def __init__(self, filenames: Sequence[str], max_integrations: int = 1000,
                 max_math_memory_gb: float = 8.0) -> None:
        # Own the path list so caller mutations cannot reorder this manager.
        self.current_infiles = list(filenames)
        # Keep the read limit independent of the total observation length.
        self.max_integrations = max_integrations
        # Pixel histories use the math budget rather than the block limit.
        self.max_math_memory_gb = max_math_memory_gb
        # Cache only geometry and provenance, never image payloads.
        self.shapes: list[tuple[int, int, int]] = []
        self.headers: list[fits.Header] = []
        # Empty collections cannot establish a detector grid or row coverage.
        if not filenames or max_integrations < 1:
            raise ValueError('Reconstruction requires nonempty bounded blocks')
        # After the first block, each block must begin at this global index.
        expected_start = None
        # Respect supplied order; sorting could hide an incorrect jump merge.
        for filename in filenames:
            # Header and file-size checks do not require loading image arrays.
            with fits.open(filename, memmap=False) as hdul:
                # Keep metadata usable after the FITS file closes.
                header = cast(fits.PrimaryHDU, hdul[0]).header.copy()
                # The primary header must accompany exactly four image planes.
                if header.get('RCPVER') != 1 or len(hdul) != len(PLANES) + 1:
                    raise ValueError('Invalid reconstruction block layout')
                # PRODUCT identifies the fitted model and its parent block.
                if not isinstance(header.get('PRODUCT'), str):
                    raise ValueError('Missing reconstruction product identity')
                # GSTART/GSTOP address the concatenated, binned source frames.
                start = _nonnegative_integer(header['GSTART'], 'GSTART')
                stop = _positive_integer(header['GSTOP'], 'GSTOP')
                # Each block also retains the jump selection that produced it.
                selection_start = _nonnegative_integer(header['SELSTART'],
                                                       'SELSTART')
                selection_stop = _positive_integer(header['SELSTOP'],
                                                     'SELSTOP')
                # A block may not contain rows outside its own jump chunk.
                if not selection_start <= start < stop <= selection_stop:
                    raise ValueError('Block lies outside its jump selection')
                # Merged results may contain several complete jump selections.
                new_selection = not self.headers or (
                    selection_start != self.headers[-1]['SELSTART']
                    or selection_stop != self.headers[-1]['SELSTOP'])
                if new_selection:
                    # Check both the new chunk's start and the old chunk's end.
                    if (start != selection_start or (self.headers
                            and expected_start
                            != self.headers[-1]['SELSTOP'])):
                        raise ValueError('Incomplete jump selection coverage')
                # All four planes must describe precisely the same 3-D block.
                shapes = []
                for name in PLANES:
                    # Resolve named extensions, independent of their ordering.
                    image = cast(fits.ImageHDU, hdul[name])
                    # Recover (integrations, y, x) from NAXIS headers only.
                    shape = _image_shape(image, filename, (3,))
                    # Payload location/length exposes truncated FITS files.
                    info = image.fileinfo()
                    # BITPIX=-64 is float64; the entire payload must exist.
                    if (image.header['BITPIX'] != -64 or info is None
                            or os.path.getsize(filename)
                            < info['datLoc'] + info['datSpan']):
                        raise ValueError('Incomplete reconstruction payload')
                    # Retain geometry only; no image object escapes this file.
                    shapes.append(shape)
                # Use the first plane as the common block geometry.
                shape = shapes[0]
                # Check plane agreement, the read limit and global continuity.
                if (any(part != shape for part in shapes)
                        or any(size < 1 for size in shape)
                        or shape[0] > max_integrations
                        or stop - start != shape[0]
                        or (expected_start is not None
                            and start != expected_start)):
                    raise ValueError('Inconsistent reconstruction alignment')
                if self.headers:
                    # Time-adjacent blocks must still share one detector grid.
                    first = self.headers[0]
                    # Matching shapes alone cannot distinguish NRS1 from NRS2.
                    if (shape[1:] != self.shapes[0][1:]
                            or header.get('RCDET') != first.get('RCDET')):
                        raise ValueError('Mixed reconstruction detector grids')
                    # The exclusive stop is the next block's expected start.
                expected_start = stop
                    # Dimension validation above establishes a fixed 3-D shape.
                self.shapes.append(cast(tuple[int, int, int], shape))
                    # Preserve source, selection and model identity.
                self.headers.append(header)
                # Adjacent blocks alone do not prove final-chunk completeness.
        if expected_start != self.headers[-1]['SELSTOP']:
            raise ValueError('Incomplete jump selection coverage')
                # Logical rows start at zero, even for a nonzero source start.
        self.shape = (sum(shape[0] for shape in self.shapes),
                      self.shapes[0][1], self.shapes[0][2])

    def __len__(self) -> int:
        """Return the number of ordered integrations.

        :return: total rows represented by the block list
        """
        # Count integrations rather than files or detector pixels.
        return self.shape[0]

    def plan_stage(self, stage: str, signature: object) -> list[_Product]:
        """Plan derived blocks without changing the current collection.

        :param stage: output label
        :param signature: JSON-serializable settings shared across all blocks
        :return: product records preserving all integration coordinates
        """
        # Include every parent so global baseline settings apply to one set.
        parents = [header['PRODUCT'] for header in self.headers]
        # Stable serialization distinguishes stages and their effective inputs.
        identity = hashlib.sha256(json.dumps(
            [stage, signature, parents]).encode()).hexdigest()
        # A plan contains metadata only; it creates no files or active state.
        products = []
        for path, shape, original in zip(self.current_infiles, self.shapes,
                                         self.headers):
            # Copy provenance so planning never edits the current header.
            header = original.copy()
            # Record both the new processing step and its immediate parent.
            header['STAGE'] = stage
            header['PARENT'] = original['PRODUCT']
            # Each output identity combines the global stage and local parent.
            header['PRODUCT'] = hashlib.sha256(
                (identity + str(original['PRODUCT'])).encode()).hexdigest()
            # Keep source, detector and jump labels from the existing basename.
            source = Path(path)
            # The digest prevents collisions between different stage settings.
            name = f'{source.stem}_{stage}_{identity[:16]}.fits'
            # Preserve the block shape and all integration-coordinate headers.
            products.append(_Product(str(source.with_name(name)), shape,
                                     header))
        # The caller writes these products before opening a new collection.
        return products

    @staticmethod
    def write_block(product: _Product,
                    arrays: dict[str, NDArray[np.float64]],
                    detector: str = '') -> None:
        """Write one complete block by atomic replacement.

        :param product: destination, shape and inherited source provenance
        :param arrays: exactly RECON, RESIDUAL, VALID and ERR float arrays
        :param detector: detector-context suffix, kept through jump merges
        :return: None; original arrays and input products are unchanged
        :raises ValueError: if any plane is absent or has the wrong shape
        """
        # Reject incomplete or unexpected payloads before creating any file.
        if set(arrays) != set(PLANES):
            raise ValueError('Reconstruction requires '
                             'RECON/RESIDUAL/VALID/ERR')
        # Writing must not mutate the caller's planned provenance header.
        header = product.header.copy()
        # Distinguish this layout from normalized FLUX/ERR/DQ input products.
        header['RCPVER'] = 1
        # Explicit detector context remains available after jump merging.
        header['RCDET'] = detector
        # The primary HDU stores provenance, not an observation-sized image.
        hdus = fits.HDUList([fits.PrimaryHDU(header=header)])
        for name in PLANES:
            # Every plane must align with the planned source integrations.
            if arrays[name].shape != product.shape:
                raise ValueError(f'Invalid reconstruction {name} shape')
            # Named float64 extensions preserve NaNs and the 1/NaN VALID mask.
            hdus.append(fits.ImageHDU(np.asarray(arrays[name], dtype=float),
                                      name=name))
        # Resolve the destination once for directory creation and replacement.
        destination = Path(product.filename)
        # Parent directories may not exist for a new processing context.
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Same-directory staging allows an atomic rename on this filesystem.
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
        # Astropy reopens the path; release mkstemp's descriptor.
        os.close(descriptor)
        try:
            # Never write an incomplete payload under the final product name.
            hdus.writeto(temporary, overwrite=True)
            # Only a successfully written file replaces the destination.
            os.replace(temporary, destination)
        finally:
            # Release HDU resources on both the successful and failing paths.
            hdus.close()
            # A failed write must not leave a reusable-looking staging file.
            if os.path.exists(temporary):
                os.unlink(temporary)

    def write_input_residuals(self, output_dir: str, suffix: str,
                              *, overwrite: bool = True) -> None:
        """Export one primary-image residual cube per source in this jump."""
        by_source: dict[str, list[int]] = {}
        for index, header in enumerate(self.headers):
            by_source.setdefault(str(header['SOURCE']), []).append(index)

        destinations: dict[Path, list[int]] = {}
        for source, indices in by_source.items():
            stem = Path(Path(source).name.removesuffix('.gz')).stem
            destination = Path(output_dir) / f'{stem}{suffix}_residuals.fits'
            if destination in destinations:
                raise ValueError(f'Duplicate residual filename: {destination}')
            destinations[destination] = indices

        for destination, indices in destinations.items():
            if not overwrite and destination.is_file():
                continue
            first = self.headers[indices[0]]
            last = self.headers[indices[-1]]
            frame_count = sum(self.shapes[index][0] for index in indices)
            _, rows, columns = self.shapes[indices[0]]
            header = fits.Header(dict(SIMPLE=True, BITPIX=-64, NAXIS=3,
                                      NAXIS1=columns, NAXIS2=rows,
                                      NAXIS3=frame_count))
            for key in ('SOURCE', 'SEGSTART', 'SEGEND', 'BSTART', 'GSTART',
                        'BINNING', 'CROPLO', 'CROPHI', 'SELSTART', 'SELSTOP',
                        'RCDET'):
                header[key] = first[key]
            for key in ('BSTOP', 'GSTOP'):
                header[key] = last[key]

            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
            os.close(descriptor)
            try:
                stream = fits.StreamingHDU(temporary, header)
                try:
                    for index in indices:
                        with fits.open(self.current_infiles[index],
                                       memmap=True) as hdul:
                            image = cast(fits.ImageHDU, hdul['RESIDUAL'])
                            stream.write(image.section[:, :, :])
                    if not stream.writecomplete:
                        raise OSError(f'Incomplete residual cube: {destination}')
                finally:
                    stream.close()
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def read_integrations(self, start: int = 0, stop: int | None = None
                          ) -> dict[str, NDArray[np.float64]]:
        """Read named planes for a bounded logical integration range.

        :param start: inclusive zero-based logical row
        :param stop: exclusive row, default one bounded block
        :return: independent float64 arrays, keyed by plane name
        :raises ValueError: if the range is empty, outside or oversized
        """
        # An omitted stop requests at most one block, not all remaining rows.
        if stop is None:
            stop = min(start + self.max_integrations, len(self))
        # These are collection-local indices, not global FITS GSTART/GSTOP IDs.
        if not 0 <= start < stop <= len(self):
            raise ValueError('Invalid reconstruction integration range')
        # Even an explicit range may not bypass the bounded-read contract.
        if stop - start > self.max_integrations:
            raise ValueError('Reconstruction read exceeds integration limit')
        # Allocate only the requested integrations for each named plane.
        arrays = dict()
        for name in PLANES:
            arrays[name] = np.empty((stop - start,) + self.shape[1:])
        # offset tracks the first logical row of each product in this list.
        offset = 0
        for filename, shape in zip(self.current_infiles, self.shapes):
            # Intersect the request with this product's logical row interval.
            begin = max(start, offset)
            end = min(stop, offset + shape[0])
            # Skip files outside the requested range without opening them.
            if begin < end:
                with fits.open(filename, memmap=False) as hdul:
                    for name in PLANES:
                        # Select the named plane without evaluating full .data.
                        image = cast(fits.ImageHDU, hdul[name])
                        # Translate collection-local rows to file-local rows.
                        section = image.section[
                            begin - offset:end - offset, :, :]
                        # Place the section at its request-local destination.
                        arrays[name][begin - start:end - start] = section
            # Advance over skipped products too, preserving logical indexing.
            offset += shape[0]
        # Returned arrays are owned by the caller; edits are not auto-saved.
        return arrays

    def iter_integrations(self) -> Iterator[
            tuple[slice, dict[str, NDArray[np.float64]]]]:
        """Yield one block at a time.

        :return: logical row slices and aligned plane dictionaries
        """
        # Yield boundaries follow the saved products, not arbitrary new splits.
        offset = 0
        for shape in self.shapes:
            # The previous exclusive stop becomes this block's inclusive start.
            stop = offset + shape[0]
            # A generator performs this read only when the caller advances it.
            yield slice(offset, stop), self.read_integrations(offset, stop)
            # Store only the next coordinate, not previously yielded arrays.
            offset = stop

    def read_pixel(self, row: int, column: int
                   ) -> dict[str, NDArray[np.float64]]:
        """Read residual and validity histories for one detector pixel.

        :param row: detector row
        :param column: detector column
        :return: RESIDUAL and VALID vectors in logical integration order
        :raises MemoryError: if the history workspace exceeds the math budget
        """
        # Reserve float64 histories plus fitting scratch with a 12x allowance.
        if len(self) * 8 * 12 > self.max_math_memory_gb * 1024 ** 3:
            raise MemoryError('Pixel baseline history exceeds math budget')
        # Baseline fitting needs only residuals and the fit-validity history.
        arrays = dict()
        for name in ('RESIDUAL', 'VALID'):
            # A full pixel history is a vector, never a full detector cube.
            arrays[name] = np.empty(len(self))
        # Read each product separately into the concatenated pixel history.
        offset = 0
        for filename, shape in zip(self.current_infiles, self.shapes):
            with fits.open(filename, memmap=False) as hdul:
                for name in arrays:
                    # Keep row/column fixed while selecting this file's frames.
                    image = cast(fits.ImageHDU, hdul[name])
                    # Only one pixel's samples are loaded from this block.
                    arrays[name][offset:offset + shape[0]] = image.section[
                        :, row, column]
            # Continue writing after the history already collected.
            offset += shape[0]
        # The caller can fit across block boundaries without partial baselines.
        return arrays

    def save(self, filename: str, table_file: str) -> None:
        """Publish the manifest last, binding it to an exact coefficient table.

        :param filename: JSON manifest path
        :param table_file: existing CSV with one coefficient row per frame
        :return: None; atomically replace the completion manifest
        :raises ValueError: if coefficient rows do not match integrations
        """
        # The table must supply exactly one coefficient row per integration.
        if len(Table.read(table_file, format='ascii.csv')) != len(self):
            raise ValueError('Coefficient table does not match product rows')
        # Revalidate disk contents rather than trusting a previous read.
        type(self)(self.current_infiles, self.max_integrations,
                   self.max_math_memory_gb)
        # The manifest is the completion marker, not another image product.
        document: dict[str, object] = dict()
        # Version the reference/table contract independently of FITS RCPVER.
        document['version'] = 1
        # Absolute paths permit reopening from another working directory.
        document['files'] = [str(Path(path).resolve())
                     for path in self.current_infiles]
        # Retain expected identities as well as paths to detect replacements.
        document['products'] = [header['PRODUCT'] for header in self.headers]
        # Logical geometry must still agree when the files are reopened.
        document['shape'] = self.shape
        # Reopened readers inherit the limits used by the producing workflow.
        document['max_integrations'] = self.max_integrations
        document['max_math_memory_gb'] = self.max_math_memory_gb
        # Bind these block references to one specific coefficient CSV.
        document['table'] = str(Path(table_file).resolve())
        # Hash exact CSV bytes so a rewritten table invalidates this manifest.
        document['table_sha256'] = hashlib.sha256(
            Path(table_file).read_bytes()).hexdigest()
        # Publication may be requested in a previously unused output directory.
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Stage beside the final manifest for atomic filesystem replacement.
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
        try:
            # fdopen owns the descriptor and closes it after serialization.
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(document, stream)
            # Readers see either the previous complete manifest or this one.
            os.replace(temporary, destination)
        finally:
            # Remove unfinished metadata if serialization/replacement fails.
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, filename: str) -> tuple['ReconstructionProducts', Table]:
        """Reopen validated products and their matching coefficient table.

        :param filename: saved completion manifest
        :return: bounded product collection and coefficient table
        :raises ValueError: if provenance, coverage or table integrity differs
        """
        # Read references first; image payloads remain on disk throughout load.
        with open(filename) as stream:
            document = json.load(stream)
        # Reject unknown formats before interpreting paths or coverage fields.
        if document['version'] != 1:
            raise ValueError('Unsupported reconstruction manifest')
        # The saved path identifies the exact table belonging to these blocks.
        table_file = document['table']
        # Verify bytes before returning coefficients from a possibly newer run.
        digest = hashlib.sha256(Path(table_file).read_bytes()).hexdigest()
        if digest != document['table_sha256']:
            raise ValueError('Reconstruction coefficient table changed')
        # Recheck FITS layout, detector identity and complete jump coverage.
        result = cls(document['files'], document['max_integrations'],
                     document['max_math_memory_gb'])
        # Existing readable files are insufficient if they describe other data.
        if (list(result.shape) != document['shape']
                or [header['PRODUCT'] for header in result.headers]
                != document['products']):
            raise ValueError('Reconstruction manifest does not match products')
        # Only the small coefficient table is materialized during reopening.
        table = Table.read(table_file, format='ascii.csv')
        # A matching hash does not replace the row-to-integration count check.
        if len(table) != len(result):
            raise ValueError('Incomplete reconstruction coefficient table')
        # Consumers receive bounded image access and its aligned coefficients.
        return result, table


# =============================================================================
# Start of code
# =============================================================================
if __name__ == '__main__':
    print('Hello World!')

# =============================================================================
# End of code
# =============================================================================