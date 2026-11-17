"""Custom constant types for SOSSISSE."""

from copy import copy
from numbers import Real
import math
from typing import Any

from aperocore.constants import constant_functions

from sossisse.core import exceptions


Const = constant_functions.Const


class MathMemoryConst(Const):
    """Preserve strict positive finite GiB limits through YAML and copies."""

    @staticmethod
    def memory(value: object) -> float:
        """Validate the original value before any framework dtype conversion.

        :param value: math budget in GiB (1024**3 bytes), not gigabits
        :return: positive finite floating budget
        :raises SossisseConstantException: for Boolean/non-real/invalid limits
        """
        emsg = ('GENERAL.MAX_MATH_MEMORY_GB must be a positive finite real '
                'number in GiB (1024**3 bytes), not a Boolean')
        if isinstance(value, bool) or not isinstance(value, Real):
            raise exceptions.SossisseConstantException(emsg)
        # Reject huge integer overflow as a configuration error, not a cast.
        try:
            memory = float(value)
        except OverflowError as error:
            raise exceptions.SossisseConstantException(emsg) from error
        if not math.isfinite(memory) or memory <= 0:
            raise exceptions.SossisseConstantException(emsg)
        return memory

    def validate(self, test_value: Any = None, quiet: bool = False,
                 source: str | None = None) -> Any:
        """Validate stored/candidate limits using the Const convention.

        :param test_value: candidate, or None to validate the stored value
        :param quiet: accepted for Const compatibility
        :param source: accepted for Const compatibility
        :return: True for stored validation, otherwise validated GiB
        :raises SossisseConstantException: for an invalid memory limit
        """
        value = self.value if test_value is None else test_value
        memory = self.memory(value)
        if test_value is None:
            self.true_value = memory
            return True
        return memory

    def copy(self, source: str | None = None) -> 'MathMemoryConst':
        """Keep strict validation when constants acquire new provenance.

        :param source: replacement provenance, or None to preserve it
        :return: independent constant retaining this validation subclass
        """
        constant = copy(self)
        if source is not None:
            constant.source = source
        return constant


class WavelengthFileConst(Const):
    """Preserve scalar and detector-specific wavelength filenames in YAML."""

    def validate(self, test_value=None, quiet=False, source=None):
        """
        Validate filenames without coercing a detector list into one string.

        :param test_value: str, list or None, candidate value; None validates
                           the stored value and updates true_value
        :param quiet: bool, accepted for compatibility with Const.validate
        :param source: str or None, accepted for Const.validate compatibility
        :return: bool for stored validation, otherwise the scalar or list copy
        :raises SossisseConstantException: if a non-null value is not a
                                          nonempty filename or filename list
        """
        # A null candidate means validate the stored configuration instead.
        value = self.value if test_value is None else test_value
        # Inspect scalar and list values uniformly without changing their type.
        filenames = value if isinstance(value, list) else [value]
        # Leave null values valid; downstream consumers decide if required.
        if value is not None:
            invalid_filenames = not filenames or any(
                not isinstance(filename, str) or not filename.strip()
                for filename in filenames)
            if invalid_filenames:
                emsg = ('WAVE_FILE must be a filename or a nonempty list '
                        'of filenames')
                raise exceptions.SossisseConstantException(emsg)
        # Stored validation follows the parent boolean-result convention.
        if test_value is None:
            self.true_value = value
            return True
        # Candidate validation returns a copy to avoid sharing mutable lists.
        return list(value) if isinstance(value, list) else value

    def copy(self, source=None):
        """
        Copy this constant while preserving detector-list values.

        :param source: str or None, replacement provenance; None delegates
                       to the parent copy implementation
        :return: WavelengthFileConst, copied constant with an isolated list
        """
        # Preserve the parent behavior when no source override is requested.
        if source is None:
            return super().copy(source=source)
        # Keep this subclass rather than rebuilding a scalar-only Const.
        constant = copy(self)
        constant.source = source
        # A copied configuration must not share its wavelength filename list.
        if isinstance(self.value, list):
            constant.value = list(self.value)
        return constant


