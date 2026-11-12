"""Custom constant types for SOSSISSE."""

from copy import copy
from collections.abc import Mapping
from numbers import Integral, Real
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


class DQThresholdConst(Const):
    """Preserve and strictly validate an occurrence fraction, without casts."""

    @staticmethod
    def fraction(value: object) -> float:
        """Return a finite real fraction in [0, 1], excluding Booleans.

        :param value: candidate maximum permitted occurrence fraction
        :return: validated fraction as a float
        :raises SossisseConstantException: for non-real or invalid fractions
        """
        # Do not let bools, strings or NaN become meaningful science settings.
        emsg = 'DQ threshold must be a finite real fraction in [0, 1]'
        if isinstance(value, bool) or not isinstance(value, Real):
            raise exceptions.SossisseConstantException(emsg)
        # Compare concrete floats after checking the original type, not a
        # coerced string/bool. Extremely large integers are invalid limits.
        try:
            fraction = float(value)
        except OverflowError as error:
            raise exceptions.SossisseConstantException(emsg) from error
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise exceptions.SossisseConstantException(emsg)
        return fraction

    def validate(self, test_value: Any = None, quiet: bool = False,
                 source: str | None = None) -> Any:
        """Validate stored/candidate fractions using the Const convention.

        :param test_value: candidate, or None to validate the stored value
        :param quiet: accepted for Const compatibility
        :param source: accepted for Const compatibility
        :return: True for stored validation, otherwise the validated fraction
        :raises SossisseConstantException: if the fraction is invalid
        """
        value = self.value if test_value is None else test_value
        fraction = self.fraction(value)
        if test_value is None:
            self.true_value = fraction
            return True
        return fraction

    def copy(self, source: str | None = None) -> 'DQThresholdConst':
        """Copy without losing strict validation on a provenance change.

        :param source: replacement provenance, or None to preserve it
        :return: independent constant retaining this validation subclass
        """
        constant = copy(self)
        if source is not None:
            constant.source = source
        return constant


class ValidDQConst(Const):
    """Preserve exact-value DQ whitelists as mappings or legacy lists."""

    @staticmethod
    def configuration(value: object) -> dict[int, float] | list[int]:
        """Validate accepted exact DQ categories and optional fraction limits.

        :param value: nonempty mapping of integer DQ to maximum fractions,
                      or a distinct nonempty legacy list of integer DQ values
        :return: isolated configuration, without coercing keys or list values
        :raises SossisseConstantException: for invalid keys, lists or limits
        """
        # Keys are exact values, not bit masks; never coerce strings or floats.
        if not isinstance(value, (Mapping, list)) or not value:
            emsg = 'VALID_DQ must be a nonempty mapping or distinct list'
            raise exceptions.SossisseConstantException(emsg)
        if any(isinstance(key, bool) or not isinstance(key, Integral)
               or key < 0 for key in value):
            emsg = 'VALID_DQ keys must be nonnegative integers, not Booleans'
            raise exceptions.SossisseConstantException(emsg)
        if isinstance(value, list):
            if len(set(value)) != len(value):
                emsg = 'VALID_DQ list must contain distinct integers'
                raise exceptions.SossisseConstantException(emsg)
            return list(value)
        return {int(key): DQThresholdConst.fraction(limit)
                for key, limit in value.items()}

    def validate(self, test_value: Any = None, quiet: bool = False,
                 source: str | None = None) -> Any:
        """Validate without converting a mapping into a list of its keys.

        :param test_value: candidate, or None to validate the stored value
        :param quiet: accepted for Const compatibility
        :param source: accepted for Const compatibility
        :return: True for stored validation, otherwise an isolated dict/list
        :raises SossisseConstantException: for invalid DQ configuration
        """
        value = self.value if test_value is None else test_value
        validated = self.configuration(value)
        if test_value is None:
            self.true_value = validated
            return True
        return validated

    def copy(self, source: str | None = None) -> 'ValidDQConst':
        """Copy this subclass and isolate mutable DQ configuration.

        :param source: replacement provenance, or None to preserve it
        :return: independent constant retaining dict/list validation
        """
        constant = copy(self)
        if source is not None:
            constant.source = source
        constant.value = self.configuration(self.value)
        if self.true_value is not None:
            constant.true_value = self.configuration(self.true_value)
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


