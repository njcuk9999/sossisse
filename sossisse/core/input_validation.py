#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate resolved SOSSISSE input FITS files against the selected mode."""
import os
from typing import Sequence

from sossisse.core import base
from sossisse.core import exceptions
from sossisse.core import io


def check_input_data_headers(mode_name: str, files: Sequence[str]):
    """Check ordered, resolved input files against the selected mode."""
    expected = base.MODE_HEADERS.get(mode_name)
    if expected is None:
        raise exceptions.SossisseConstantException(
            f'No input-header definition for instrument mode {mode_name}')
    if len(files) == 0:
        raise exceptions.SossisseFileException(
            f'No input files found for instrument mode {mode_name}')
    for filename in files:
        if not os.path.isabs(filename):
            raise exceptions.SossisseFileException(
                f'Input file must be an absolute path: {filename}')
        if not os.path.isfile(filename):
            raise exceptions.SossisseFileException(
                f'Input file does not exist: {filename}')
        header = io.load_header(filename, ext=0)
        for key, accepted in expected.items():
            value = header.get(key)
            normalized = str(value).strip().upper() if value is not None else ''
            if normalized in accepted:
                continue
            expected_text = ', '.join(sorted(accepted))
            found_text = normalized if normalized else '<missing>'
            emsg = (f'Input data does not match {mode_name}\n\n'
                    f'File: {filename}\n\n'
                    f'Header: {key}={found_text}; '
                    f'expected {expected_text}.')
            raise exceptions.SossisseFileException(emsg)