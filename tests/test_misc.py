#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for miscellaneous display helpers."""

import re

import pytest

from sossisse.core import misc


@pytest.mark.parametrize('object_name', [
    'WD1856-534',
    'WD 1856-534',
    'WD1856+534',
    'WD 1856+534',
    'WD\u00a01856-534',
    'WD_1856+534',
    'WD(1856)+534',
    '',
])
def test_art_object_names(object_name):
    """Object names must not crash or break the banner's alignment."""
    banner = misc.art('Linear Recon ' + object_name, 'blue', 'CYAN')
    plain_banner = re.sub(r'\x1b\[[0-9;]*m', '', banner)
    rows = plain_banner.strip('\n').splitlines()
    assert len(rows) == 5
    assert len({len(row) for row in rows}) == 1


@pytest.mark.parametrize('whitespace', [' ', '\u00a0', '\t', '\n'])
def test_art_normalizes_whitespace(whitespace):
    """Display whitespace consistently without modifying the input name."""
    assert misc.art('WD' + whitespace + '1856') == misc.art('WD 1856')


@pytest.mark.parametrize('character', [
    '_', '(', '/', '\u2605', '\u0131', '\u017f',
])
def test_art_unsupported_character(character):
    """Missing glyphs use the existing question-mark glyph."""
    assert misc.art('WD' + character + '1856') == misc.art('WD?1856')


def test_art_plus_glyph():
    """Render a plus sign explicitly rather than dropping or replacing it."""
    banner = re.sub(r'\x1b\[[0-9;]*m', '', misc.art('+'))
    rows = [row.strip() for row in banner.strip('\n').splitlines()]
    assert rows[1:4] == ['║  │  ║', '║ ─┼─ ║', '║  │  ║']