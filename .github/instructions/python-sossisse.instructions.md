---
description: "Use when writing or reviewing SOSSISSE Python source and tests."
applyTo: '**/*.py'
---

# SOSSISSE Python Rules

## Layout and Imports

- Start new Python modules with `#!/usr/bin/env python` and the UTF-8 encoding
  declaration. Use four spaces for indentation.
- Apply the companion grammar instructions, including the 79-character limit.
- Group standard-library, third-party, and local imports with blank lines.
- Avoid wildcard imports. Prefer importing a module when using multiple
  functions from it, and call those functions through the module namespace.
- Use separate imports unless parentheses are needed to satisfy line length.
- Put module constants after imports and before classes and functions.
- Separate top-level definitions with two blank lines. Follow existing section
  banners for variables, classes, functions, and worker functions as needed.
- Keep instrument-specific helpers in the owning instrument module unless
  there is a concrete shared abstraction. Do not add empty sections or a
  demonstration main block to library modules.

## Comments and Docstrings

- Document modules, classes, and functions with descriptive docstrings.
- Follow local `:param`, `:return`, and `:raises` conventions. Explain units,
  array shapes, optional values, parameter mutations, filesystem effects,
  calibration matching, caching, and errors when relevant.
- Comment most executable lines in scientific and validation workflows,
  normally immediately above the statement. Explain intent and assumptions,
  not merely the statement's spelling; continuations can share one comment.
- Define trace terminology explicitly: SOSS diffraction orders, NIRSpec
  detectors (1=NRS1, 2=NRS2), or an unnumbered trace for other instruments.
- Distinguish shared SOSS reconstruction amplitudes from independently
  extracted detector light curves. Preserve detector-specific timestamps.

## Parameters and Validation

- Gather stable settings into meaningful locals near function entry. Keep
  conditional reads behind their guards and refresh after parameter mutation.
- Preserve `params.get('PREFIX')` views and use their relative `rget` reads.
  Require a value only when its consuming behavior cannot handle `None`.
- Keep framework parameters out of independent numerical helpers: unpack
  fixed-key parameter bags into named arguments. Natural variable-key mappings
  such as detector-to-result dictionaries remain appropriate.
- Preserve calibration associations, trace identity, and intermediate-file
  isolation. Never concatenate simultaneous detectors as time segments.
- Preserve existing user edits and scientific behavior during style changes.
- Add focused regression tests and run SOSSISSE tests separately from POGOS
  tests, whose dependency stubs can interfere with collection.