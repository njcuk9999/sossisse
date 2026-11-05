---
description: "Python grammar conventions for SOSSISSE source and tests."
applyTo: '**/*.py'
---

# SOSSISSE Python Grammar

## Module Template

- Follow the existing Python module template: shebang, UTF-8 declaration,
  module docstring, imports, and the sections shown below in this order.
- Import `base` from the current project's core package, never another
  project's package. Set `__NAME__` to the actual dotted module name.
- Put classes in `Define classes`, above `Define functions`. Put private
  module-level helper functions in `Define functions`.
- Preserve the standard section dividers and executable footer.

```python
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


# =============================================================================
# Define functions
# =============================================================================


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
```

## Strings and Line Length

- Prefer single-quoted strings. Use double quotes when needed for embedded
  apostrophes or the syntax of the string content.
- Use triple double quotes for docstrings.
- Keep every Python line strictly below 80 characters, including comments,
  docstrings, imports, and tests: 79 characters is the maximum.
- Never use backslash line continuations. Use implicit continuation within
  parentheses, aligning continued arguments with the first argument.
- Split long strings at logical boundaries using implicit concatenation.
- Split complicated conditions and calculations into meaningful named parts
  rather than compressing names or nesting expressions to meet the limit.
- Break before binary operators when wrapping expressions.

## Dictionaries

- Prefer `dict()` followed by explicit assignments over dictionary literals.
- Use `dict(name=value)` for keyword containers when keys are identifiers.
- Preserve natural variable-key mappings; do not replace them with fixed
  argument lists solely for styling.

## Delegated Calls

- Collect long forwarded positional arguments in a context-named container,
  such as `extract_args`, and call with `*extract_args`.
- Collect forwarded keyword arguments using `extract_kwargs = dict(...)`
  and call with `**extract_kwargs`.
- Reserve bare `args` and `kwargs` for actual function inputs or parser output.
- Keep delegated calls and multi-target unpacking as separate statements.
- Do not add forwarding containers to ordinary short calls without a benefit.

```python
extract_args = [filename, trace, wavelength]
extract_kwargs = dict(save_results=True, overwrite=False)
outputs = extractor.run(*extract_args, **extract_kwargs)
flux, uncertainty = outputs
```

## Signatures

- Keep short signatures on one line. Align wrapped parameters with the first
  parameter, preserving four-space indentation for the function body.
- Prefer meaningful parameter names; do not shorten them to fit a line.