# this is adapted from https://github.com/astropy/astropy/blob/master/astropy/version.py
# see https://github.com/astropy/astropy/pull/10774 for a discussion on why this needed.
from __future__ import annotations

try:
    try:
        from ._dev_version import version
    except ImportError:
        from ._version import version
except Exception:
    import warnings

    warnings.warn(
        (
            "Could not determine lightcurves version. This indicates "
            "a broken installation. Please install lightcurves from "
            "the local git repository."
        ),
        category=ImportWarning,  # or UserWarning if you prefer
        stacklevel=2,  # point at the *importer*, not here
    )
    del warnings
    version = "0.0.0"

__version__ = version
