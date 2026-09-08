"""Imports are lazy (PEP 562): ``OgStructure`` (used by diep's CLI for all structure I/O
and relaxation/MD) is the common case, and pulling in ``GA``/``BACD``/``SymmetryFunctions``
eagerly drags in ase.mep, pandas/pyarrow, and the gpaw/dscribe probes even when nothing but
``OgStructure`` is needed. Resolving names on demand keeps ``from oganesson import
OgStructure`` cheap while every name below still imports correctly on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_MODULE_BY_NAME = {
    "OgAI": "oganesson.ogai",
    "BACD": "oganesson.descriptors",
    "SymmetryFunctions": "oganesson.descriptors",
    "OgStructure": "oganesson.ogstructure",
    "GA": "oganesson.genetic_algorithms",
    "ROSA": "oganesson.descriptors",
    "DScribeACSF": "oganesson.descriptors",
    "DScribeSOAP": "oganesson.descriptors",
    "DScribeCoulombMatrix": "oganesson.descriptors",
    "DScribeEwaldSumMatrix": "oganesson.descriptors",
    "DScribeSineMatrix": "oganesson.descriptors",
}

__all__ = tuple(sorted(_MODULE_BY_NAME))


def __getattr__(name: str):
    """Resolve a package attribute to its defining module on first access."""
    try:
        module_name = _MODULE_BY_NAME[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    try:
        return getattr(importlib.import_module(module_name), name)
    except AttributeError as ex:
        # ROSA / DScribe* are only defined on oganesson.descriptors when their optional
        # backend (gpaw / dscribe) is installed.
        raise ImportError(
            f"og:{name} is unavailable because its optional backend is not installed."
        ) from ex


def __dir__():
    return __all__


if TYPE_CHECKING:  # keep static analysis and IDEs working
    from oganesson.descriptors import (
        BACD,
        SymmetryFunctions,
        ROSA,
        DScribeACSF,
        DScribeSOAP,
        DScribeCoulombMatrix,
        DScribeEwaldSumMatrix,
        DScribeSineMatrix,
    )
    from oganesson.genetic_algorithms import GA
    from oganesson.ogai import OgAI
    from oganesson.ogstructure import OgStructure
