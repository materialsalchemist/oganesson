import importlib as imp

from oganesson.ogstructure import OgStructure
from oganesson.native_structure import OgNativeStructure

__all__ = ["OgStructure", "OgNativeStructure"]

# Optional high-level components that require additional dependencies.
try:
    from oganesson.ogai import OgAI
    __all__.append("OgAI")
except Exception:
    pass

try:
    from oganesson.genetic_algorithms import GA
    __all__.append("GA")
except Exception:
    pass

try:
    from oganesson.descriptors import BACD, SymmetryFunctions
    __all__ += ["BACD", "SymmetryFunctions"]
except Exception:
    # descriptors require ase/pymatgen and optional backends
    pass

# Optional descriptors
try:
    imp.util.find_spec('gpaw')
    from oganesson.descriptors import ROSA
    __all__.append("ROSA")
except Exception:
    pass

try:
    imp.util.find_spec('dscribe')
    from oganesson.descriptors import DScribeACSF, DScribeSOAP, DScribeCoulombMatrix, DScribeEwaldSumMatrix, DScribeSineMatrix
    __all__ += ["DScribeACSF", "DScribeSOAP", "DScribeCoulombMatrix", "DScribeEwaldSumMatrix", "DScribeSineMatrix"]
except Exception:
    pass

__all__ = tuple(__all__)
