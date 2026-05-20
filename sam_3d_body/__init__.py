"""Vendored SAM-3D modules used by sam-3d-body-funscript.

This package intentionally avoids importing the original estimator or MHR
runtime. The funscript model only needs the backbone, prompt encoder, decoder,
and small utility modules.
"""

__version__ = "funscript-vendor"
