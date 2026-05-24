# Generation Module
"""
Grounded multimodal generation with citation support.
Packs retrieved text chunks and images into prompt for GLM-4V.
"""
from .generator import GroundedGenerator

__all__ = ["GroundedGenerator"]
