"""
HMM Tracer - A package for tracing axons using Hidden Markov Models.

This package provides tools for tracing axons in microscopy images
using Hidden Markov Models from the alva_machinery package.
"""

__version__ = '0.1.0'

from hmm_tracer.core.preprocessing import ImagePreprocessor
from hmm_tracer.core.tracing import AxonTracer
from hmm_tracer.core.graph import NeuriteGraph