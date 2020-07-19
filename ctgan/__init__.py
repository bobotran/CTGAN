# -*- coding: utf-8 -*-

"""Top-level package for ctgan."""

__author__ = 'MIT Data To AI Lab'
__email__ = 'dailabmit@gmail.com'
__version__ = '0.2.2.dev1'

from ctgan.demo import load_demo
from ctgan.synthesizer import CTGANSynthesizer
from ctgan.glow_synthesizer import CTGlowSynthesizer

__all__ = (
    'CTGANSynthesizer',
    'CTGlowSynthesizer',
    'load_demo'
)
