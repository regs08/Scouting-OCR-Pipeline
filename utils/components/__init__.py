"""
Components package for the OCR pipeline.
Contains reusable components that can be configured and added to different pipeline stages.
"""

from .directory_creator import DirectoryCreator

__all__ = [
    'DirectoryCreator',
] 