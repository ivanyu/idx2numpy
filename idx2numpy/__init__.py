# -*- coding: utf-8 -*-

"""
idx2numpy package provides a tool for converting files from IDX format to
numpy.ndarray. You can meet files in IDX format, e.g. when you're going
to read the MNIST database of handwritten digits provided by Yann LeCun at
http://yann.lecun.com/exdb/mnist/
The description of IDX format also can be found on this page.
"""

from __future__ import absolute_import

from .converters import convert_from_string
from .converters import convert_from_file
from .converters import convert_to_string
from .converters import convert_to_file
from .FormatError import FormatError

from .version import __version__

__all__ = ['convert_from_string', 'convert_from_file', 'FormatError']
