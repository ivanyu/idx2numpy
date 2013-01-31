from __future__ import absolute_import
from __future__ import with_statement

from .FormatError import FormatError


def convert_from_file(file):
    '''
    Reads the content of file in IDX format, converts it into numpy.ndarray and
    returns it.
    file is a file-like object (with read() method) or a file name.
    '''
    return None


def convert_from_string(idx_string):
    '''
    Converts string which presents file in IDX format into numpy.ndarray and
    returns it.
    '''
    return None


def _internal_convert(input):
    '''
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    '''
    return None
