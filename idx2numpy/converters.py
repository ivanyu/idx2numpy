from __future__ import absolute_import
from __future__ import with_statement

import struct
import numpy
import operator
import contextlib
from six import string_types as six_string_types
from six.moves import reduce
try:
    from StringIO import StringIO as BytesIO  # for python 2.5
except ImportError:
    from io import BytesIO

from .FormatError import FormatError


def convert_from_file(file):
    '''
    Reads the content of file in IDX format, converts it into numpy.ndarray and
    returns it.
    file is a file-like object (with read() method) or a file name.
    '''
    if isinstance(file, six_string_types):
        with open(file, 'rb') as f:
            return _internal_convert(f)
    else:
        return _internal_convert(file)


def convert_from_string(idx_string):
    '''
    Converts string which presents file in IDX format into numpy.ndarray and
    returns it.
    '''
    with contextlib.closing(BytesIO(idx_string)) as sio:
        return _internal_convert(sio)


def _internal_convert(input):
    '''
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    '''
    '''
    Converts file in IDX format provided by file-like input into numpy.ndarray
    and returns it.
    '''

    # Possible data types.
    # Keys are IDX data type codes.
    # Values: (ndarray data type name, name for struct.unpack, size in bytes).
    DATA_TYPES = {
        0x08: ('ubyte', 'B', 1),
        0x09: ('byte', 'b', 1),
        0x0B: ('int16', 'h', 2),
        0x0C: ('int32', 'i', 4),
        0x0D: ('float', 'f', 4),
        0x0E: ('double', 'd', 8)
    }

    # Read the "magic number" - 4 bytes.
    try:
        mn = struct.unpack('>BBBB', input.read(4))
    except struct.error:
        raise FormatError(struct.error)

    # First two bytes are always zero, check it.
    if mn[0] != 0 or mn[1] != 0:
        msg = ("Incorrect first two bytes of the magic number: " +
               "0x{0:02X} 0x{1:02X}".format(mn[0], mn[1]))
        raise FormatError(msg)

    # 3rd byte is the data type code.
    dtype_code = mn[2]
    if dtype_code not in DATA_TYPES:
        msg = "Incorrect data type code: 0x{0:02X}".format(dtype_code)
        raise FormatError(msg)

    # 4th byte is the number of dimensions.
    dims = int(mn[3])

    # See possible data types description.
    dtype, dtype_s, el_size = DATA_TYPES[dtype_code]

    # 4-byte integer for length of each dimension.
    try:
        dims_sizes = struct.unpack('>'+'I'*dims, input.read(4*dims))
    except struct.error as e:
        raise FormatError('Dims sizes: {0}'.format(e))

    # Full length of data.
    full_length = reduce(operator.mul, dims_sizes, 1)

    result_array = numpy.ndarray(shape=[full_length], dtype=dtype)

    # Read data "in the line".
    unpack_str = '>' + dtype_s*full_length
    try:
        result_array[0:full_length] = struct.unpack(
            unpack_str, input.read(full_length * el_size))
    except struct.error as e:
        raise FormatError('Data: {0}'.format(e))

    # Check for superfluous data.
    if len(input.read(1)) > 0:
        raise FormatError('Superfluous data detected.')

    # Reshape data according to dimensions sizes.
    result = numpy.reshape(result_array, dims_sizes)
    return result
