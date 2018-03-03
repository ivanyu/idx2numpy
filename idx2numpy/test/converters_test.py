# -*- coding: utf-8 -*-

import sys
import unittest
import idx2numpy
import contextlib
import numpy as np
import os
import struct

try:
    from StringIO import StringIO as BytesIO  # for python 2.5
except ImportError:
    from io import BytesIO

# unittest in Python 2.6 and lower doesn't have assertSequenceEqual method,
# so simple alternative is provided.
if sys.version_info < (2, 7):
    class TestCaseBase(unittest.TestCase):
        @staticmethod
        def _to_list(nd):
            return [x for x in nd]

        def assertSequenceEqual(self, seq1, seq2):
            self.assertEquals(list(seq1), list(seq2))
else:
    class TestCaseBase(unittest.TestCase):
        @staticmethod
        def _to_list(nd):
            return [x for x in nd]


class TestConvertFromFile(TestCaseBase):
    def setUp(self):
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'files')

    def test_empty_file_on_disk(self):
        file = os.path.join(self.files_dir, 'empty.idx')
        self.assertRaises(idx2numpy.FormatError,
                          idx2numpy.convert_from_file, file)

    def test_correct_file_on_disk(self):
        file = os.path.join(self.files_dir, 'correct.idx')
        self.assertSequenceEqual(
            [0x0A, 0x0B, 0x0C],
            self._to_list(idx2numpy.convert_from_file(file)))


class TestConvertFromString(TestCaseBase):
    def test_empty_string(self):
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string, b'')

    def test_incorrect_magic_number(self):
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string, b'\x00\x00')
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string, b'\x01\x00\x08\x00')
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string, b'\x00\x01\x08\x00')
        # Incorrect type code.
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string, b'\x00\x00\x01\x00')
        # Incorrect dimension size.
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string,
            b'\x00\x00\x08\x01\x00\x00\x00')
        # Incorrect data length.
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string,
            b'\x00\x00\x08\x01\x00\x00\x00\x02\x01')
        # Superfluous data
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_from_string,
            b'\x00\x00\x08\x01\x00\x00\x00\x02\x01\x02\x03\x04')

    def test_correct(self):
        # Unsigned byte.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x08\x01\x00\x00\x00\x03' +
            b'\x0A' +
            b'\x0B' +
            b'\xFF')
        self.assertEqual(np.ndim(result), 1)
        self.assertEqual(np.shape(result), (3,))
        self.assertSequenceEqual(
            self._to_list(result),
            [0x0A, 0x0B, 0xFF])

        # Signed byte.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x09\x01\x00\x00\x00\x04' +
            b'\xFE' +
            b'\xFF' +
            b'\x00' +
            b'\xAA')
        self.assertEqual(np.ndim(result), 1)
        self.assertEqual(np.shape(result), (4,))
        self.assertSequenceEqual(
            self._to_list(result),
            [-2, -1, 0x00, -86])

        # Short.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x0B\x01\x00\x00\x00\x02' +
            b'\xF0\x05' +
            b'\x00\xFF')
        self.assertEqual(np.ndim(result), 1)
        self.assertEqual(np.shape(result), (2,))
        self.assertSequenceEqual(
            self._to_list(result),
            [-4091, 255])

        # Integer.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x0C\x01\x00\x00\x00\x03' +
            b'\x00\xFF\x00\xFF' +
            b'\x80\x00\x00\x00' +
            b'\x00\x00\x00\x00')
        self.assertEqual(np.ndim(result), 1)
        self.assertEqual(np.shape(result), (3,))
        self.assertSequenceEqual(
            self._to_list(result),
            [0x00FF00FF, -0x80000000, 0x00])

        # Float.
        # So fat, no tests.

        # Double.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x0E\x01\x00\x00\x00\x05' +
            b'\x3F\xF0\x00\x00\x00\x00\x00\x00' +
            b'\x40\x00\x00\x00\x00\x00\x00\x00' +
            b'\xC0\x00\x00\x00\x00\x00\x00\x00' +
            b'\x00\x00\x00\x00\x00\x00\x00\x00' +
            b'\x80\x00\x00\x00\x00\x00\x00\x00')
        self.assertEqual(np.ndim(result), 1)
        self.assertEqual(np.shape(result), (5,))
        self.assertSequenceEqual(
            self._to_list(result),
            [1.0, 2.0, -2.0, 0.0, -0.0])


class TestConvertToString(TestCaseBase):
    def test_empty_array(self):
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string, np.array([]))

    def test_unsupported_ndarray_formats(self):
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([True, False], dtype='bool_'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='int64'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='uint16'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='uint32'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='uint64'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='float16'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='complex64'))
        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string,
            np.array([1], dtype='complex128'))

    def test_very_high_dimensional_ndarray(self):
        HIGH_DIMENSIONS = 256

        # Generate a high dimensional array containing 1 element
        high_dim_arr = 1
        for i in range(HIGH_DIMENSIONS):
            high_dim_arr = [high_dim_arr]

        self.assertRaises(
            idx2numpy.FormatError,
            idx2numpy.convert_to_string, np.array(high_dim_arr))

    def test_correct(self):
        # Unsigned byte.
        result = idx2numpy.convert_to_string(
            np.array([0x0A, 0x0B, 0xFF], dtype='uint8'))
        self.assertEqual(result,
                         b'\x00\x00\x08\x01\x00\x00\x00\x03' +
                         b'\x0A' +
                         b'\x0B' +
                         b'\xFF')

        # Signed byte.
        result = idx2numpy.convert_to_string(
            np.array([-2, -1, 0x00, -86], dtype='int8'))
        self.assertEqual(result,
                         b'\x00\x00\x09\x01\x00\x00\x00\x04' +
                         b'\xFE' +
                         b'\xFF' +
                         b'\x00' +
                         b'\xAA')

        # Short.
        result = idx2numpy.convert_to_string(
            np.array([-4091, 255], dtype='int16'))
        self.assertEqual(result,
                         b'\x00\x00\x0B\x01\x00\x00\x00\x02' +
                         b'\xF0\x05' +
                         b'\x00\xFF')

        # Integer.
        result = idx2numpy.convert_to_string(
            np.array([0x00FF00FF, -0x80000000, 0x00], dtype='int32'))
        self.assertEqual(result,
                         b'\x00\x00\x0C\x01\x00\x00\x00\x03' +
                         b'\x00\xFF\x00\xFF' +
                         b'\x80\x00\x00\x00' +
                         b'\x00\x00\x00\x00')

        # Float.
        # No less fat, still no tests.

        # Double.
        result = idx2numpy.convert_to_string(
            np.array([1.0, 2.0, -2.0, 0.0, -0.0], dtype='float64'))
        self.assertEqual(result,
                         b'\x00\x00\x0E\x01\x00\x00\x00\x05' +
                         b'\x3F\xF0\x00\x00\x00\x00\x00\x00' +
                         b'\x40\x00\x00\x00\x00\x00\x00\x00' +
                         b'\xC0\x00\x00\x00\x00\x00\x00\x00' +
                         b'\x00\x00\x00\x00\x00\x00\x00\x00' +
                         b'\x80\x00\x00\x00\x00\x00\x00\x00')

        # Large array
        large_length_bytes = b'\x00\x01\x00\x00'
        large_length = struct.unpack('>I', large_length_bytes)[0]
        result = idx2numpy.convert_to_string(
            np.zeros(large_length, dtype='uint8'))
        self.assertEqual(result,
                         b'\x00\x00\x08\x01' + large_length_bytes +
                         b'\x00' * large_length)


class TestConvertToFile(TestCaseBase):
    def setUp(self):
        self._test_output_file = '.test'

        # Unsigned byte.
        self._ndarr_to_convert = np.array([0x0A, 0x0B, 0xFF], dtype='uint8')
        self._expected = (b'\x00\x00\x08\x01\x00\x00\x00\x03' +
                          b'\x0A' +
                          b'\x0B' +
                          b'\xFF')

    def tearDown(self):
        try:
            os.remove(self._test_output_file)
        except:
            pass

    def test_correct(self):
        with contextlib.closing(BytesIO()) as bytesio:
            idx2numpy.convert_to_file(bytesio, self._ndarr_to_convert)
            self.assertEqual(bytesio.getvalue(), self._expected)

    def test_correct_with_filename_argument(self):
        idx2numpy.convert_to_file(self._test_output_file, self._ndarr_to_convert)

        with open(self._test_output_file, 'rb') as fp:
            read_bytes = fp.read()
            self.assertEqual(read_bytes, self._expected)


if __name__ == '__main__':
    unittest.main()
