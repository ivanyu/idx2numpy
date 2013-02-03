import unittest
import idx2numpy
import numpy as np
import os
import struct


def _to_list(nd):
    return [x for x in nd]


class TestConvertFromFile(unittest.TestCase):

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
            _to_list(idx2numpy.convert_from_file(file)))


class TestConvertFromString(unittest.TestCase):

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
        self.assertEquals(np.ndim(result), 1)
        self.assertEquals(np.shape(result), (3,))
        self.assertSequenceEqual(
            _to_list(result),
            [0x0A, 0x0B, 0xFF])

        # Signed byte.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x09\x01\x00\x00\x00\x04' +
            b'\xFE' +
            b'\xFF' +
            b'\x00' +
            b'\xAA')
        self.assertEquals(np.ndim(result), 1)
        self.assertEquals(np.shape(result), (4,))
        self.assertSequenceEqual(
            _to_list(result),
            [-2, -1, 0x00, -86])

        # Short.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x0B\x01\x00\x00\x00\x02' +
            b'\xF0\x05' +
            b'\x00\xFF')
        self.assertEquals(np.ndim(result), 1)
        self.assertEquals(np.shape(result), (2,))
        self.assertSequenceEqual(
            _to_list(result),
            [-4091, 255])

        # Integer.
        result = idx2numpy.convert_from_string(
            b'\x00\x00\x0C\x01\x00\x00\x00\x03' +
            b'\x00\xFF\x00\xFF' +
            b'\x80\x00\x00\x00' +
            b'\x00\x00\x00\x00')
        self.assertEquals(np.ndim(result), 1)
        self.assertEquals(np.shape(result), (3,))
        self.assertSequenceEqual(
            _to_list(result),
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
        self.assertEquals(np.ndim(result), 1)
        self.assertEquals(np.shape(result), (5,))
        self.assertSequenceEqual(
            _to_list(result),
            [1.0, 2.0, -2.0, 0.0, -0.0])


if __name__ == '__main__':
    unittest.main()
