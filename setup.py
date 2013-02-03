#!/usr/bin/env python
"""
A Python package which provides tools to convert files from IDX format
(described at http://yann.lecun.com/exdb/mnist/) into numpy.ndarray.
"""

from setuptools import setup
import os

README = open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

# Allow setup.py to be run from any path.
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

PACKAGE = "idx2numpy"
TESTS_PACKAGE = "idx2numpy.test"
NAME = "idx2numpy"
DESCRIPTION = ("A Python package which provides tools to convert files from "
               "IDX format (described at http://yann.lecun.com/exdb/mnist/) "
               "into numpy.ndarray.")
AUTHOR = "Ivan Yurchenko"
AUTHOR_EMAIL = "ivan0yurchenko@gmail.com"
URL = "https://github.com/ivanyu/idx2numpy"
VERSION = __import__(PACKAGE).__version__

MAINTAINER = AUTHOR
MAINTAINER_EMAIL = AUTHOR_EMAIL

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license="MIT License",
    url=URL,
    packages = [PACKAGE, TESTS_PACKAGE],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2.5",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    zip_safe=False,
    test_suite="idx2numpy.test",
    keywords='mnist numpy'
)
