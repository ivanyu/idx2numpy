# -*- coding: utf-8 -*-
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
VERSION_FILE = "idx2numpy.version"
NAME = "idx2numpy"
DESCRIPTION = ("A Python package which provides tools to convert files "
               "to and from IDX format "
               "(described at http://yann.lecun.com/exdb/mnist/) "
               "into numpy.ndarray.")
AUTHOR = "Ivan Yurchenko"
AUTHOR_EMAIL = "ivan0yurchenko@gmail.com"
URL = "https://github.com/ivanyu/idx2numpy"

exec(open('%s/version.py' % PACKAGE).read())
VERSION = __version__

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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.0",
        "Programming Language :: Python :: 3.1",
        "Programming Language :: Python :: 3.2",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    zip_safe=False,
    test_suite="idx2numpy.test",
    keywords='mnist numpy',
    install_requires=[
        "numpy",
        "six"
    ]
)
