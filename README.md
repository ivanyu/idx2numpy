idx2numpy
=========

A Python package which provides tools to convert files from IDX format
(described [here](http://yann.lecun.com/exdb/mnist/)) into numpy.ndarray.

Installation
============

The easiest way to install is by using pip to pull it from PyPI:

    pip install tweepy

You can also clone the Git repository from Github and install 
the package manually:

    git clone https://github.com/ivanyu/idx2numpy.git
    python setup.py install

** Note: ** Unfortunately, work with Python 3 hasn't been tested yet.

Usage
=====

    import idx2numpy
    ndarr = idx2numpy.convert_from_file('myfile.idx')
    
    f = open('myfile.idx', 'rb)
    ndarr = idx2numpy.convert_from_file(f)
    
    s = f.read()
    ndarr = idx2numpy.convert_from_string(s)
