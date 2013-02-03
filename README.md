idx2numpy
=========

idx2numpy package provides a tool for converting files from IDX format to
numpy.ndarray. You can meet files in IDX format, e.g. when you're going
to read the [MNIST database of handwritten digits]
(http://yann.lecun.com/exdb/mnist/) provided by Yann LeCun.

The description of IDX format also can be found on this page.


Installation
============

The easiest way to install is by using pip to pull it from PyPI:

    pip install idx2numpy

You can also clone the Git repository from Github and install 
the package manually:

    git clone https://github.com/ivanyu/idx2numpy.git
    python setup.py install

**Note:** Unfortunately, work with Python 3 hasn't been tested yet.

Usage
=====

    import idx2numpy
    ndarr = idx2numpy.convert_from_file('myfile.idx')
    
    f = open('myfile.idx', 'rb)
    ndarr = idx2numpy.convert_from_file(f)
    
    s = f.read()
    ndarr = idx2numpy.convert_from_string(s)
