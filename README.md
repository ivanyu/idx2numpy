idx2numpy
=========

[![Build Status](https://travis-ci.org/ivanyu/idx2numpy.svg?branch=master)](https://travis-ci.org/ivanyu/idx2numpy)

`idx2numpy` package provides a tool for converting files to and from
IDX format to `numpy.ndarray`. You can meet files in IDX format,
e.g. when you're going to read the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) provided by Yann LeCun.

The description of IDX format also can be found on this page.

Installation
============

The easiest way to install is by using pip to pull it from PyPI:

    pip install idx2numpy

You can also clone the Git repository from Github and install 
the package manually:

    git clone https://github.com/ivanyu/idx2numpy.git
    python setup.py install

Usage
=====

```python
import idx2numpy

# Reading
ndarr = idx2numpy.convert_from_file('myfile.idx')

f_read = open('myfile.idx', 'rb')
ndarr = idx2numpy.convert_from_file(f_read)

s = f_read.read()
ndarr = idx2numpy.convert_from_string(s)

# Writing    
idx2numpy.convert_to_file('myfile_copy.idx', ndarr)

f_write = open('myfile_copy2.idx', 'wb')
idx2numpy.convert_to_file(f_write, ndarr)

s = convert_to_string(ndarr)
```

License
=======
MIT license (see `LICENSE` file)
