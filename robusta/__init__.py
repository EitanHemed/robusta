# TODO - credit all!

__version__ = '0.0.1'
__author__ = 'Eitan Hemed'
__author_email__ = 'Eitan.Hemed@gmail.com'

#### Notes:
# Currently works with:
# #numpy 1.15.4
# #pandas 0.23.4
# # rpy2 2.9.4
# # and you should manually install dateutil:
#  use conda install -c anaconda python-dateutil

# First init a PyRIO object
from .misc.pyrio import PyRIO
pyr = PyRIO()

from . import groupwise #, correlations, regressions
#from . import api
from .misc.datasets import get_available_datasets, load_dataset

