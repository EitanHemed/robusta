# TODO - credit all!
__author__ = 'Eitan Hemed'
__author_email__ = 'Eitan.Hemed@gmail.com'

# First init a PyRIO object
from .misc.pyrio import PyRIO
pyr = PyRIO()

from . import groupwise #, correlations, regressions
from .misc.datasets import get_available_datasets, load_dataset

