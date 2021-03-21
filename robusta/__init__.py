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

from .pyrio import PyRIO

pyr = PyRIO()
# from groupwise import *
from . import groupwise
from . import base
from . import formula_tools
from . import utils
from . import datasets
from . import examples

from .api import *

# from .regressions import *
# from .correlations import *
