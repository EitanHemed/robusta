__version__ = '0.0.1'
__author__ = 'Eitan Hemed'
__author_email__ = 'Eitan.Hemed@gmail.com'
__all__ = ['']


#### Notes:
# Currently works with:
# #numpy 1.15.4
# #pandas 0.23.4
# # rpy2 2.9.4
# # and you should manually install dateutil:
#  use quitconda install -c anaconda python-dateutil

from itertools import repeat
from rpy2.robjects import pandas2ri, numpy2ri, packages

#from outdated import warn_if_outdated

numpy2ri.activate()
pandas2ri.activate()
#warn_if_outdated('robusta', __version__)

# TODO - add functionality to list R envirnoment and select based on others.

r_libraries_to_include = ['backports', 'tidyr',
                          'datasets', 'broom', 'afex', 'car',
                          'emmeans', 'base', 'BayesFactor', 'stats', 'effsize', 'psych']

class PyRIO:
    instances_count = 0  # This is a static counter

    def __init__(self,
                 required_packs=None, cran_mirror_idx=1):
        """
        @rtype: A singleton:
        Python-R I/O. Used throughout robusta to pass objects to R/Python from Python/R.
        """
        self.cran_mirror_idx = cran_mirror_idx
        if required_packs is None:
            required_packs = r_libraries_to_include
        if PyRIO.instances_count == 1:  # Check if the number of instances present are more than one.
            del self
            print("A PyRIO object has already been initialized. Please use existing")
            return
        PyRIO.instances_count += 1

        self.rpackages = packages
        self._get_rutils()
        self.required_rpacks = required_packs
        self.get_required_rpackages()

    def get_required_rpackages(self):
        [setattr(self.rpackages, pack, self.import_rpackage(pack)) for pack in self.required_rpacks]

    def import_rpackage(self, pack):
        """This utility checks whether the package is installed and only then imports the package"""
        self._verify_if_rpack_installed(pack)
        return self.rpackages.importr(pack)

    def _verify_if_rpack_installed(self, pack):
        if not self.rpackages.isinstalled(pack):
            self._install_rpack(pack)

    def _install_rpack(self, pack):
        self.rpackages.utils.install_packages(pack)

    def _get_rutils(self):
        """It is important to get the utils package in order to to """
        setattr(self.rpackages, 'utils', self.import_rpackage('utils'))
        self.rpackages.utils.chooseCRANmirror(ind=self.cran_mirror_idx)

pyr = PyRIO()


