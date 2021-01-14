# TODO: Selection of R mirror for session.
# TODO: How to install R packages without operating as administrator.

"""
pyrio is the module that contains the link to the R backend of robusta.
pyrio stand for Python-R Interface Object.
pyrio is in charge of the following:
    - Starting an R session.
    - Transferring objects into the R environment and back.
"""
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri, packages, rinterface, r
#packages, rinterface, r
# from outdated import warn_if_outdated

numpy2ri.activate()
pandas2ri.activate()
# warn_if_outdated('robusta', __version__)

# TODO - add functionality to list R envirnoment and select based on others.
# TODO - add error handling of importing a missing library?
# TODO - limit 'doMC' to non-windows OS
# TODO - Write names of imported libraries alphabethically,
#  Avoid re-imports (e.g., use a set).

r_libraries_to_include = ['backports', 'tidyr', 'lme4', 'emmeans',
                          'datasets', 'broom', 'afex', 'utils',
                          'dplyr', 'datarium', 'tibble', 'generics',
                          'base', 'BayesFactor', 'stats', 'effsize', 'psych',
                          'ppcor',
                          'Matrix']


class PyRIO:
    #instances_count = 0  # This is a static counter

    def __init__(self,
                 required_packs=None, cran_mirror_idx=1):
        """
        @rtype: A singleton:
        Python-R I/O. Used throughout robusta to pass objects to R/Python from Python/R.
        """
        self.cran_mirror_idx = cran_mirror_idx
        if required_packs is None:
            required_packs = r_libraries_to_include
        #if PyRIO.instances_count == 1:  # Check if the number of instances present are more than one.
        #    del self
        #    print(
        #        "A PyRIO object has already been initialized. Please use existing")
        #    return
        #PyRIO.instances_count += 1

        self.rpackages = packages
        self.rinterface = rinterface
        self._get_r_utils()
        self.required_rpacks = required_packs
        self.get_required_rpackages()

    def get_required_rpackages(self):
        [setattr(self.rpackages, pack.lower(), self.import_r_package(pack)) for
         pack in self.required_rpacks]

    def import_r_package(self, pack):
        """This utility checks whether the package is installed and only then imports the package"""
        self._verify_if_r_package_installed(pack)
        return self.rpackages.importr(pack)

    def _verify_if_r_package_installed(self, pack):
        if not self.rpackages.isinstalled(pack):
            self._install_r_package(pack)

    def _install_r_package(self, pack):
        self.rpackages.utils.install_packages(pack)

    def _get_r_utils(self):
        """It is important to get the utils package in order to to """
        setattr(self.rpackages, 'utils', self.import_r_package('utils'))
        self.rpackages.utils.chooseCRANmirror(ind=self.cran_mirror_idx)

    def get_imported_packages(self):
        raise NotImplementedError
