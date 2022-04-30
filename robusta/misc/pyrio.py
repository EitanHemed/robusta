# TODO: Selection of R mirror for session.
# TODO: How to install R packages without operating as administrator.

"""
pyrio is the module that contains the link to the R backend of robusta.
pyrio stand for Python-R Interface Object.
pyrio is in charge of the following:
    - Starting an R session.
    - Transferring objects into the R environment and back.
"""
import os
import warnings
from tqdm import tqdm
from rpy2.situation import get_r_home
import numpy as np

# Patch for Windows, see https://github.com/rpy2/rpy2/issues/796#issuecomment-872364985
if os.name == 'nt':
    os.environ['R_HOME'] = get_r_home()

# Now we can import robjects
from rpy2.robjects import pandas2ri, numpy2ri, packages, rinterface
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.conversion import localconverter

numpy2ri.activate()
# warn_if_outdated('robusta', __version__)

# TODO - add functionality to list R envirnoment and select based on others.
# TODO - add error handling of importing a missing library?
# TODO - limit 'doMC' to non-windows OS
#  Avoid re-imports (e.g., use a set).

BUILT_IN_R_PACKAGES = ['base', 'utils', 'stats', 'datasets', ]
THIRD_PARTY_R_PACKAGES = ['generics', 'broom',
                          'afex', 'BayesFactor',
                          'datarium',
                          'effsize', 'emmeans', 'lme4',
                          'ppcor', 'psych',
                          'ARTool', 'rstatix'
                          # 'dplyr',  'tibble', 'tidyr', 'Matrix',
                          # 'backports',
                          ]


class PyRIO:
    # TODO - find out if we can use an existing singelton implementation
    instances_count = 0  # This is a static counter
    n_cpus = os.cpu_count()

    def __init__(self, cran_mirror_idx=1):
        """
        @rtype: A singleton:
        Python-R I/O. Used throughout robusta to pass objects to R/Python
        from Python/R.
        """

        self.cran_mirror_idx = cran_mirror_idx

        if PyRIO.instances_count == 1:  # Check if the number of instances present are more than one.
            # del self
            warnings.warn(
                "A PyRIO object has already been initialized.")
            # return
        PyRIO.instances_count += 1

        self.rpackages = packages
        self.rinterface = rinterface
        self._get_r_utils()
        self.rpackages.utils.chooseCRANmirror(
            ind=1)

        self.get_required_rpackages()

    def get_required_rpackages(self):
        print("Initializing robusta. Please wait.")

        for pkg in tqdm(BUILT_IN_R_PACKAGES,
                        desc="Importing built-in R packages", ):
            self.import_package(pkg, built_in=True)

        for pkg in tqdm(THIRD_PARTY_R_PACKAGES,
                        desc="Importing third-party R packages", ):
            self.import_package(pkg, built_in=False)

    def import_package(self, pkg, built_in=False):
        import_func = (self._import_built_in_r_package if built_in
                       else self._import_third_party_r_package)
        setattr(self.rpackages, pkg, import_func(pkg))

    def _import_built_in_r_package(self, pack):
        return self.rpackages.importr(pack)

    def _import_third_party_r_package(self, pack):
        """This utility checks whether the package is installed and only then imports the package"""
        try:
            return self.rpackages.importr(pack)
        except packages.PackageNotInstalledError:
            self._install_r_package(pack)
            return self.rpackages.importr(pack)

    #
    #         self._verify_if_r_package_installed(pack)
    #         return self.rpackages.importr(pack)
    #     return self.rpackages.importr(pack)
    #
    # def _verify_if_r_package_installed(self, pack):
    #     if not self.rpackages.isinstalled(pack):
    #         self._install_r_package(pack)

    def _install_r_package(self, pack):
        print(f'Installing: {pack}...')
        self.rpackages.utils.install_packages(
            pack, dependencies=np.array(("Depends", "Imports", "Enhances")),
            Ncpus=self.n_cpus)

    def _get_r_utils(self):
        setattr(self.rpackages, 'utils',
                self._import_built_in_r_package('utils'))

    def get_imported_packages(self):
        raise NotImplementedError
