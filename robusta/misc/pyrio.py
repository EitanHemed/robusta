# TODO: Selection of R mirror for session.
# TODO: How to install R packages without operating as administrator.

"""
pyrio is the module that contains the link to the R backend of robusta.
pyrio stand for Python-R Interface Object.
pyrio is in charge of the following:
    - Starting an R session.
    - Transferring objects into the R environment and back.
"""
from os import cpu_count
import warnings
from tqdm import tqdm
from rpy2.robjects import pandas2ri, numpy2ri, packages, rinterface
from rpy2.robjects.conversion import localconverter

numpy2ri.activate()
# warn_if_outdated('robusta', __version__)

# TODO - add functionality to list R envirnoment and select based on others.
# TODO - add error handling of importing a missing library?
# TODO - limit 'doMC' to non-windows OS
#  Avoid re-imports (e.g., use a set).

required_r_libraries = ['base', 'datasets', 'stats', 'utils', 'generics', 'broom',
                       # 'dplyr',  'tibble', 'tidyr', 'Matrix',
                        # 'backports',
                        'afex', 'BayesFactor',
                         'datarium',
                         'effsize', 'emmeans', 'lme4',
                          'ppcor', 'psych',
                          'ARTool', 'rstatix'
                          ]


class PyRIO:
    # TODO - find out if we can use an existing singelton implementation
    instances_count = 0  # This is a static counter
    n_cpus = cpu_count()

    required_packs = required_r_libraries

    def __init__(self, cran_mirror_idx=1):
        """
        @rtype: A singleton:
        Python-R I/O. Used throughout robusta to pass objects to R/Python
        from Python/R.
        """
        print("Initializing robusta. Please wait.")

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
        self.get_required_rpackages()

    def get_required_rpackages(self):
        for pkg in tqdm(self.required_packs):
            self.import_package(pkg)

    def import_package(self, pkg):
        setattr(self.rpackages, pkg, self._import_r_package(pkg))

    def _import_r_package(self, pack):
        """This utility checks whether the package is installed and only then imports the package"""
        self._verify_if_r_package_installed(pack)
        return self.rpackages.importr(pack)

    def _verify_if_r_package_installed(self, pack):
        if not self.rpackages.isinstalled(pack):
            self._install_r_package(pack)

    def _install_r_package(self, pack):
        warnings.warn(f'Installing: {pack}...')
        self.rpackages.utils.install_packages(pack, dependencies=True, Ncpus=self.n_cpus)

    def _get_r_utils(self):
        setattr(self.rpackages, 'utils', self._import_r_package('utils'))
        self.rpackages.utils.chooseCRANmirror(ind=self.cran_mirror_idx)

    def get_imported_packages(self):
        raise NotImplementedError
