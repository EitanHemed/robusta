# TODO: Selection of R mirror for session.
# TODO: How to install R packages without operating as administrator.

"""
pyrio is the module that contains the link to the R backend of robusta.
pyrio stand for Python-R Interface Object.
pyrio is in charge of the following:
    - Starting an R session.
    - Transferring objects into the R environment and back.
"""

import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri, numpy2ri
import rpy2.robjects.packages as rpackages
numpy2ri.activate()

class PyRIO:
    instances_count = 0  # This is a static counter

    # PYTON-R-Interface-Object

    def __init__(self,
                 packs=[]): #'broom', 'afex', 'car', 'emmeans', 'base', 'BayesFactor', 'stats', 'effsize', 'psych'

        if PyRIO.instances_count == 1:  # Check if the number of instances present are more than one.
            del self
            print
            "A PyRIO object has already been initialized. Please use existing"
            return
        PyRIO.instances_count += 1

        self.packs = packs
        self.cur_r_objects = self.__get_r_objects__()

    def __get_r_objects__(self):
        self.__verify_if_installled__()
        return dict(zip(self.packs, map(
            lambda x: rpackages.importr(x), self.packs)))

    def __df__(self, df, to):
        # this conversion works on pandas 0.23.4 - but doesn't work on 0.25
        if to == 'r':
            return pandas2ri.py2ri(df)
        if to == 'py':
            return pd.DataFrame(pandas2ri.ri2py(df))

    def __verify_if_installled__(self):
        [self.__install_if_required__(pack) if not rpackages.isinstalled(pack)
         else None for pack in self.packs]

    def __install_if_required__(self, pack):
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(pack) #robjects.vectors.StrVector(pack))

