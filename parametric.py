import robusta  # So we can get the PyR singleton
import numpy as np
from dataclasses import dataclass
from typing import Type
import pandas as pd
from pandas import DataFrame
from robusta import utils
import itertools


@dataclass
class BaseParametric:
    """
    A basic class to handle pre-requisites of most common parametric tests.
    """
    _perform_aggregation: bool
    data: Type[DataFrame]
    ind_vars_between = str
    ind_vars_within = str
    dep_var: str
    id_var: str
    agg_func: str

    def __init__(
            self,
            id_var='',
            dep_var=None,
            ind_vars_between='',
            ind_vars_within='',
            data=None,
            agg_func='mean'):
        self.agg_func = agg_func
        self.max_levels = None
        self.data = data
        # Verify identity variable integrity
        self.id_var, self._perform_aggregation = self._set_id_var(id_var)
        # Verify dependent variable integrity
        self.dep_var = self._set_dep_var(dep_var)
        # Verify independent variables integrity
        self.ind_vars_between = self._set_ind_vars(ind_vars_between)
        self.ind_vars_within = self._set_ind_vars(ind_vars_within)
        self._run_pre_reqs()
        if self._perform_aggregation:
            self.data = self._aggregate_data()
        # self.results = self.beautify(self._r_results)

    def _run_pre_reqs(self) -> None:
        """
        @rtype: None
        """
        utils.verify_float(self.data[self.dep_var].values)  # To test whether DV can be coerced to float

    def _set_id_var(self, id_var):
        if id_var not in self.data.columns:
            raise KeyError
            print("Dependent variable not found in data")

        return id_var, len(self.data[id_var].unique()) < len(self.data)

    def _set_ind_vars(self, ind_vars: object) -> list:
        """Make sure that the you have a list of independent variables"""
        if type(ind_vars) is list:
            return ind_vars
        elif ind_vars is None:
            print('yalla')
            return []
        elif type(ind_vars) is str:
            if ind_vars == '':
                return []
            else:
                return [ind_vars]
        else:
            raise TypeError

    def _verify_ind_vars(self):
        _ind_vars = self.ind_vars_between + self.ind_vars_within
        for i in _ind_vars:
            utils.verify_levels(self.data[i].values, self.max_levels)

    def _verify_vars_in_data(self):
        _vars = self.ind_vars_between + self.ind_vars_within + [self.id, self.dep_var]

    def _set_dep_var(self, dep_var):
        if dep_var not in self.data.columns:
            print("Dependent variable not found in data")
            raise KeyError
        else:
            return dep_var

    def _aggregate_data(self):
        return self.data.groupby(
            [i for i in list(itertools.chain.from_iterable(
                itertools.repeat(x, 1) if isinstance(x, str) else x for x in
                [self.id_var, self.ind_vars_between, self.ind_vars_within]))
             if i != '']).agg(self.agg_func).reset_index()


@dataclass
class TTest(BaseParametric):
    """
    A general T-test class.
    """
    x: list = None
    y: list = None
    tail: str = "both"
    paired: bool = None
    max_levels: int = 2

    def __init__(self,
                 x=None,
                 y=None,
                 data=None,
                 dep_var=None,
                 ind_var=None,
                 ):

        # if self.paired is False:

        super().__init__(
            data=None,
            dep_var=None,
            ind_vars_between=None,
            ind_vars_within=None,
            paired=False
        )

    def _infer_input(self):
        if self.data is None:
            pass
        else:
            if type(self.data) == pd.core.frame.DataFrame:
                self.data.groupby(
                    [self.id] + max([self.ind_vars_between, self.ind_vars_within])).agg(self.agg_func).reset_index()

    def _run_analysis(self):
        self._results = pyr.cur_r_objects['stats'].t_test(
            x=self.x, y=self.y, paired=self.paired, alternative=self.tail)


@dataclass
class Anova(BaseParametric):
    """
    """
    anova_type: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anova_type = self._infer_anova_type()
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _infer_anova_type(self):
        if self.ind_vars_between == []:
            return 'Within'
        if self.ind_vars_within == []:
            return 'Between'
        else:
            return 'Mixed'

    def _run_analysis(self):
        if self.anova_type == 'Between':
            return robusta.pyr.rpackages.afex.aov_ez(self.id_var, self.dep_var, utils.convert_df(self.data),
                                                     between=self.ind_vars_between)
        if self.anova_type == 'Within':
            return robusta.pyr.rpackages.afex.aov_ez(self.id_var, self.dep_var, utils.convert_df(self.data),
                                                     within=self.ind_vars_within)
        else:
            return robusta.pyr.rpackages.afex.aov_ez(self.id_var, self.dep_var, utils.convert_df(self.data),
                                                     within=self.ind_vars_within,
                                                     between=self.ind_vars_between)

    def _finalize_results(self):
        return utils.tidy(self._r_results, 'Anova')
