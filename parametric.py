import robusta as rst
from dataclasses import dataclass
import typing
import numpy as np
import pandas as pd
from pandas import DataFrame
import itertools


@dataclass
class BaseParametric:
    """
    A basic class to handle pre-requisites of T-Tests and ANOVAs.

    Parameters
    ----------
    subject
    """
    subject: str
    dependent: str
    between = str
    within = str
    data: typing.Type[DataFrame]
    agg_func: str
    _perform_aggregation: bool
    _perform_aggregation: bool
    max_levels = None

    def __init__(
            self,
            subject='',
            dependent=None,
            between='',
            within='',
            data=None,
            formula='',
            agg_func='mean',
            **kwargs
    ):
        self.agg_func = agg_func
        self.data = data
        # Verify identity variable integrity
        self.subject, self._perform_aggregation = self._set_subject(subject)
        # Verify dependent variable integrity
        self.dependent = self._set_dependent_var(dependent)
        rst.utils.verify_float(self.data[self.dependent].values)  # To test whether DV can be coerced to float
        # Verify independent variables integrity
        self.between = self._set_independent_vars(between)
        self.within = self._set_independent_vars(within)
        self.data = self._select_data()
        if self._perform_aggregation:
            self.data = self._aggregate_data()
        self.data.dropna(inplace=True)

    def _select_data(self):
        return self.data[self.between + self.within + [self.subject, self.dependent]].copy()

    def _set_subject(self, subject):
        if subject not in self.data.columns:
            raise KeyError('Subject ID variable {} not found in data'.format(subject))
        return subject, len(self.data[subject].unique()) < len(self.data)

    def _set_independent_vars(self, ind_vars: object) -> list:
        """Make sure that the you have a list of independent variables"""
        if type(ind_vars) is list:
            return ind_vars
        elif ind_vars is None:
            return []
        elif type(ind_vars) is str:
            if ind_vars == '':
                return []
            else:
                return [ind_vars]
        else:
            raise TypeError

    def _verify_independent_vars(self):
        _ind_vars = self.between + self.within
        for i in _ind_vars:
            rst.utils.verify_levels(self.data[i].values, self.max_levels)

    def _set_dependent_var(self, dependent):
        if dependent not in self.data.columns:
            print("Dependent variable not found in data")
            raise KeyError
        else:
            return dependent

    def _aggregate_data(self):
        return self.data.groupby(
            rst.utils.to_list([self.subject, self.between, self.within])).agg(self.agg_func).reset_index()

    def _finalize_results(self):
        return rst.utils.tidy(self._r_results, type(self).__name__)


class Anova(BaseParametric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anova_type = self._infer_anova_type()
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _infer_anova_type(self):
        if not self.between:
            self.between = rst.pyr.rinterface.NULL
            return 'Within'
        if not self.within:
            self.within = rst.pyr.rinterface.NULL
            return 'Between'
        else:
            return 'Mixed'

    def _run_analysis(self):
        try:
            return rst.pyr.rpackages.afex.aov_ez(self.subject, self.dependent, rst.utils.convert_df(self.data),
                                                 between=self.between, within=self.within)
        # TODO: Add reliance on aov_ez aggregation functionality.
        # TODO: Add this functionality - sig_symbols= rst.pyr.vectors.StrVector(["", "", "", ""]))
        except rst.pyr.rinterface.RRuntimeError as e:
            if "Following ids are in more than one between subjects condition":
                raise RuntimeError('For Between-Subject or Mixed designs, each subject ID must be unique')
            else:
                raise e

class TTest(BaseParametric):
    """
    A general T-test class.
    """

    # x: np.ndarray = np.empty(0)
    # y: np.ndarray = np.empty(0)
    # TODO: Add possibility to insert X and Y
    def __init__(
            self,
            tail: str = "two.sided",
            paired: bool = False,
            special_inits=None,
            **kwargs
    ):
        self.max_levels = 2
        self.tail = tail
        self.paired = paired
        super().__init__(**kwargs)
        self._get_independent_var()
        self._build_ttest_data()
        if special_inits is not None:
            special_inits()
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _get_independent_var(self):
        if not self.between:
            self.independent = self.within
            self.type = 'Paired-Samples'
            self.paired = True
        elif not self.within:
            self.independent = self.between
            self.type = 'Independent-Samples'
            self.paired = False
        else:
            raise RuntimeError('Must specify either Between or Within parameter, but not both')

    def _build_ttest_data(self):
        self.x, self.y = self.data.groupby(self.independent)[self.dependent].apply(lambda s: s.values)

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.t_test(
            x=self.x,
            y=self.y,
            paired=self.paired, alternative=self.tail)


class PairwiseComparison:
    def __init__(self, data, correction_method):
        self.data = data
        self.correction_method = correction_method

class MixedModel:
    pass
    # TODO: This! ^
    # Here is a working example using the mtcars dataset.
    #  m1 = rst.pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # rst.utils.convert_df(rst.pyr.rpackages.afex.nice(m1))