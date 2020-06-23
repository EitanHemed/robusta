import abc
import typing
from dataclasses import dataclass
import abc
import pandas as pd

import robusta as rst


class AbstractClass(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._pre_process()
        self._process()

    def _pre_process(self):
        self._set_controllers()
        self._select_input_data()
        self._test_input_data()
        self._transform_input_data()

    def _process(self):
        self._analyze()
        self._tidy_results()

    def _set_controllers(self):
        pass

    @abc.abstractmethod
    def _select_input_data(self):
        pass

    @abc.abstractmethod
    def _transform_input_data(self):
        pass

    @abc.abstractmethod
    def _analyze(self):
        pass

    @abc.abstractmethod
    def _tidy_results(self):
        self._results = pd.DataFrame()

    def get_results(self):
        return self._results.apply(pd.to_numeric, errors='ignore').copy()



class X:
    """
    A basic class to handle pre-requisites of statistical analysis.
    """

    data: typing.Type[pd.DataFrame]
    subject: typing.Union[str, None]
    between = typing.Union[str, list, None]
    within = typing.Union[str, list, None]
    dependent: typing.Union[str, None]
    formula: typing.Union[str, None]
    agg_func: typing.Union[str, typing.Callable]
    _perform_aggregation: bool
    _perform_aggregation: bool
    max_levels = None

    def __init__(
            self,
            data=None,
            dependent=None,
            between=None,
            within=None,
            subject=None,
            formula=None,
            agg_func='mean',
            **kwargs
    ):

        # self.agg_func = agg_func
        # self.data = data
        # self.max_levels = kwargs.get('max_levels', None)
        self._get_variables_and_formula(formula, dependent, between, within,
                                        subject)
        self._data = self._select_data()
        if self._perform_aggregation:
            self.data = self._aggregate_data()
        self.data.dropna(inplace=True)

    def _get_variables_and_formula(self, formula, dependent, between, within,
                                   subject):
        # Parse variables from formula or build formula from entered variables
        if formula is None:
            # Variables setting routine
            self._set_variables(dependent, between, within, subject)
            # Build model formula from entered variables
            self.formula, self._r_formula = self._get_formula_from_vars()
        elif isinstance(formula, str):  # subject is None:
            # Parse models from entered formula
            self.formula, self._r_formula = formula, rst.pyr.rpackages.stats.formula(
                formula)
            dependent, between, within, subject = self._get_vars_from_formula()
            # Variables setting routine
            self._set_variables(dependent, between, within, subject)
        else:
            raise RuntimeError("")

    def _set_variables(self, dependent, between, within, subject):
        # Verify identity variable integrity
        self.subject, self._perform_aggregation = self._set_subject(subject)
        # Verify dependent variable integrity
        self.dependent = self._set_dependent_var(dependent)
        # To test whether DV can be coerced to float
        rst.utils.verify_float(self.data[self.dependent].values)
        # Verify independent variables integrity
        self._between = self._set_independent_vars(between)
        self._within = self._set_independent_vars(within)
        self.independent = self._between + self._within
        self._verify_independent_vars()

    def _get_formula_from_vars(self):
        return None, None

    def _get_vars_from_formula(self):
        return None, [], [], None

    def _set_subject(self, subject):
        if subject not in self.data.columns:
            raise KeyError(
                'Subject ID variable {} not found in data'.format(subject))
        return subject, len(self.data[subject].unique()) < len(self.data)

    def _set_independent_vars(self, ind_vars: object) -> list:
        """Make sure that the you have a list of independent variables"""
        if isinstance(ind_vars, list):
            return ind_vars
        if ind_vars is None:
            return []
        if isinstance(ind_vars, str):
            if ind_vars == '':
                return []
            return [ind_vars]
        if isinstance(self, T1Sample):
            return []
        raise TypeError

    def _verify_independent_vars(self):
        # _ind_vars = self.between + self.within
        for i in self.independent:
            rst.utils.verify_levels(self.data[i].values, self.max_levels)

    def _set_dependent_var(self, dependent):
        if dependent not in self.data.columns:
            raise KeyError("Dependent variable not found in data")
        return dependent

    def _select_data(self):
        data = self.data[
            self.independent +
            [self.subject, self.dependent]].copy()
        # Make sure we are handing R a DataFrame with factor variables,
        # as some packages do not coerce factor type.
        data[self.independent] = data[
            self.independent].astype('category')
        return data

    def _aggregate_data(self):
        return self.data.groupby(
            rst.utils.to_list(
                [self.subject] + self.independent)).agg(
            self.agg_func).reset_index()

    def _run_analysis(self):
        pass

    def _finalize_results(self):
        pass
        # return rst.utils.tidy(self._r_results, type(self).__name__)

    def get_df(self, **kwargs):
        return self.results.copy()

    def get_text_report(self):
        pass

    def get_latex_report(self, **kwargs):
        pass


class _PairwiseCorrelation:

    def __init__(self, x=None, y=None, data=None,
                 nan_action='raise',
                 **kwargs):
        self.data = data
        self.x = x
        self.y = y
        self._data = self._test_input()
        self.nan_action = nan_action
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _test_input(self):
        if self.data is None:
            if sum([isinstance(i, str) for i in [self.x, self.y]]) == 2:
                raise ValueError('Specify dataframe and enter `x` and `y`'
                                 ' as strings.')
            try:
                _data = pd.DataFrame(columns=['x', 'y'],
                                     data=np.array([self.x, self.y]).T)
            except ValueError:
                raise ValueError('`x` and ``y` are not of the same length')

            if _data.isnull().values.any():
                if self.nan_action == 'raise':
                    raise ValueError('NaN in data, either specify action or '
                                     'remove')
                if self.nan_action == 'drop':
                    _data.dropna(inplace=True)
                if self.nan_action == 'replace':
                    if self.nan_action in ('mean', 'median', 'mode'):
                        raise NotImplementedError
            return _data

        if sum([isinstance(i, str) for i in [self.x, self.y]]) == 2:
            try:
                _data = self.data[[self.x, self.y]].copy()
            except KeyError:
                raise KeyError(f"Either `x` or ({self.x}),`y` ({self.y})"
                               f" are not columns in data")
        else:
            raise ValueError('Either enter `data` as a pd.DataFrame'
                             'and `x` and `y` as two column names, or enter'
                             '`x` and `y` as np.arrays')
        return _data
