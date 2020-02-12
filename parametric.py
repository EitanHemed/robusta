import typing
import pandas as pd
from dataclasses import dataclass
import robusta as rst


@dataclass
class BaseParametric:
    """
    A basic class to handle pre-requisites of T-Tests and ANOVAs.

    Parameters
    ----------
    subject
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
        self.agg_func = agg_func
        self.data = data
        self._get_variables_and_formula(formula, dependent, between, within, subject)
        self.data = self._select_data()
        if self._perform_aggregation:
            self.data = self._aggregate_data()
        self.data.dropna(inplace=True)

    def _get_variables_and_formula(self, formula, dependent, between, within, subject):
        # Parse variables from formula or build formula from entered variables
        if formula is None:
            # Variables setting routine
            self._set_variables(dependent, between, within, subject)
            # Build model formula from entered variables
            self.formula, self._r_formula = self._get_formula_from_vars()
        elif subject is None:
            # Parse models from entered formula
            self.formula, self._r_formula = formula, rst.pyr.rpackages.stats.formula(formula)
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

    def _get_formula_from_vars(self):
        return None, None

    def _get_vars_from_formula(self):
        return None, [], [], None

    def _set_subject(self, subject):
        if subject not in self.data.columns:
            raise KeyError('Subject ID variable {} not found in data'.format(subject))
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
        raise TypeError

    def _verify_independent_vars(self):
        _ind_vars = self._between + self._within
        for i in _ind_vars:
            rst.utils.verify_levels(self.data[i].values, self.max_levels)

    def _set_dependent_var(self, dependent):
        if dependent not in self.data.columns:
            print("Dependent variable not found in data")
            raise KeyError
        return dependent

    def _select_data(self):
        data = self.data[
            self._between + self._within + [self.subject, self.dependent]].copy()
        # Make sure we are handing R a DataFrame with factor variables, as some packages do not coerce factor type.
        data[self._between + self._within] = data[self._between + self._within].astype('category')
        return data

    def _aggregate_data(self):
        return self.data.groupby(
            rst.utils.to_list(
                [self.subject, self._between, self._within])).agg(
            self.agg_func).reset_index()

    def _finalize_results(self):
        return rst.utils.tidy(self._r_results, type(self).__name__)


"""
class TTest(BaseParametric):
    
    #A general T-test class.
    

    # TODO: Add possibility to insert X and Y
    def __init__(
            self,
            tail: str = "two.sided",
            paired: bool = False,
            run_special_inits=None,
            **kwargs
    ):
        self.max_levels = 2
        self.tail = tail
        self.paired = paired
        super().__init__(**kwargs)
        self._get_independent_var()
        self._build_ttest_data()
        if run_special_inits is not None:
            self.special_inits()
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _get_formula_from_vars(self, formula):
        return self.utils.

    def _get_vars_from_formula(self):
        return None, [], [], None

    def _get_independent_var(self):
        if not self._between:
            self.independent = self._within
            self.test_type = 'Paired-Samples'
            self.paired = True
        elif not self._within:
            self.independent = self._between
            self.test_type = 'Independent-Samples'
            self.paired = False
        else:
            raise RuntimeError('Must specify either Between or Within parameter, but not both')

    def _build_ttest_data(self):
        self.x, self.y = self.data.groupby(
            getattr(self, 'independent'))[getattr(self, 'dependent')].apply(
            lambda s: s.values)

    def _run_analysis(self):
        if self.type == 'OneSample':
            return rst.pyr.rpackages.stats.t_test(x, mu=self.mu)
        return rst.pyr.rpackages.stats.t_test(
            x=self.x,
            y=self.y,
            paired=self.paired, alternative=self.tail)
"""

class Anova(BaseParametric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_type = self._infer_test_type()
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _get_formula_from_vars(self):
        return rst.utils.build_lm4_style_formula(
            dependent=self.dependent,
            between=self._between,
            within=self._within,
            subject=self.subject
        )

    def _get_vars_from_formula(self):
        return rst.utils.parse_variables_from_lm4_style_formula(self.formula)

    def _infer_test_type(self):
        # Mainly for the user, not really doing anything in terms of code
        if not self._between:
            self._between = []
            return 'Within'
        if not self._within:
            self._within = []
            return 'Between'
        return 'Mixed'

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.anova(
            rst.pyr.rpackages.afex.aov_4(
                formula=self._r_formula,
                data=rst.utils.convert_df(self.data)))
        # TODO: Add reliance on aov_ez aggregation functionality.
        # TODO: Add this functionality - sig_symbols= rst.pyr.vectors.StrVector(["", "", "", ""]))

class TTest(Anova):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_formula_from_vars(self, formula):
        return rst.utils.build_ttest_style_formula(
            dependent=self.dependent,
            between=self._between,
            within=self._within,
            subject=self.subject
        )

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
