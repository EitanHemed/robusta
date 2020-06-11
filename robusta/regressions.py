import typing
import textwrap
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas_flavor import register_dataframe_accessor
import robusta as rst

__all__ = [
    'LinearRegression', 'BayesianLinearRegression',
    'LogisticRegression',
    'MixedModel'
]


@dataclass
class _BaseRegression:
    formula: typing.Union[str, None]
    formula: typing.Union[str, None]

    def __init__(self,
                 data=None,
                 formula=None,
                 subject=None
                 ):
        self.data = data
        self.formula, self._r_formula = formula, rst.pyr.rpackages.stats.formula(
            formula)

    def _test_subject_kwarg(self):
        pass
        # TODO - if linear regression check whether we should aggregate on
        #  subject.
        # TODO - if mixed model check whether we have multiple observations of
        #  each subject within each level.

    def _validate_binary_variables(self, variable_name):
        if self.data[variable_name].unique().shape[0] != 2:
            raise RuntimeError(f"Column {variable_name} should contain only"
                               f"the values 0 and 1")


class LinearRegression(_BaseRegression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.lm(
            **{
                'formula': self._r_formula,
                'data': self.data,
                # 'weights': np.nan,
                # 'singular.ok': np.nan,
                # 'offset': np.nan
            }
        )

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(self._r_results))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')


class BayesianLinearRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_analysis(self):
        return rst.pyr.rpackages.base.data_frame(
            rst.pyr.rpackages.bayesfactor.generalTestBF(
                formula=self._r_formula, data=self.data, progress=False))

    def _finalize_results(self):
        return rst.utils.convert_df(self._r_results,
                                    'model').drop(columns=['time',
                                                           'code'])


class LogisticRegression(_BaseRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO - this is likely to break on spaces before the tilda sign.
        self.dependent = self.formula.split('~')[0]
        self._validate_binary_variables(self.dependent)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.glm(
            formula=self.formula, data=self.data,
            family='binomial'
        )

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.base.data_frame(
                rst.pyr.rpackages.generics.tidy(
                    self._r_results)))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')

class BayesianLogisticRegression(LogisticRegression):
    pass


class MixedModel:

    def __init__(self,
                 levels,
                 **kwargs):
        self.levels = levels
        self.formula, self._r_formula = self.get_formula()

        super().__init__(**kwargs)

    def _run_analysis(self):
        return rst.pyr.rpackages.lme4.lmer(
            **{
                'formula': self._r_formula,
                'data': self.data,
                # 'weights': np.nan,
                # 'singular.ok': np.nan,
                # 'offset': np.nan
            }
        )

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(self._r_results))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')
    # Here is a working example using the mtcars dataset.
    #  m1 = rst.pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # rst.utils.convert_df(rst.pyr.rpackages.afex.nice(m1))
