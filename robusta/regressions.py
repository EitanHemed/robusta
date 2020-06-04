import typing
import textwrap
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas_flavor import register_dataframe_accessor
import robusta as rst

__all__ = [
    'LinearRegression', 'MixedModel'
]


@dataclass
class _BaseRegression:
    formula: typing.Union[str, None]
    formula: typing.Union[str, None]

    def __init__(self,
                 data=None,
                 formula=None,
                 ):
        self.data = data
        self.formula, self._r_formula = formula, rst.pyr.rpackages.stats.formula(
            formula)


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
            rst.pyr.rpackages.broom.tidy_lm(self._r_results))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')

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
            rst.pyr.rpackages.broom.tidy_lm(self._r_results))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')
    # Here is a working example using the mtcars dataset.
    #  m1 = rst.pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # rst.utils.convert_df(rst.pyr.rpackages.afex.nice(m1))


