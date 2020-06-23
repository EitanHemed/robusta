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
class _BaseRegression(rst.base.AbstractClass):
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
        self._prepare()
        super().__init__()
        # self._r_results = self._run_analysis()
        # self.results = self._finalize_results()

    def _prepare(self):
        self._test_subject_kwarg()
        self._set_variables()
        self._test_input()

    def _set_variables(self):
        # self._vars = rst.utils.parse_variables_from_general_formula(
        #    self.formula)
        vp = rst.formula_tools.VariablesParser(self.formula)
        self._vars = vp.get_variables()
        print(self._vars)

    def _test_input(self):

        if not isinstance(self.data, pd.DataFrame):
            raise RuntimeError('No data!')

        cols = self.data.columns
        for v in self._vars:
            if v not in cols:
                raise KeyError(f'Variable {v} from formula not found in data!')

        _data = self.data[self._vars].copy()

        if _data.isnull().values.any():
            if self.nan_action == 'raise':
                raise ValueError(
                    'NaN in data, either specify action or '
                    'remove')
            if self.nan_action == 'drop':
                _data.dropna(inplace=True)
            if self.nan_action == 'replace':
                if self.nan_action in ('mean', 'median', 'mode'):
                    raise NotImplementedError

        return _data

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

    def _analyze(self):
        pass

    def _tidy_results(self):
        pass

    def get_results(self):
        pass


class LinearRegression(_BaseRegression):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.stats.lm(
            **{
                'formula': self._r_formula,
                'data': self.data,
                # 'weights': np.nan,
                # 'singular.ok': np.nan,
                # 'offset': np.nan
            }
        )

    def _tidy_results(self):
        self._results = rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(self._r_results))

    def get_results(self):
        return self._results.apply(
            pd.to_numeric, errors='ignore')


class BayesianLinearRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.base.data_frame(
            rst.pyr.rpackages.bayesfactor.generalTestBF(
                formula=self._r_formula, data=self.data, progress=False))

    def _tidy_results(self):
        self._results = rst.utils.convert_df(self._r_results,
                                             'model').drop(columns=['time',
                                                                    'code'])


class LogisticRegression(_BaseRegression):
    def __init__(self, **kwargs):
        f = rst.formula_tools.VariablesParser(kwargs['formula'])
        print(f.get_variables())

        super().__init__(**kwargs)
        # TODO - this is likely to break on spaces before the tilda sign.
        self.dependent = self.formula.split('~')[0]
        self._validate_binary_variables(self.dependent)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.stats.glm(
            formula=self.formula, data=self.data,
            family='binomial'
        )

    def _tidy_results(self):
        self._results = rst.utils.convert_df(
            rst.pyr.rpackages.base.data_frame(
                rst.pyr.rpackages.generics.tidy(
                    self._r_results)))

    def get_results(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')


class BayesianLogisticRegression(LogisticRegression):

    def __init__(self):
        raise NotImplementedError

    def _analyze(self):
        return rst.pyr.rpackages.brms.brm(
            formula=self.formula,
            data=self._data,
            family=rst.pyr.rpackages.brms.bernoulli(link='logit')
        )


class MixedModel:

    def __init__(self,
                 levels,
                 **kwargs):
        self.levels = levels
        self.formula, self._r_formula = self.get_formula()
        super().__init__(**kwargs)

    def _set_variables(self):
        self._vars = rst.utils.parse_variables_from_lm4_style_formula(
            self.formula)
        print(self._vars)

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
