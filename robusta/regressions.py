import typing
import textwrap
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas_flavor import register_dataframe_accessor
import robusta as rst
import warnings

warnings.warn("Currently the regressions module is under development,"
              "nothing is promised to work correctly.")

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
        self.formula = formula
        super().__init__()

    def _set_controllers(self):
        # First we parse the variables from the formula
        self._set_variables_from_formula()
        # So we can make sure that the formula is compatible
        self._set_formula()

    def _set_variables_from_formula(self):

        vp = rst.formula_tools.VariablesParser(self.formula)
        self.dependent, self.between, self.within, self.subject = vp.get_variables()
        self._vars = rst.utils.to_list(vp.get_variables())

    def _set_formula(self):
        # Beautify the formula
        fp = rst.formula_tools.FormulaParser(
            self.dependent, self.between, self.within, self.subject
        )
        frml = fp.get_formula()
        frml = frml.replace(f'+(1|{self.subject})', '')
        self.formula = frml
        self._r_formula = rst.pyr.rpackages.stats.formula(frml)

    def _select_input_data(self):
        _data = None

        if not isinstance(self.data, pd.DataFrame):
            raise RuntimeError('No data!')
        _data = self.data[self._vars].copy()
        self._input_data = _data

    def _test_input_data(self):
        for v in rst.utils.to_list(self._vars):
            if v not in self._input_data.columns:
                raise KeyError(f'Variable {v} from formula not found in data!')

        if self._input_data.isnull().values.any():
            if self.nan_action == 'raise':
                raise ValueError(
                    'NaN in data, either specify action or '
                    'remove')
            if self.nan_action == 'drop':
                self._input_data.dropna(inplace=True)
            if self.nan_action == 'replace':
                if self.nan_action in ('mean', 'median', 'mode'):
                    raise NotImplementedError

    def _transform_input_data(self):
        pass

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
        self._results = rst.utils.convert_df(
            rst.pyr.rpackages.base.data_frame(
                rst.pyr.rpackages.generics.tidy(
                    self._r_results)))

    def get_results(self):
        return self._results.apply(
            pd.to_numeric, errors='ignore')


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


class BayesianLinearRegression(LinearRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    #def _set_formula(self):
        #super()._set_formula()
        #frml = self.formula + f"+{self.subject}"
        #self.formula = frml
        #self._r_formula = rst.pyr.rpackages.stats.formula(frml)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.base.data_frame(
            rst.pyr.rpackages.bayesfactor.generalTestBF(
                formula=self._r_formula, data=self._input_data, progress=False,
                whichRandom=self.subject, neverExclude=self.subject))

    def _tidy_results(self):
        self._results = rst.utils.convert_df(self._r_results,
                                             'model').drop(columns=['time',
                                                                    'code'])


class LogisticRegression(_BaseRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO - this is likely to break on spaces before the tilda sign.
        # self.dependent = self.formula.split('~')[0]

    def _test_input_data(self):
        super()._test_input_data()
        self._validate_binary_variables(self.dependent)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.stats.glm(
            formula=self.formula, data=self.data,
            family='binomial'
        )


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
