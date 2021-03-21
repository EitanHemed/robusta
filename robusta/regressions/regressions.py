import re
import typing
import warnings
from dataclasses import dataclass

import pandas as pd

from .. import base, formula_tools, utils, pyr

warnings.warn("Currently the regressions module is under development,"
              "nothing is promised to work correctly.")

__all__ = [
    'LinearRegression', 'BayesianLinearRegression',
    'LogisticRegression',
    'MixedModel'
]


@dataclass
class _BaseRegression(base.AbstractClass):
    formula: typing.Union[str, None]

    """
    Parameters
    ----------
    formula : str
        An r-like formula specifying the regression model. Must contain dependent,
        at least one independent variable and subject term
    data : pd.DataFrame
        A pd.DataFrame containing the model variables. Either one row per subject
        or more.
    """

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

        vp = formula_tools.VariablesParser(self.formula)
        self.dependent, self.between, self.within, self.subject = vp.get_variables()
        self._vars = utils.to_list(vp.get_variables())

    def _set_formula(self):
        """
        fp = formula_tools.FormulaParser(
             self.dependent, self.between, self.within, self.subject)
        frml = fp.get_formula()
        print(frml)
        frml = frml.replace(f'+(1|{self.subject})', '')
        #frml = self.formula.replace(f'+(1|{self.subject})', '')
        print(frml)
        self.formula = frml
        # Else, just use the entered formula
        """
        pattern = re.compile(
            r'\s*\+{1,1}\(*\s*1{1,1}\s*\|{1,1}\s*' + self.subject + r'\s*\)*')
        frml = re.sub(pattern, '', self.formula)
        self._r_formula = pyr.rpackages.stats.formula(frml)

    def _select_input_data(self):
        _data = None

        if not isinstance(self.data, pd.DataFrame):
            raise RuntimeError('No data!')
        _data = self.data[self._vars].copy()
        self._input_data = _data

    def _test_input_data(self):
        for v in utils.to_list(self._vars):
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
        self._results = utils.convert_df(
            pyr.rpackages.base.data_frame(
                pyr.rpackages.generics.tidy(
                    self._r_results)))

    def get_results(self):
        return self._results.apply(
            pd.to_numeric, errors='ignore')

    def predict(self, new_data: pd.DataFrame, type: str = 'response'):
        """
        @param new_data: pd.DataFrame
            A dataframe containing the same variables as used on the model.
        @param type: str
            Options are 'response' and 'type'. Default is 'response'.
        @return:
        """
        return convert_df(pyr.rpackages.stats.predict(
            self._r_results, new_data, type=self.default_predict_type))


class LinearRegression(_BaseRegression):
    """

    Parameters
    ----------
    # TODO - implement these:
    weights
    method
    model
    singular_ok
    contrasts
    offset

    Returns
    -------



    Inplemented R function: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
    """

    def _analyze(self):
        self._r_results = pyr.rpackages.stats.lm(
            **{
                'formula': self._r_formula,
                'data': self.data,
                # 'weights': np.nan,
                # 'singular.ok': np.nan,
                # 'offset': np.nan
            }
        )

    # This can potentially be a static rather than a class method,
    # as it may be relevant to a logistic regression as well
    def predict(self,
                new_data: pd.DataFrame,
                interval_type: typing.Union[str, None]):

        if interval_type is None:
            interval_type = 'none'
        else:
            if interval_type not in ['prediction', 'confidence']:
                raise ValueError("interval type must be one of None, "
                                 "'confidence' or 'prediction'")

        convert_df(pyr.rpackages.stats.predict(
            self._r_results,
            new_data,
            interval_type=interval_type
        ))


class BayesianLinearRegression(LinearRegression):
    """

        Parameters
        ----------
        # TODO - implement these:
        whichRandom
        whichModels
        neverExclude
        rscaleFixed
        multicore
        method
        noSample
        Returns
        -------

        Inplemented R function: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
        """

    def __init__(self,
                 iterations: int = 10000,
                 exclude_subject: bool = True,
                 # TODO add type hints to the next argument
                 never_exclude=pyr.rinterface.NULL,
                 **kwargs):
        self.iterations = iterations
        self.exclude_subject = exclude_subject
        self.never_exclude = never_exclude
        super().__init__(**kwargs)

    def _analyze(self):
        # raise NotImplementedError("Currently the formula tools just assumes all interactions rather than "
        #                          "going by the supplied formula. This has to be solved ASAP.")
        self._r_results = pyr.rpackages.base.data_frame(
            pyr.rpackages.bayesfactor.generalTestBF(
                formula=self._r_formula, data=self._input_data, progress=False,
                whichRandom=pyr.rinterface.NULL if self.exclude_subject else self.subject,
                neverExclude=pyr.rinterface.NULL,
                iterations=self.iterations))

    def _tidy_results(self):
        self._results = utils.convert_df(self._r_results,
                                             'model').drop(columns=['time',
                                                                    'code'])

    def get_report(self, mode: str = 'df'):

        if mode == 'df':
            return pyr.rpackages.report.as_data_frame_report(
                self._r_results)
        if mode == 'verbose':
            return pyr.rpackages.report.report(self._r_results)


class LogisticRegression(_BaseRegression):

    def _test_input_data(self):
        super()._test_input_data()
        self._validate_binary_variables(self.dependent)

    def _analyze(self):
        self._r_results = pyr.rpackages.stats.glm(
            formula=self._r_formula, data=self.data,
            family='binomial'
        )


class BayesianLogisticRegression(LogisticRegression):

    def __init__(self):
        raise NotImplementedError

    def _analyze(self):
        return pyr.rpackages.brms.brm(
            formula=self.formula,
            data=self._data,
            family=pyr.rpackages.brms.bernoulli(link='logit')
        )


class MixedModel:

    def __init__(self,
                 levels,
                 **kwargs):
        self.levels = levels
        self.formula, self._r_formula = self.get_formula()
        super().__init__(**kwargs)

    def _set_variables(self):
        self._vars = utils.parse_variables_from_lm4_style_formula(
            self.formula)

    def _run_analysis(self):
        return pyr.rpackages.lme4.lmer(
            **{
                'formula': self._r_formula,
                'data': self.data,
                # 'weights': np.nan,
                # 'singular.ok': np.nan,
                # 'offset': np.nan
            }
        )

    def _finalize_results(self):
        return utils.convert_df(
            pyr.rpackages.generics.tidy(self._r_results))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')
    # Here is a working example using the mtcars dataset.
    #  m1 = pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # utils.convert_df(pyr.rpackages.afex.nice(m1))
