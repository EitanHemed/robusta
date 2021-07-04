import pandas as pd

from .. import pyr
from ..misc import utils, base

BF_COLUMNS = ['model', 'bf', 'error']
REDUNDANT_BF_COLUMNS = ['time', 'code']


class RegressionResults(base.BaseResults):

    # def _tidy_results(self):
    #     self._results = utils.convert_df(
    #         pyr.rpackages.base.data_frame(
    #             pyr.rpackages.generics.tidy(
    #                 self.r_results)))
    #
    # def get_results(self):
    #     return self._results.apply(
    #         pd.to_numeric, errors='ignore')

    def predict(self, new_data: pd.DataFrame, type: str = 'response'):
        """
        @param new_data: pd.DataFrame
            A dataframe containing the same variables as used on the model.
        @param type: str
            Options are 'response' and 'type'. Default is 'response'.
        @return:
        """
        return utils.convert_df(pyr.rpackages.stats.predict(
            self.r_results, new_data, type=self.default_predict_type))


class LinearRegressionResults(RegressionResults):
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
    pass


class BayesianLinearRegression(LinearRegressionResults):
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

    def __init__(self, r_results, mode='bf'):
        self.mode = mode

        super().__init__(r_results)

    # def _tidy_results(self):
    #     if self.mode == 'bf':
    #         return self.r_results[BF_COLUMNS] # return utils.convert_df(self.r_results, 'model')[BF_COLUMNS]
    #     else:
    #         return self.r_results[BF_COLUMNS] # utils.convert_df(self.r_results)

    def _tidy_results(self):
        return self.r_results.drop(columns=REDUNDANT_BF_COLUMNS)

    # def get_report(self, mode: str = 'df'):
    #
    #     if mode == 'df':
    #         return pyr.rpackages.report.as_data_frame_report(
    #             self.r_results)
    #     if mode == 'verbose':
    #         return pyr.rpackages.report.report(self.r_results)


class LogisticRegressionResults(RegressionResults):

    def _test_input_data(self):
        super()._test_input_data()
        self._validate_binary_variables(self.dependent)

    def _analyze(self):
        self.r_results = pyr.rpackages.stats.glm(
            formula=self._r_formula, data=self.data,
            family='binomial'
        )


class BayesianLogisticRegression(LogisticRegressionResults):

    def __init__(self):
        raise NotImplementedError

    def _analyze(self):
        return pyr.rpackages.brms.brm(
            formula=self.formula,
            data=self._data,
            family=pyr.rpackages.brms.bernoulli(link='logit')
        )


class MixedModelResults:

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
            pyr.rpackages.generics.tidy(self.r_results))

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')
    # Here is a working example using the mtcars dataset.
    #  m1 = pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # utils.convert_df(pyr.rpackages.afex.nice(m1))
