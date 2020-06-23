import warnings
import pandas as pd
import numpy as np
import robusta as rst

__all__ = ['ChiSquare', 'Correlation', 'PartCorrelation',
           'PartialCorrelation', 'BayesCorrelation']


class _PairwiseCorrelation(rst.base.AbstractClass):

    def __init__(self, x=None, y=None, data=None,
                 nan_action='raise',
                 **kwargs):
        self.data = data
        self.x = x
        self.y = y
        self.nan_action = nan_action
        super().__init__()

    def _select_input_data(self):
        _data = None

        if self.data is None:
            if sum([isinstance(i, str) for i in [self.x, self.y]]) == 2:
                raise ValueError('Specify dataframe and enter `x` and `y`'
                                 ' as strings.')
            try:
                _data = pd.DataFrame(columns=['x', 'y'],
                                     data=np.array([self.x, self.y]).T)
            except ValueError:
                raise ValueError('`x` and ``y` are not of the same length')

        elif isinstance(self.data, pd.DataFrame):
            if sum([isinstance(i, str) for i in [self.x, self.y]]) == 2:
                try:
                    _data = self.data[[self.x, self.y]].copy()
                except KeyError:
                    raise KeyError(f"Either `x` or ({self.x}),`y` ({self.y})"
                                   f" are not columns in data")

        if _data is None:  # Failed to parse data from input
            raise ValueError('Either enter `data` as a pd.DataFrame'
                             'and `x` and `y` as two column names, or enter'
                             '`x` and `y` as np.arrays')

        self._input_data = _data

    def _test_input_data(self):
        return

        if self._input_data.isnull().values.any():
            if self.nan_action == 'raise':
                raise ValueError('NaN in data, either specify action or '
                                 'remove')
            if self.nan_action == 'drop':
                self._input_data.dropna(inplace=True)
            if self.nan_action == 'replace':
                if self.nan_action in ('mean', 'median', 'mode'):
                    raise NotImplementedError

    def _transform_input_data(self):
        pass


class ChiSquare(_PairwiseCorrelation):
    """
    Run a frequentist Chi-Square \u03C7\u00B2 test of independence.

    .. _Implemented R function stats::t.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/chisq.test


    Parameters
    ----------
    x, y : [str, array-like]
        Input to test. Either strings with the names of columns on the
        `data` dataframe or two array-like objects. Each array-like object
        must contain only two unique values.
    data : pd.DataFrame, optional
        Dataframe of data to test, with the columns specieid in `x` and `y`.
        Each of the relevant columns must contain only two unique values.

    Returns
    -------


    """

    def __init__(self, apply_correction=True,
                 **kwargs):
        self.apply_correction = apply_correction
        super().__init__(**kwargs)

    def _transform_input_data(self):
        self._crosstab_data()
        self._test_crosstablated_data()

    def _test_crosstablated_data(self):
        if self.crosstabulated_data.min().min() < 5:
            warnings.warn(
                'Less than 5 observations in some cell(s). Assumptions'
                'may be violated.')

    def get_text(self, alpha=.05):
        _dat = self.get_results().to_dict('records')[0]
        significance_result = np.where(_dat['p.value'] < .05, 'a', 'no')
        main_clause = (f"A Chi-Square test of independence shown"
                       f" {significance_result} significant association between"
                       f" {self._input_data.columns[0]} and {self._input_data.columns[1]}"
                       )
        result_clause = (f"[\u03C7\u00B2({_dat['parameter']:.0f})" +
                         f" = {_dat['statistic']:.2f}, p = {_dat['p.value']:.3f}]")

        return f'{main_clause} {result_clause}.'

    def _crosstab_data(self):
        """Returns the crosstabulation of `x` and `y` categories."""
        self.crosstabulated_data = pd.crosstab(*self._input_data.values.T)

    def _analyze(self):
        """Runs a Chi-Square test"""
        self._r_results = rst.pyr.rpackages.stats.chisq_test(
            self.crosstabulated_data,
            correct=self.apply_correction)

    def _tidy_results(self):
        """Tidy the test results and return as pd.DataFrame"""
        self._results = rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(self._r_results))


class Correlation(_PairwiseCorrelation):
    """Calculate a correlation coefficient (test?).

    Parameters
    ----------
    method : str
        Type of correlation coefficient. Possible values are
        'pearson', 'spearman' or 'kendall'. Default (None) raises ValueError.

    Returns
    -------

    """

    def __init__(self, method='pearson', **kwargs):
        self.method = method
        super().__init__(**kwargs)

    def _test_input_data(self):
        if self.method not in ('pearson', 'spearman', 'kendall'):
            raise ValueError('Invalid correlation coefficient method - specify'
                             ' either `pearson`, `spearman` or `kendall`')

        super()._test_input_data()

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.stats.cor_test(
            *self._input_data.values.T,
            method=self.method
        )


class _TriplewiseCorrelation(Correlation):

    def __init__(self, z=None, **kwargs):
        self.z = z
        super().__init__(**kwargs)

    def _select_input_data(self):
        _data = None

        if self.data is None:
            if sum([isinstance(i, str) for i in [self.x, self.y, self.z]]) == 3:
                raise ValueError('Specify dataframe and enter `x`, `y` and `z`'
                                 ' as strings.')
            try:
                _data = pd.DataFrame(columns=['x', 'y', 'z'],
                                     data=np.array([self.x, self.y, self.z]).T)
            except ValueError:
                raise ValueError('`x`, `y` and `z` are not of the same length')

        elif sum([isinstance(i, str) for i in [self.x, self.y, self.z]]) == 3:
            try:
                _data = self.data[[self.x, self.y, self.z]].copy()
            except KeyError:
                raise KeyError(f"Either `x` ({self.x}),`y` ({self.y}) or `z`"
                               f" {self.z} are not columns in data")

        if _data is None:
            raise ValueError('Either enter `data` as a pd.DataFrame'
                             'and `x`, `y` and `z` as two column names, or enter'
                             '`x`, `y` and `z` as np.arrays')

        self._input_data = _data

    # def _aggregate_data(self):
    #     if _data.isnull().values.any():
    #         if self.nan_action == 'raise':
    #             raise ValueError('NaN in data, either specify action or '
    #                              'remove')
    #         if self.nan_action == 'drop':
    #             _data.dropna(inplace=True)
    #         if self.nan_action == 'replace':
    #             if self.nan_action in ('mean', 'median', 'mode'):
    #                 raise NotImplementedError
    #     return _data

    def _tidy_results(self):
        self._results = rst.utils.convert_df(self._r_results)



class PartialCorrelation(_TriplewiseCorrelation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.ppcor.pcor_test(
            *self._input_data.values.T,
            method=self.method)


class PartCorrelation(_TriplewiseCorrelation):
    """Semi-Partial Correlation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.ppcor.spcor_test(
            *self._input_data.values.T,
            method=self.method)


# TODO - see what other
class BayesCorrelation(_PairwiseCorrelation):

    def __init__(self,
                 rscale_prior=None,
                 null_interval=None,
                 posterior=None, **kwargs):

        if rscale_prior is None:
            self.rscale_prior = 'medium'
        if null_interval is None:
            self.null_interval = [-1, 1]
        if posterior is None:
            self.posterior = False

        super().__init__(**kwargs)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.bayesfactor.correlationBF(
            *self._input_data.values.T,
            nullInterval=self.null_interval,
            rscale_prior=self.rscale_prior,
            posterior=self.posterior,
        )

    def _tidy_results(self):
        self._results = rst.utils.convert_df(
            rst.pyr.rpackages.base.data_frame(self._r_results,
                                              'model')).drop(columns=['time',
                                                                      'code'])
