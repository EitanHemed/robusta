import pandas as pd
import numpy as np
import robusta as rst

__all__ = ['ChiSquare', 'Correlation', 'BayesCorrelation']


class _BaseCorrelation:

    def __init__(self, x=None, y=None, data=None,
                 nan_action='raise',
                 **kwargs):
        self.data = data
        self.x = x
        self.y = y
        self._data = self._test_xy()
        self.nan_action = nan_action

    def _test_xy(self):
        if self.data is None:
            if isinstance(self.x, str) or isinstance(self.y, str):
                raise ValueError('Specify enter dataframe and `x` and `y` as'
                                 ' strings.')
            try:
                _data = pd.DataFrame(columns=['x', 'y'],
                                     data=np.array([self.x, self.y]).T)
            except TypeError:
                raise TypeError('X and Y are not of the same length')

            if np.nan(_data).max():
                if self.nan_action == 'raise':
                    raise ValueError('NaN in data, either specify action or '
                                     'remove')
                if self.nan_action == 'drop':
                    _data.dropna(inplace=True)
                if self.nan_action == 'replace':
                    if self.nan_action in ('mean', 'median', 'mode'):
                        raise NotImplementedError
        if isinstance(self.x, str) and isinstance(self.y, str):
            #if self.x in self.data.columns and self.y in self.data.columns:
            try:
                _data = self.data[[self.x, self.y]].copy()
            except KeyError:
                raise KeyError(f"Either `x` ({self.x}) or `y` ({self.y}) are not"
                        f"columns in data")

        else:
            raise ValueError('Either enter `data` as a pd.DataFrame'
                             'and `x` and `y` as two column names, or enter'
                             '`x` and `y` as two np.arrays')
        return _data


    def get_df(self):
        pass

    def get_text(self):
        pass

    def _finalize_results(self):
        pass

    def _run_analysis(self):
        pass


class ChiSquare(_BaseCorrelation):
    def __init__(self, apply_correction=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.crosstabulated_data = self._crosstab_data(self._data)
        self.apply_correction = apply_correction
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def get_df(self):
        return self.results.apply(
            pd.to_numeric, errors='ignore')

    def get_text(self, alpha=.05):
        _dat = self.get_df().to_dict('records')[0]
        significance_result = np.where(_dat['p.value'] < .05, 'a', 'no')
        main_clause = (f"A Chi-Square test of independence shown"
                       f" {significance_result} significant association between"
                       f" {self._data.columns[0]} and {self._data.columns[1]}"
                       )
        result_clause = (f"[\u03C7\u00B2({_dat['parameter']:.0f})" +
                         f" = {_dat['statistic']:.2f}, p = {_dat['p.value']:.3f}]")

        return f'{main_clause} {result_clause}.'

    def _crosstab_data(self, data):
        return pd.crosstab(*self._data.values.T)

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.chisq_test(self.crosstabulated_data,
                                                  correct=self.apply_correction)

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(self._r_results))


class Correlation(_BaseCorrelation):

    def __init__(self, method='pearson', **kwargs):
        super().__init__(**kwargs)

        if method not in ('pearson', 'spearman', 'kendall'):
            raise ValueError('Invalid correlation coefficient method - specify'
                             ' either `pearson`, `spearman` or `kendall`')
        self.method = method
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.cor_test(
            *self._data.values.T,
            method=self.method
        )


class PartCorreleation(_BaseCorrelation):

    def __init__(self, z=None, part_type=None, **kwargs):
        pass


# TODO - see what other
class BayesCorrelation(Correlation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run_analysis(self):
        return rst.pyr.rpackages.bayes_factor.correlationBF(
            *self._data.values.T,
            nullInterval=self.nullInterval,
            rscale_prior=None,
            posterior=self.posterior,
        )
