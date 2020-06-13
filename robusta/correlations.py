import pandas as pd
import numpy as np
import robusta as rst

__all__ = ['CorrChiSquare']


class _BaseCorrelation:

    def __init__(self, var1=None, var2=None, data=None, **kwargs):
        self.data = data
        self.var1 = var1
        self.var2 = var2
        self.x, self.y = self.data[self.var1, self.var2].values.T

    def get_df(self):
        pass

    def get_text(self):
        pass

    def _finalize_results(self):
        pass

    def _run_analysis(self):
        pass


class CorrChiSquare(_BaseCorrelation):
    def __init__(self, apply_correction=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.crosstabulated_data = self._crosstab_data(self.data)
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
                       f" {self.var1} and {self.var2}"
                       )
        result_clause = (f"[\u03C7\u00B2({_dat['parameter']:.0f})" +
                         f" = {_dat['statistic']:.2f}, p = {_dat['p.value']:.3f}]")

        return f'{main_clause} {result_clause}.'

    def _crosstab_data(self, data):
        return pd.crosstab(self.x, self.y)

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.chisq_test(self.crosstabulated_data,
                                                  correct=self.apply_correction)

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(self._r_results))


class CorrPearson(_BaseCorrelation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.cor_test(
            x=self.x,
            y=self.y,
            method='pearson'
        )


class CorrSpearman(_BaseCorrelation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.cor_test(
            x=self.x,
            y=self.y,
            method='spearman'
        )


class CorrKendall(_BaseCorrelation):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.cor_test(
            x=self.x,
            y=self.y,
            method='kendall'
        )


# TODO - see what other
class BayesCorrelation(CorrPearson, **kwargs):

    def __init__(self):
        super().__init__(**kwargs)

    def _run_analysis(self):
        return rst.pyr.rpackages.bayes_factor.correlationBF(
            x=self.x, y=self.y
        )
