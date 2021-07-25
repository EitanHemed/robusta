import typing
import warnings
import pandas as pd
import numpy as np
from .. import pyr
from ..misc import utils, base

BF_COLUMNS = ['model', 'bf', 'error']
REDUNDANT_BF_COLUMNS = ['time', 'code']

DEFAULT_GROUPWISE_NULL_INTERVAL = (-np.inf, np.inf)
PRIOR_SCALE_STR_OPTIONS = ('medium', 'wide', 'ultrawide')
DEFAULT_ITERATIONS: int = 10000
DEFAULT_MU: float = 0.0

T_TEST_COLUMNS_RENAME = {'statistic': 't', 'parameter': 'df', 'estimate': 'mean', 'p.value': 'p-value'}
T_TEST_COHEN_COLUMN_NAMES = ['Cohen-d Low', 'Cohen-d', 'Cohen-d High']
T_TEST_RETURNED_COLUMNS = [T_TEST_COLUMNS_RENAME['statistic'], T_TEST_COLUMNS_RENAME['parameter'],
                           T_TEST_COLUMNS_RENAME['p.value']]

BAYES_T_TEST_COLUMNS_RENAME = {'bf': 'BF', 'error': 'BF-Error'}
BAYES_T_TEST_RETURNED_COLUMNS = BAYES_T_TEST_COLUMNS_RENAME.values()

WILCOX_COLUMNS_RENAME = {'statistic': 'Z', 'p.value': 'p-value'}
WILCOX_RETURNED_COLUMNS = WILCOX_COLUMNS_RENAME.values()

ANOVA_COLUMNS_RENAME = {'Effect': 'Term', 'p.value': 'p-value', 'pes': 'Partial Eta-Squared'}
ANOVA_DF_COLUMNS = ['df1', 'df2']
ANOVA_RETURNED_COLUMNS = list(ANOVA_COLUMNS_RENAME.values()) + ['F', 'df']

KWT_COLUMNS_RENAME = {'statistic': 'H', 'p.value': 'p-value', 'parameter': 'df'}
KWT_RETURNED_COLUMNS = KWT_COLUMNS_RENAME.values()

ART_COLUMNS_RENAME = {'Pr..F.': 'p-value', 'Df': ANOVA_DF_COLUMNS[0], 'Df.res': ANOVA_DF_COLUMNS[1]}
ART_RETURNED_COLUMNS = ANOVA_RETURNED_COLUMNS



class GroupwiseResults(base.BaseResults):

    def get_text(self, mode: str = 'df'):
        raise NotImplementedError


class TTestResults(GroupwiseResults):
    columns_rename = T_TEST_COLUMNS_RENAME
    returned_columns = T_TEST_RETURNED_COLUMNS

    def _reformat_r_output_df(self):
        df = super()._reformat_r_output_df()

        vals = df.to_dict('records')[0]
        # TODO - how to handle unequal group size
        cohen_d = pyr.rpackages.psych.t2d(vals['t'], n=vals[self.columns_rename['parameter']])
        cohen_d_ci = pyr.rpackages.psych.d_ci(cohen_d, n=vals[self.columns_rename['parameter']])

        df[T_TEST_COHEN_COLUMN_NAMES] = cohen_d_ci  # Skip the middle value

        return df[self.returned_columns + T_TEST_COHEN_COLUMN_NAMES]


class BayesResults(GroupwiseResults):
    # TODO check whether correctly inherits __init__ from parent
    columns_rename = BAYES_T_TEST_COLUMNS_RENAME
    returned_columns = BAYES_T_TEST_RETURNED_COLUMNS

    def __init__(self, r_results, mode='bf'):
        self.mode = mode

        super().__init__(r_results)

    def _tidy_results(self):
        if self.mode == 'bf':
            return utils.convert_df(self.r_results, 'model')[BF_COLUMNS]
        else:
            return utils.convert_df(self.r_results)

    def _get_r_output_df(self):
        return self._tidy_results()

    def _reformat_r_output_df(self):
        return self.get_df()

    def get_df(self):
        return self._get_r_output_df()


class WilcoxonResults(GroupwiseResults):

    columns_rename = WILCOX_COLUMNS_RENAME
    returned_columns = WILCOX_RETURNED_COLUMNS


class AnovaResults(GroupwiseResults):

    columns_rename = ANOVA_COLUMNS_RENAME
    returned_columns = ANOVA_RETURNED_COLUMNS

    def _tidy_results(self):
        # TODO - this is an ugly hack. The problem is that up until now we didn't have the auto convertsion to recarray
        #  that removes the rownames. This needs to be fixed.
        return pyr.rpackages.afex.nice(self.r_results, es='pes')

        # anova_table = utils.convert_df(
        #     pyr.rpackages.stats.anova(
        #         self.r_results))
        # missing_column = utils.convert_df(pyr.rpackages.afex.nice(self.r_results))['Effect'].values
        # anova_table.insert(loc=0, column='term', value=missing_column)
        # return anova_table

    def _reformat_r_output_df(self):
        df = super()._reformat_r_output_df()
        _dofs = pd.DataFrame(df['df'].str.split(', ').tolist(), columns=ANOVA_DF_COLUMNS,
                             index=df.index)
        df = pd.concat([df.drop(columns=['df']), _dofs], axis=1)

        # TODO - Refactor this.
        df['F'] = df['F'].str.extract(r'(\d*\.\d+|\d+)').astype(float).values
        df['p-value'] = df['p-value'].str.extract(r'(\d*\.\d+|\d+)').astype(float).values
        df['df1'] = df['df1'].str.extract(r'(\d*\.\d+|\d+)').astype(float).values
        df['df2'] = df['df2'].str.extract(r'(\d*\.\d+|\d+)').astype(float).values
        df[ANOVA_COLUMNS_RENAME['pes']] = df[ANOVA_COLUMNS_RENAME['pes']].str.extract(r'(\d*\.\d+|\d+)').astype(float).values
        #df[[]]apply(lambda s: s.str.extract(r'(\d*\.\d+|\d+)').astype(float)).values

        return df

class KruskalWallisTestResults(GroupwiseResults):
    columns_rename = KWT_COLUMNS_RENAME
    returned_columns = KWT_RETURNED_COLUMNS

    def _tidy_results(self):
        return pyr.rpackages.generics.tidy(self.r_results)

    def get_margins(self):
        raise NotImplementedError("Not applicable to non-parametric ANOVA")


class FriedmanTestResults(KruskalWallisTestResults):

    def get_margins(self):
        raise NotImplementedError("Not applicable to non-parametric ANOVA")


class AlignedRanksTestResults(GroupwiseResults):
    columns_rename = ART_COLUMNS_RENAME
    returned_columns = ART_RETURNED_COLUMNS

    def get_margins(self):
        raise NotImplementedError("Not applicable to non-parametric ANOVA")

    def _tidy_results(self):
        return pyr.rpackages.stats.anova(
            self.r_results)

