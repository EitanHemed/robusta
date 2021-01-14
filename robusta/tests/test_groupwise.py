import unittest
import sys
import numpy as np
import pandas as pd
import robusta as rst

sys.path.append('./')

class TestT1Sample(unittest.TestCase):

    def test_t1sample_get_results(self):
        sleep = rst.datasets.data('sleep')
        res = rst.T1Sample(data=sleep.loc[sleep['group'] == '1'],
                           dependent='extra', subject='ID',
                           independent='group').get_results()
        r_res = rst.pyrio.r(
        """
        library(broom)
        data.frame(tidy(t.test(x=sleep[sleep$group == 1, 'extra'])))
        """
        )
        pd.testing.assert_frame_equal(res, r_res)


class TestBayesT1Sample(unittest.TestCase):

    def test_bayes1sample(self):
        sleep = rst.datasets.data('sleep')
        data = sleep.loc[sleep['group'] == '1'].assign(
            diff_scores=sleep.groupby('ID')['extra'].apply(
                lambda s: s.diff(-1).max()).values)
        res = rst.BayesT1Sample(data=data,
                                dependent='diff_scores', subject='ID',
                                independent='group',
                                null_interval=[-np.inf, 0]).get_results()
        r_res = rst.pyrio.r(
        """
        library(BayesFactor)
        diff = (sleep[sleep$group == 1, 'extra'] 
            - sleep[sleep$group == 2, 'extra'])
        res = data.frame(
            ttestBF(
                diff, nullInterval=c(-Inf, 0)))[, c('bf', 'error')]
        """
        )
        pd.testing.assert_frame_equal(
            res[['bf', 'error']],
            r_res
        )


class TestAnova(unittest.TestCase):

    def test_between_anova(self):
        anova = rst.Anova(data=rst.datasets.data('ToothGrowth'),
                          dependent='len', subject='dataset_rownames',
                          between=['supp', 'dose']).get_results()
        r_res = rst.pyrio.r(
        """library(broom)
        library(afex)
        library(tibble)
        data.frame(
            tidy(
                anova(
                    aov_ez(rownames_to_column(ToothGrowth, 'dataset_rownames'),
                            id='dataset_rownames', between=c('supp', 'dose'),
                            dv='len', es='pes')
                    )
                )
        )
        """)

        pd.testing.assert_frame_equal(anova, r_res)

    def test_margins(self):
        df = rst.datasets.data('anxiety').set_index(
            ['id', 'group']).filter(regex='^t[1-3]$').stack().reset_index().rename(columns={0: 'score',
                                                                   'level_2': 'time'})
        anova_time = rst.Anova(data=df, within='time',
                               dependent='score', subject='id')
        margins_time = anova_time.get_margins('time')

        anova_time_by_group = rst.Anova(
            data=df, between='group', within='time',
            dependent='score', subject='id')
        margins_time_by_group = anova_time_by_group.get_margins(
            ('time', 'group'))

        # TODO - do this in one call to r (i.e., return a vector
        #  of two dataframes).
        r_res_time = rst.pyrio.r(
        """
        library(afex)
        library(emmeans)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)
        
        r_res_time = data.frame(emmeans(aov_ez(data=data_long,
                within='time', dv='score', id='id'), 'time', type='response'))
        """
        )
        r_res_time_by_group = rst.pyrio.r(
        """
        r_res_time_by_group = data.frame(emmeans(aov_ez(data=data_long, between='group', 
            within='time', id='id', dv='score'), c('time', 'group'), 
            type='response'))
        """
        )

        pd.testing.assert_frame_equal(
            margins_time, r_res_time)
        pd.testing.assert_frame_equal(
            margins_time_by_group, r_res_time_by_group)


class TestBayesAnova(unittest.TestCase):

    def test_bayes_anova(self):
        # TODO write a better test case - as Bayes factors can get obscenely
        #  large or small rounding errors can fail the test.
        #  also the fact that

        anova = rst.BayesAnova(data=rst.datasets.data('ToothGrowth'),
                               dependent='len', subject='dataset_rownames',
                               between=['supp', 'dose'], iterations=100000
                               ).get_results()  # .round(4)
        r_res = rst.pyrio.r(
        """
        ToothGrowth$dose = as.factor(ToothGrowth$dose)
        rownames_to_column(
            data.frame(
                anovaBF(len ~ supp*dose, data=ToothGrowth),
                iterations=10000)[ , c('bf', 'error')], 'model')
        """
        )

        # TODO - currently there is very partial testing of the values magnitude
        #    as they are either very large or very small.

        # The partial testing is because of the great/small magnitude of
        # Bayes factors and their error terms.
        pd.testing.assert_frame_equal(anova.iloc[:2], r_res.iloc[:2])


if __name__ == '__main__':
    unittest.main()
