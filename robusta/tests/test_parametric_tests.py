import unittest
import sys

import numpy as np
import pandas as pd

sys.path.append('./')

import robusta as rst


class TestT1Sample(unittest.TestCase):

    def test_t1sample_get_df(self):
        sleep = rst.datasets.data('sleep')
        t = rst.T1Sample(data=sleep.loc[sleep['group'] == '1'],
                         dependent='extra', subject='ID', independent='group')
        res = t.get_df().to_dict('records')[0]
        [self.assertAlmostEqual(a1, a2, 4) for (a1, a2) in
         zip(
             res.values(),
             [0.75, 1.32571, 0.2175978, 9, -0.5297804, 2.02978,
              'One Sample t-test', 'two.sided'])]


class TestBayesT1Sample(unittest.TestCase):

    def test_bayes1sample(self):
        sleep = rst.datasets.data('sleep')
        data = sleep.loc[sleep['group'] == '1'].assign(
            diff_scores=sleep.groupby('ID')['extra'].apply(
                lambda s: s.diff(-1).max()).values)
        res = rst.BayesT1Sample(data=data,
                                dependent='diff_scores', subject='ID',
                                independent='group',
                                null_interval=[-np.inf, 0]).get_df()
        r_res = pd.DataFrame(columns=['bf', 'error'],
                             data=np.array([[34.416936, 2.549779e-07],
                                            [0.100825, 6.210317e-04]]))
        pd.testing.assert_frame_equal(
            res[['bf', 'error']],
            r_res
        )


class TestAnova(unittest.TestCase):

    def test_between_anova(self):
        anova = rst.Anova(data=rst.datasets.data('ToothGrowth'),
                          dependent='len', subject='row_names',
                          between=['supp', 'dose']).get_df().round(4)

        r_res = pd.DataFrame(
            data=np.array([
                ['supp', 'dose', 'supp:dose'],
                [1.0, 2.0, 2.0],
                [54.0] * 3,
                [13.1871418] * 3,
                [15.57197945, 91.99996489, 4.10699109],
                [0.22382545, 0.77310918, 0.13202791],
                [0.0002, 0.0000, 0.0219]
            ]).T,
            columns=['term', 'num.Df', 'den.Df', 'MSE', 'statistic', 'pes',
                     'p.value']).apply(pd.to_numeric, errors='ignore').round(4)

        pd.testing.assert_frame_equal(anova, r_res, check_exact=True)

    def test_margins(self):
        df = rst.datasets.data('anxiety').drop(columns=['row_names']).set_index(
            ['id', 'group']).stack().reset_index().rename(columns={0: 'score',
                                                                   'level_2': 'time'})
        anova_time = rst.Anova(data=df, within='time',
                               dependent='score', subject='id')
        margins_time = anova_time.get_margins('time')

        anova_time_by_group = rst.Anova(
            data=df, between='group', within='time',
            dependent='score', subject='id')

        margins_time_by_group = anova_time_by_group.get_margins(
            ('time', 'group'))
        """
        # Now in R...
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)
        margins_time = emmeans(aov_ez(data=data_long, between='group', 
            within='time', dv='score', id='id'), 'time', type='response')

        margins_time_by_group = emmeans(aov_ez(data=data_long, between='group', 
            within='time', id='id', dv='score'), c('time', 'group'), 
            type='response')
        """
        r_res_time = pd.read_csv(pd.compat.StringIO("""
               time   emmean        SE       df lower.CL upper.CL
               t1 16.91556 0.2323967 43.99952 16.44719 17.38392
               t2 16.13556 0.2323967 43.99952 15.66719 16.60392
               t3 15.19778 0.2323967 43.99952 14.72941 15.66614"""),
                                 delim_whitespace=True,
                                 dtype={'time': 'category'})
        r_res_time_by_group = pd.read_csv(pd.compat.StringIO("""
        time group   emmean        SE       df lower.CL upper.CL
               t1  grp1 17.08667 0.4025229 43.99952 16.27543 17.89790
               t2  grp1 16.92667 0.4025229 43.99952 16.11543 17.73790
               t3  grp1 16.50667 0.4025229 43.99952 15.69543 17.31790
               t1  grp2 16.64667 0.4025229 43.99952 15.83543 17.45790
               t2  grp2 16.46667 0.4025229 43.99952 15.65543 17.27790
               t3  grp2 15.52667 0.4025229 43.99952 14.71543 16.33790
               t1  grp3 17.01333 0.4025229 43.99952 16.20210 17.82457
               t2  grp3 15.01333 0.4025229 43.99952 14.20210 15.82457
               t3  grp3 13.56000 0.4025229 43.99952 12.74877 14.37123"""),
                                          delim_whitespace=True,
                                          dtype={'time': 'category',
                                                 'group': 'category'})
        pd.testing.assert_frame_equal(
            margins_time, r_res_time, check_exact=False, check_less_precise=5)
        pd.testing.assert_frame_equal(
            margins_time_by_group, r_res_time_by_group, check_exact=False,
            check_less_precise=5)


class TestBayesAnova(unittest.TestCase):

    def test_bayes_anova(self):
        # TODO write a better test case - as Bayes factors can get obscenely
        #  large or small rounding errors can fail the test.
        #  also the fact that

        anova = rst.BayesAnova(data=rst.datasets.data('ToothGrowth'),
                               dependent='len', subject='row_names',
                               between=['supp', 'dose'], iterations=100000
                               ).get_df()  # .round(4)

        anova = anova.loc[anova['model'].isin(['supp', 'dose'])]

        r_res = pd.DataFrame(
            data=np.array([
                ['supp', 1.198757e+00, 8.941079e-05],
                ['dose', 4.983636e+12, 1.189630e-08],
                # ['supp + dose', 2.878842e+14, 1.592787e-03],
                # ['supp + dose + supp:dose', 7.743991e+14, 2.729397e-03]
            ]),
            columns=['model', 'bf', 'error']).apply(pd.to_numeric,
                                                    errors='ignore')  # .round(4)

        pd.testing.assert_series_equal(anova['model'], r_res['model'],
                                       check_exact=False)
        # The clanky testing is because of the great/small magnitude of
        # Bayes factors and their error terms.
        [np.testing.assert_approx_equal(y1, y2)
         for (y1, y2) in zip(
            anova[['bf', 'error']].values.flatten(),
            r_res[['bf', 'error']].values.flatten())]


if __name__ == '__main__':
    unittest.main()
