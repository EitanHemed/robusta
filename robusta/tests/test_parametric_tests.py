import unittest
import sys

import numpy as np
import pandas as pd

sys.path.append('./')

import robusta as rst


class TestT1Sample(unittest.TestCase):

    def test_t1sample_get_df(self):
        sleep = rst.datasets.load('sleep')
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
        sleep = rst.datasets.load('sleep')
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
        anova = rst.Anova(data=rst.datasets.load('ToothGrowth'),
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


class TestBayesAnova(unittest.TestCase):

    def test_bayes_anova(self):
        # TODO write a better test case - as Bayes factors can get obscenely
        #  large or small rounding errors can fail the test.
        #  also the fact that

        anova = rst.BayesAnova(data=rst.datasets.load('ToothGrowth'),
                               dependent='len', subject='row_names',
                               between=['supp', 'dose'], iterations=100000
                               ).get_df()  # .round(4)

        anova = anova.loc[anova['model'].isin(['supp', 'dose'])]

        r_res = pd.DataFrame(
            data=np.array([
                ['supp', 1.198757e+00, 8.941079e-05],
                ['dose', 4.983636e+12, 1.189630e-08],
                #['supp + dose', 2.878842e+14, 1.592787e-03],
                #['supp + dose + supp:dose', 7.743991e+14, 2.729397e-03]
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
