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
        pd.testing.assert_frame_equal(
            res[['bf', 'error']],
            pd.DataFrame(columns=['bf', 'error'],
                         data=np.array([[34.416936, 2.549779e-07],
                                        [0.100825, 6.210317e-04]]))
        )


if __name__ == '__main__':
    unittest.main()
