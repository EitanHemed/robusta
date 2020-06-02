import unittest

import pandas as pd
import numpy as np

import robusta as rst


class TestLinearRegression(unittest.TestCase):
    def test_rdoc_example(self):
        data = pd.DataFrame(columns=['group', 'weight'],
                            data=np.array([np.repeat(['Ctl', 'Trt'], 10),
                                           [4.17, 5.58, 5.18, 6.11, 4.50, 4.61,
                                            5.17, 4.53, 5.33, 5.14, 4.81, 4.17,
                                            4.41, 3.59, 5.87, 3.83, 6.03, 4.89,
                                            4.32, 4.69]]).T)
        data['weight'] = data['weight'].astype(float).values

        res = rst.LinearRegression(formula='weight ~ group', data=data).get_df()
        print(res)

        r_res = pd.DataFrame(columns=['term', 'estimate', 'std.error',
                                      'statistic', 'p.value'],
                             data=[['(Intercept)', 5.032, 0.2202177, 22.85012,
                                    9.547128e-15],
                                   ['groupTrt', -0.371, 0.3114349, -1.19126,
                                    2.490232e-01]]).apply(
            pd.to_numeric, errors='ignore')
        print(r_res)

        pd.testing.assert_frame_equal(res, r_res)

    if __name__ == '__main__':
        unittest.main()
