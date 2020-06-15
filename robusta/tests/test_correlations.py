import unittest

import pandas as pd

import robusta as rst


class Test_BaseCorrelation(unittest.TestCase):

    def test_two_str_and_empty_dataframe(self):
        with self.assertRaises(KeyError):
            rst.correlations._BaseCorrelation(
                x='x', y='y', data=pd.DataFrame()
            )

class TestChiSquare(unittest.TestCase):
    def test_chisquare_get_df(self):
        res = rst.ChiSquare(x='am', y='vs', data=rst.datasets.data('mtcars')
                            ).get_df().drop(columns=['row_names'])
        """
        # Now in R...
        options(width=120)
        library(broom)
        library(readr) 
        data.frame(tidy(chisq.test(table(mtcars[ , c('am', 'vs')]))))
        """

        r_res = pd.read_csv(pd.compat.StringIO(
            """statistic,p.value,parameter,method
            0.34753550543024225,0.5555115470131495,1,Pearson's Chi-squared test with Yates' continuity correction"""
        ), lineterminator='\n', skipinitialspace=True,
            dtype={'parameter': 'int32'})
        # TODO - find a way to avoid this ugly recasting ^

        pd.testing.assert_frame_equal(res, r_res, check_exact=False,
                                      check_less_precise=5)

    def test_chisquare_get_text(self):
        nonsignificant_res = rst.ChiSquare(x='am', y='vs',
                                           data=rst.datasets.data('mtcars')
                                           ).get_text()
        self.assertEqual(
            nonsignificant_res,
            'A Chi-Square test of independence shown '
            'no significant association between am and '
            'vs [χ²(1) = 0.35, p = 0.556].')

        significant_res = rst.ChiSquare(x='gear', y='vs',
                                        data=rst.datasets.data('mtcars')
                                        ).get_text()
        self.assertEqual(
            significant_res,
            'A Chi-Square test of independence shown '
            'a significant association between gear and '
            'vs [χ²(2) = 12.22, p = 0.002].')


class TestCorrelation(unittest.TestCase):

    def test_correlation_method(self):
        data = rst.datasets.data('attitude')

        # Test that when no
        with self.assertRaises(ValueError):
            rst.Correlation(data=data, x='rating', y='advance',
                            method=None)
            rst.Correlation(data=data, x='rating', y='advance',
                            method='fisher')



if __name__ == '__main__':
    unittest.main()
