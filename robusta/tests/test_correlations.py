import unittest

import numpy as np
import pandas as pd

import robusta as rst


class Test_PairwiseCorrelation(unittest.TestCase):

    def test_two_str_and_empty_dataframe(self):
        with self.assertRaises(KeyError):
            rst.correlations._PairwiseCorrelation(
                x='x', y='y', data=pd.DataFrame()
            )


class TestChiSquare(unittest.TestCase):
    def test_chisquare_get_df(self):
        res = rst.ChiSquare(x='am', y='vs', data=rst.datasets.data('mtcars'),
                    apply_correction=True).get_results()
        r_res = rst.pyrio.r(
        """
        library(broom) 
        data.frame(tidy(chisq.test(table(mtcars[ , c('am', 'vs')]))))
        """
        )

        # TODO - find a way to avoid this ugly recasting ^

        pd.testing.assert_frame_equal(res, r_res)

    def test_chisquare_get_text(self):
        with self.assertRaises(NotImplementedError):

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


class Test_TriplewiseCorrelation(unittest.TestCase):

    def test_z_argument(self):
        data = rst.datasets.data('attitude')

        with self.assertRaises(ValueError):
            rst.correlations._TriplewiseCorrelation(
                x='x', y='y', z='z', data=None, method='pearson'
            )

            rst.correlations._TriplewiseCorrelation(
                x=[1, 2, 3], y=[4, 5, 2], z=[0, 2], data=None, method='pearson'
            )

        with self.assertRaises(KeyError):
            rst.correlations._TriplewiseCorrelation(
                x='x', y='y', z='ZZ', data=pd.DataFrame(
                    data=np.random.rand(10, 3), columns=['x', 'y', 'z']
                )
            )

        with self.assertRaises(ValueError):
            rst.correlations._TriplewiseCorrelation(
                x=[], y='y', z='z', data=pd.DataFrame(
                    columns=['x', 'y', 'z']
                )
            )


class TestPartialCorrelation(unittest.TestCase):

    def test_output(self):
        satv = [500, 550, 450, 400, 600, 650, 700, 550, 650, 550]
        hsgpa = [3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1]
        fgpa = [2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9]
        res = rst.PartialCorrelation(x=satv, y=hsgpa, z=fgpa, method='pearson'
                                     ).get_results()

        r_res = rst.pyrio.r(
        """
        library(ppcor)
        SATV <-  c(500, 550, 450, 400, 600, 650, 700, 550, 650, 550)
        HSGPA <- c(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1)
        FGPA <-  c(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9)
        pcor.test(SATV, HSGPA, FGPA)
        """
        )
        # The method column is returned as a categorical type
        r_res['Method'] = r_res['Method'].astype(str).values


        pd.testing.assert_frame_equal(res, r_res)


class TestPartCorrelation(unittest.TestCase):

    def test_output(self):
        satv = [500, 550, 450, 400, 600, 650, 700, 550, 650, 550]
        hsgpa = [3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1]
        fgpa = [2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9]
        res = rst.PartCorrelation(x=satv, y=hsgpa, z=fgpa, method='pearson'
                                  ).get_results()

        r_res = rst.pyrio.r(
        """
        # Now in R...
        library(ppcor)
        library(broom)
        SATV <-  c(500, 550, 450, 400, 600, 650, 700, 550, 650, 550)
        HSGPA <- c(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1)
        FGPA <-  c(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9)
        spcor.test(SATV, HSGPA, FGPA)
        """
        )

        # The method column is returned as a categorical type
        r_res['Method'] = r_res['Method'].astype(str).values

        pd.testing.assert_frame_equal(res, r_res, check_dtype=False)


class TestBayesCorrelation(unittest.TestCase):

    def test_output(self):
        res = rst.BayesCorrelation(x='Sepal.Width', y='Sepal.Length',
                                   data=rst.datasets.data('iris')).get_results(
        )
        r_res = rst.pyrio.r(
        """
        library(BayesFactor)
        library(tibble)

        r_res = rownames_to_column(
            data.frame(correlationBF(y = iris$Sepal.Length,
            x = iris$Sepal.Width))[, c('bf', 'error')], 'model')
        """
        )
        pd.testing.assert_frame_equal(res, r_res)

if __name__ == '__main__':
    unittest.main()
