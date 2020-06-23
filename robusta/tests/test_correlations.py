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
        res = rst.ChiSquare(x='am', y='vs', data=rst.datasets.data('mtcars')
                            ).get_results().drop(columns=['row_names'])
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
                                     ).get_results().drop(columns=['row_names'])

        """
        # Now in R...
        library(ppcor)
        library(readr)
        SATV <-  c(500, 550, 450, 400, 600, 650, 700, 550, 650, 550)
        HSGPA <- c(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1)
        FGPA <-  c(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9)
        cat(format_csv(pcor.test(SATV, HSGPA, FGPA)))
        """

        r_res = pd.read_csv(pd.compat.StringIO("""
        estimate,p.value,statistic,n,gp,Method
        0.5499630584146348,0.12500736420985112,1.7421990896994466,10,1,pearson"""
                                               ), skipinitialspace=True,
                            dtype={'n': 'int64', 'gp': 'int64'})
        # TODO - figure out a way around the following problem - the read_csv
        #  function defines integers as int32, while the returned result is
        #  int 64.
        pd.testing.assert_frame_equal(res, r_res, check_dtype=False)


class TestPartCorrelation(unittest.TestCase):

    def test_output(self):
        satv = [500, 550, 450, 400, 600, 650, 700, 550, 650, 550]
        hsgpa = [3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1]
        fgpa = [2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9]
        res = rst.PartCorrelation(x=satv, y=hsgpa, z=fgpa, method='pearson'
                                  ).get_results().drop(columns=['row_names'])

        """
        # Now in R...
        library(ppcor)
        library(readr)
        SATV <-  c(500, 550, 450, 400, 600, 650, 700, 550, 650, 550)
        HSGPA <- c(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1)
        FGPA <-  c(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9)
        cat(format_csv(spcor.test(SATV, HSGPA, FGPA)))
        """

        r_res = pd.read_csv(pd.compat.StringIO("""
        estimate,p.value,statistic,n,gp,Method
        0.31912011585544464,0.4025675494246841,0.8908934739230737,10,1,pearson"""
                                               ), skipinitialspace=True,
                            dtype={'n': 'int64', 'gp': 'int64'})
        # TODO - figure out a way around the following problem - the read_csv
        #  function defines integers as int32, while the returned result is
        #  int 64.
        pd.testing.assert_frame_equal(res, r_res, check_dtype=False)


class TestBayesCorrelation(unittest.TestCase):

    def test_output(self):
        res = rst.BayesCorrelation(x='Sepal.Width', y='Sepal.Length',
                                   data=rst.datasets.data('iris')).get_results(
        ).drop(columns=['row_names'])

        """
        # Now in R
        options(width=120)
        library(BayesFactor)
        library(tibble)
        library(readr)

         r_res = rownames_to_column(
            data.frame(correlationBF(y = iris$Sepal.Length,
            x = iris$Sepal.Width))[, c('bf', 'error')], 'model')
        cat(format_delim(r_res, ';'))
        """

        r_res = pd.read_csv(pd.compat.StringIO("""
        model;bf;error
        Alt., r=0.333;0.5090175116477023;0"""), skipinitialspace=True,
                            sep=';')

if __name__ == '__main__':
    unittest.main()
