import unittest

import pandas as pd
import numpy as np
import robusta as rst


class TestLinearRegression(unittest.TestCase):
    def test_simple_linear_regression(self):

        data = pd.DataFrame(columns=['group', 'weight'],
                            data=np.array([np.repeat(['Ctl', 'Trt'], 10),
                                           [4.17, 5.58, 5.18, 6.11, 4.50, 4.61,
                                            5.17, 4.53, 5.33, 5.14, 4.81, 4.17,
                                            4.41, 3.59, 5.87, 3.83, 6.03, 4.89,
                                            4.32, 4.69]]).T)
        data.reset_index(drop=False, inplace=True)
        data.rename(columns={'index': 'dataset_rownames'}, inplace=True)

        res = rst.LinearRegression(
            formula='weight ~ group + 1| dataset_rownames', data=data).get_results()
        r_res = rst.misc.pyrio.r(
        """
        # Source:
        # https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
        library(broom)
        ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
        trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
        group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
        weight <- c(ctl, trt)
        lm.D9 <- data.frame(tidy(lm(weight ~ group)))
        """
        )

        pd.testing.assert_frame_equal(res, r_res)

    def test_multiple_linear_regression(self):

        res = rst.LinearRegression(
            data=rst.misc.datasets.data('marketing'),
            formula='sales~youtube*facebook + 1|dataset_rownames',
            subject='dataset_rownames').get_results()

        r_res = rst.misc.pyrio.r(
        """
        library(datarium)
        library(broom)
        data.frame(tidy(lm(sales ~ youtube*facebook, data=marketing)))
        """
        )

        pd.testing.assert_frame_equal(res, r_res)

    def test_missing_columns(self):
        df = rst.misc.datasets.data('sleep')
        with self.assertRaises(KeyError):
            rst.LinearRegression(data=df, formula='extra ~ condition + 1|ID')



class TestBayesianLinearRegression(unittest.TestCase):

    def test_simple_bayesian_regression(self):
        res = rst.BayesianLinearRegression(data=rst.misc.datasets.data(
            'attitude'), formula='rating ~ complaints + 1|dataset_rownames').get_results()

        r_res = rst.misc.pyrio.r(
        """
        library(BayesFactor)
        library(tibble)
        rownames_to_column(
            data.frame(generalTestBF(
                rating ~ complaints, data=attitude)
            ), 'model')[, c('model', 'bf', 'error')]
        """
        )
        pd.testing.assert_frame_equal(res, r_res)

    def test_multiple_bayesian_regression(self):
        res = rst.BayesianLinearRegression(
            data=rst.misc.datasets.data('attitude'),
            formula='rating ~ privileges * complaints + raises + 1| dataset_rownames'
        ).get_results()

        r_res = rst.misc.pyrio.r(
        """
        library(BayesFactor)
        library(tibble)
        library(readr)
        r_res = rownames_to_column(
            data.frame(generalTestBF(
                rating ~ privileges * complaints + raises,
                data=attitude))[, c('bf', 'error')], 'model')
        r_res
        """
        )

        pd.testing.assert_frame_equal(res, r_res)

class TestLogisticRegression(unittest.TestCase):

    def test_simple_logistic_regression(self):

        res = rst.LogisticRegression(
            formula='group~extra+1 | ID', data=rst.misc.datasets.data('sleep')
                                     ).get_results()

        r_res = rst.misc.pyrio.r(
        """
        library(readr)
        library(broom)
        data.frame(tidy(glm(group ~ extra, family='binomial', data=sleep)))
        """
        )

        pd.testing.assert_frame_equal(res, r_res)


    def test_multiple_logistic_regression(self):

        res = rst.LogisticRegression(
            formula='group~extra+1 | ID', data=rst.misc.datasets.data('sleep')
                                     ).get_results()

        r_res = rst.misc.pyrio.r(
        """
        library(broom)
        data.frame(tidy(glm(group ~ extra, family='binomial', data=sleep)))
        """
        )

        pd.testing.assert_frame_equal(res, r_res)


if __name__ == '__main__':
    unittest.main()
