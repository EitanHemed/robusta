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
        """"
        # Now in R...
        library(broom)
        ctl <- c(4.17,5.58,5.18,6.11,4.50,4.61,5.17,4.53,5.33,5.14)
        trt <- c(4.81,4.17,4.41,3.59,5.87,3.83,6.03,4.89,4.32,4.69)
        group <- gl(2, 10, 20, labels = c("Ctl","Trt"))
        weight <- c(ctl, trt)
        lm.D9 <- data.frame(tidy(lm(weight ~ group)))
        # Source:
        # https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
        """
        r_res = pd.read_csv(pd.compat.StringIO(
            """term estimate std.error statistic p.value
         (Intercept) 5.032 0.2202177 22.85012 9.547128e-15
         groupTrt -0.371 0.3114349 -1.19126 2.490232e-01"""),
            delim_whitespace=True)


        pd.testing.assert_frame_equal(res, r_res, check_exact=False,
                                      check_less_precise=5)

    def test_multiple_linear_regression(self):

        res = rst.LinearRegression(
            data=rst.datasets.data('marketing'),
            formula='sales~youtube*facebook + 1|dataset_rownames',
            subject='dataset_rownames').get_results()

        """# Now in R...
        library(datarium)
        library(broom)
        data.frame(tidy(lm(sales ~ youtube*facebook, data=marketing)))
        """
        r_res = pd.read_csv(pd.compat.StringIO("""
                      term     estimate    std.error statistic      p.value
              (Intercept) 8.1002642437 2.974456e-01 27.232755 1.541461e-68
                  youtube 0.0191010738 1.504146e-03 12.698953 2.363605e-27
                 facebook 0.0288603399 8.905273e-03  3.240815 1.400461e-03
         youtube:facebook 0.0009054122 4.368366e-05 20.726564 2.757681e-51"""),
                            delim_whitespace=True)
        pd.testing.assert_frame_equal(res, r_res, check_exact=False,
                                      check_less_precise=5)

    def test_missing_columns(self):
        df = rst.datasets.data('sleep')
        with self.assertRaises(KeyError):
            rst.LinearRegression(data=df, formula='extra ~ condition + 1|ID')



class TestBayesianLinearRegression(unittest.TestCase):

    def test_simple_bayesian_regression(self):
        res = rst.BayesianLinearRegression(data=rst.datasets.data(
            'attitude'), formula='rating ~ complaints + 1|dataset_rownames').get_results()

        """
        # Now in R...
        library(BayesFactor)
        library(tibble)
        rownames_to_column(
            data.frame(generalTestBF(
                rating ~ complaints, data=attitude)
            ), 'model')[, c('model', 'bf', 'error')]"""
        r_res = pd.read_csv(pd.compat.StringIO("""
               model       bf        error
            complaints 417938.6 8.103442e-05"""), delim_whitespace=True)

        pd.testing.assert_frame_equal(res, r_res, check_exact=False,
                                      check_less_precise=5)

    def test_multiple_bayesian_regression(self):
        res = rst.BayesianLinearRegression(
            data=rst.datasets.data('attitude'),
            formula='rating ~ privileges * complaints + raises + 1| dataset_rownames'
        ).get_results()

        """
        # Now in R...
        options(width=120)
        library(BayesFactor)
        library(tibble)
        library(readr)
        r_res = rownames_to_column(
            data.frame(generalTestBF(
                rating ~ privileges * complaints + raises,
                data=attitude))[, c('bf', 'error')], 'model')
        cat(format_csv(r_res))
        """
        r_res = pd.read_csv(pd.compat.StringIO("""model,bf,error
                privileges,3.1777840803638457,4.909535745074062e-6
                complaints,417938.63103894354,8.103441775038745e-5
                privileges + complaints,75015.22791152913,2.2572461956397294e-6
                privileges + complaints + privileges:complaints,22722.666671159677,3.8549279462551135e-7
                raises,47.00917068489743,7.99188994864696e-5
                privileges + raises,25.899240795685817,1.2255655318996648e-7
                complaints + raises,77498.98872118635,2.2495934504310574e-6
                privileges + complaints + raises,18230.7918028669,2.756405679591185e-7
                privileges + complaints + privileges:complaints + raises,6308.610998910358,3.0041097303250905e-6
                """), lineterminator='\n', skipinitialspace =True)

        pd.testing.assert_frame_equal(res, r_res, check_exact=False,
                                      check_less_precise=5)

class TestLogisticRegression(unittest.TestCase):

    def test_simple_logistic_regression(self):

        res = rst.LogisticRegression(
            formula='group~extra+1 | ID', data=rst.datasets.data('sleep')
                                     ).get_results()

        """
        # Now in R...
        library(readr)
        library(broom)
        cat(format_csv(
            data.frame(tidy(glm(group ~ extra, family='binomial', data=sleep)))
            ))
        """
        r_res = pd.read_csv(
            pd.compat.StringIO("""
                term,estimate,std.error,statistic,p.value
                (Intercept),-0.6928304001240985,0.6232561393553371,-1.111630285488604,0.2662971454269263
                extra,0.46520171520154907,0.27660635201304135,1.681818627142792,0.09260401556399948"""),
        skipinitialspace=True)

        pd.testing.assert_frame_equal(res, r_res)


    def test_multiple_logistic_regression(self):

        res = rst.LogisticRegression(
            formula='group~extra+1 | ID', data=rst.datasets.data('sleep')
                                     ).get_results()

        """
        # Now in R...
        library(readr)
        library(broom)
        cat(format_csv(
            data.frame(tidy(glm(group ~ extra, family='binomial', data=sleep)))
            ))
        """
        r_res = pd.read_csv(
            pd.compat.StringIO("""
                term,estimate,std.error,statistic,p.value
                (Intercept),-0.6928304001240985,0.6232561393553371,-1.111630285488604,0.2662971454269263
                extra,0.46520171520154907,0.27660635201304135,1.681818627142792,0.09260401556399948"""),
        skipinitialspace=True)

        pd.testing.assert_frame_equal(res, r_res)


if __name__ == '__main__':
    unittest.main()
