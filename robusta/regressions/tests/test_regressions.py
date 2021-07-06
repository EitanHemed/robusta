import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import r

import robusta as rst


def test_simple_linear_regression():
    data = pd.DataFrame(columns=['group', 'weight'],
                        data=np.array([np.repeat(['Ctl', 'Trt'], 10),
                                       [4.17, 5.58, 5.18, 6.11, 4.50, 4.61,
                                        5.17, 4.53, 5.33, 5.14, 4.81, 4.17,
                                        4.41, 3.59, 5.87, 3.83, 6.03, 4.89,
                                        4.32, 4.69]]).T)
    data.reset_index(drop=False, inplace=True)
    data.rename(columns={'index': 'dataset_rownames'}, inplace=True)

    r_res = r(
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
    m = rst.api.linear_regression(
        formula='weight ~ group + 1| dataset_rownames',
        data=data)

    res = m.fit()

    pd.testing.assert_frame_equal(res._get_r_output_df(), rst.misc.utils.convert_df(r_res))


def test_multiple_linear_regression():
    m = rst.api.linear_regression(
        data=rst.misc.datasets.load_dataset('marketing'),
        formula='sales~youtube*facebook + 1|dataset_rownames',
        subject='dataset_rownames')
    res = m.fit()

    r_res = r(
        """
        library(datarium)
        library(broom)
        data.frame(tidy(lm(sales ~ youtube*facebook, data=marketing)))
        """
    )

    pd.testing.assert_frame_equal(res._get_r_output_df(), rst.misc.utils.convert_df(r_res))


def test_missing_columns():
    df = rst.misc.datasets.load_dataset('sleep')
    with pytest.raises(KeyError):
        m = rst.api.linear_regression(
            data=df, formula='extra ~ condition + 1|ID')
        res = m.fit()


def test_simple_bayesian_regression():
    m = rst.api.bayes_linear_regression(data=rst.misc.datasets.load_dataset(
        'attitude'),
        formula='rating ~ complaints + 1|dataset_rownames')
    res = m.fit()

    r_res = r(
        """
        library(BayesFactor)
        library(tibble)
        rownames_to_column(
            data.frame(generalTestBF(
                rating ~ complaints, data=attitude)
            ), 'model')[, c('model', 'bf', 'error')]
        """
    )
    pd.testing.assert_frame_equal(res._get_r_output_df(), rst.misc.utils.convert_df(r_res))


def test_multiple_bayesian_regression():
    m = rst.api.bayes_linear_regression(
        data=rst.misc.datasets.load_dataset('attitude'),
        formula='rating ~ privileges * complaints + raises + 1| dataset_rownames'
    )
    res = m.fit()

    r_res = r(
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

    pd.testing.assert_frame_equal(res._get_r_output_df(), rst.misc.utils.convert_df(r_res))


def test_simple_logistic_regression():
    m = rst.api.logistic_regression(
        formula='group~extra+1 | ID', data=rst.misc.datasets.load_dataset('sleep')
    )
    res = m.fit()

    r_res = r(
        """
        library(readr)
        library(broom)
        data.frame(tidy(glm(group ~ extra, family='binomial', data=sleep)))
        """
    )

    pd.testing.assert_frame_equal(res._get_r_output_df(), rst.misc.utils.convert_df(r_res))


def test_multiple_logistic_regression():
    m = rst.api.logistic_regression(
        formula='group~extra+1 | ID', data=rst.misc.datasets.load_dataset('sleep')
    )
    res = m.fit()

    r_res = r(
        """
        library(broom)
        data.frame(tidy(glm(group ~ extra, family='binomial', data=sleep)))
        """
    )

    pd.testing.assert_frame_equal(res._get_r_output_df(), rst.misc.utils.convert_df(r_res))
