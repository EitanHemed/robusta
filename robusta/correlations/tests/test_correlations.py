import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import r

import robusta as rst
# from ..correlations_models import _PairwiseCorrelationModel, _TriplewiseCorrelationModel
from robusta.correlations.models import (_PairwiseCorrelationModel,
                                         _TriplewiseCorrelationModel)

BIVARIATE_CORRELEATION_METHODS = ['kendall', 'spearman', 'pearson']

ATTITUDE_DATA, MTCARS_DATA, IRIS_DATA = map(rst.datasets.load_dataset,
                                            ['attitude', 'mtcars', 'iris'])
TRIPLEWISE_CORRS_XYZ = dict(
    x=(500, 550, 450, 400, 600, 650, 700, 550, 650, 550),
    y=(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1),
    z=(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9))


def test_pairwise_correlation_faulty_input():
    with pytest.raises(KeyError):
        _PairwiseCorrelationModel(
            x='x', y='y', data=pd.DataFrame()).fit()
    with pytest.raises(ValueError):
        _PairwiseCorrelationModel(
            x='x', y='y', data=None
        ).fit()
    with pytest.raises(ValueError):
        _PairwiseCorrelationModel(
            x=np.random.randint(0, 10, 20),
            y=np.random.randint(0, 10, 19), data=None
        ).fit()

    # X is string, y is an array
    with pytest.raises(ValueError):
        _PairwiseCorrelationModel(
            x='rating',
            y=np.random.randint(0, 10, 19),
            data=ATTITUDE_DATA
        ).fit()

    # Input strings but both are not in the data
    with pytest.raises(ValueError):
        _PairwiseCorrelationModel(
            X='SCORE',
            y='SALES',
            data=ATTITUDE_DATA
        ).fit()


def test_chi_square_output():
    m = rst.api.chisquare(x='am', y='vs', data=MTCARS_DATA,
                          apply_correction=True)
    res = m.fit()
    r_res = r(
        """
        library(broom) 
        data.frame(tidy(chisq.test(table(mtcars[ , c('am', 'vs')]))))
        """
    )
    pd.testing.assert_frame_equal(res._get_r_output_df(), r_res)


# def test_chisquare_get_text():
#     with pytest.raises(NotImplementedError):
#         nonsignificant_res = rst.ChiSquare(x='am', y='vs',
#                                            data=MTCARS_DATA
#                                            ).get_text()
#         self.assertEqual(
#             nonsignificant_res,
#             'A Chi-Square test of independence shown '
#             'no significant association between am and '
#             'vs [χ²(1) = 0.35, p = 0.556].')
#
#         significant_res = rst.ChiSquare(x='gear', y='vs',
#                                         data=MTCARS_DATA
#                                         ).get_text()
#         self.assertEqual(
#             significant_res,
#             'A Chi-Square test of independence shown '
#             'a significant association between gear and '
#             'vs [χ²(2) = 12.22, p = 0.002].')

def test_correlation_faulty_input():
    # Faulty method argument
    with pytest.raises(ValueError):
        # Incorrect method specified
        rst.api.correlation(data=ATTITUDE_DATA, x='rating', y='advance',
                            method='fisher').fit()


@pytest.mark.parametrize('method', BIVARIATE_CORRELEATION_METHODS)
def test_correleation_output(method):
    m = rst.api.correlation(data=IRIS_DATA,
                            x='Sepal.Length', y='Sepal.Width',
                            method=method)
    res = m.fit()

    r_res = r(f"""
    library(broom)
    data.frame(tidy(cor.test(x=iris$Sepal.Length, y=iris$Sepal.Width,
        method='{method}'
    ))) 
    """)
    pd.testing.assert_frame_equal(res._get_r_output_df(), r_res)


def test_triplewise_correlation_z_argument():
    with pytest.raises(ValueError):
        _TriplewiseCorrelationModel(
            x='x', y='y', z='z', data=None, method='pearson'
        ).fit()

    with pytest.raises(ValueError):
        _TriplewiseCorrelationModel(
            x=[1, 2, 3], y=[4, 5, 2], z=[0, 2], data=None, method='pearson'
        ).fit()

    with pytest.raises(KeyError):
        _TriplewiseCorrelationModel(
            x='x', y='y', z='ZZ', data=pd.DataFrame(
                data=np.random.rand(10, 3), columns=['x', 'y', 'z']
            )
        ).fit()

    with pytest.raises(ValueError):
        _TriplewiseCorrelationModel(
            x=[], y='y', z='z', data=pd.DataFrame(
                columns=['x', 'y', 'z']
            )
        ).fit()


def test_partial_correleation_output():
    m = rst.api.part_correlation(method='pearson',
                                 **TRIPLEWISE_CORRS_XYZ)
    res = m.fit()

    r_res = r(
        """
        library(ppcor)
        SATV <-  c(500, 550, 450, 400, 600, 650, 700, 550, 650, 550)
        HSGPA <- c(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1)
        FGPA <-  c(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9)
        spcor.test(c{x}, c{y}, c{z})
        """.format(**TRIPLEWISE_CORRS_XYZ)
    )
    # The method column is returned as a categorical type
    # r_res['Method'] = r_res['Method'].astype(str).values

    pd.testing.assert_frame_equal(res._get_r_output_df(), r_res)


def test_part_output():
    m = rst.api.partial_correlation(**TRIPLEWISE_CORRS_XYZ, method='pearson'
                                    )
    res = m.fit()

    r_res = r(
        """
        # Now in R...
        library(ppcor)
        library(broom)
        # SATV <-  c({x})
        # HSGPA <- c({y})
        # FGPA <-  c({z})
        pcor.test(c{x}, c{y}, c{z})
        """.format(**TRIPLEWISE_CORRS_XYZ)
    )

    # The method column is returned as a categorical type
    # r_res['Method'] = r_res['Method'].astype(str).values

    pd.testing.assert_frame_equal(res._get_r_output_df(), r_res, check_dtype=False)


def test_bayes_correleation():
    m = rst.api.bayes_correlation(x='Sepal.Width', y='Sepal.Length',
                                  data=IRIS_DATA)
    res = m.fit()
    r_res = r(
        """
        library(BayesFactor)
        library(tibble)

        r_res = rownames_to_column(
            data.frame(correlationBF(y = iris$Sepal.Length,
            x = iris$Sepal.Width))[, c('bf', 'error')], 'model')
        """
    )
    pd.testing.assert_frame_equal(res._get_r_output_df(), r_res)
