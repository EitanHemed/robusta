import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import r

import robusta as rst

from robusta.correlations.models import (_PairwiseCorrelation,
                                         _TriplewiseCorrelation)

BIVARIATE_CORRELEATION_METHODS = ['kendall', 'spearman', 'pearson']

ATTITUDE_DATA, MTCARS_DATA, IRIS_DATA = map(rst.load_dataset,
                                            ['attitude', 'mtcars', 'iris'])
TRIPLEWISE_CORRS_XYZ = dict(
    x=(500, 550, 450, 400, 600, 650, 700, 550, 650, 550),
    y=(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1),
    z=(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9))


def test_pairwise_correlation_faulty_input():
    with pytest.raises(KeyError):
        _PairwiseCorrelation(
            x='x', y='y', data=pd.DataFrame())
    with pytest.raises(ValueError):
        _PairwiseCorrelation(
            x='x', y='y', data=None
        )
    with pytest.raises(ValueError):
        _PairwiseCorrelation(
            x=np.random.randint(0, 10, 20),
            y=np.random.randint(0, 10, 19), data=None
        )

    # X is string, y is an array
    with pytest.raises(ValueError):
        _PairwiseCorrelation(
            x='rating',
            y=np.random.randint(0, 10, 19),
            data=ATTITUDE_DATA
        )

    # Input strings but both are not in the data
    with pytest.raises(ValueError):
        _PairwiseCorrelation(
            X='SCORE',
            y='SALES',
            data=ATTITUDE_DATA
        )


def test_chi_square_output():
    m = rst.correlations.models.ChiSquare(x='am', y='vs', data=MTCARS_DATA,
                          apply_correction=True)

    r_res = r(
        """
        library(broom) 
        data.frame(tidy(chisq.test(table(mtcars[ , c('am', 'vs')]))))
        """
    )
    pd.testing.assert_frame_equal(m._results._get_r_output_df(), r_res)

def test_correlation_faulty_input():
    # Faulty method argument
    with pytest.raises(ValueError):
        # Incorrect method specified
        rst.correlations.models.Correlation(data=ATTITUDE_DATA, x='rating', y='advance',
                            method='fisher')

@pytest.mark.parametrize('method', BIVARIATE_CORRELEATION_METHODS)
def test_correleation_output(method):
    m = rst.correlations.models.Correlation(data=IRIS_DATA,
                            x='Sepal.Length', y='Sepal.Width',
                            method=method)

    r_res = r(f"""
    library(broom)
    data.frame(tidy(cor.test(x=iris$Sepal.Length, y=iris$Sepal.Width,
        method='{method}'
    ))) 
    """)
    pd.testing.assert_frame_equal(m._results._get_r_output_df(), r_res)


def test_triplewise_correlation_z_argument():
    with pytest.raises(ValueError):
        _TriplewiseCorrelation(
            x='x', y='y', z='z', data=None, method='pearson'
        )

    with pytest.raises(ValueError):
        _TriplewiseCorrelation(
            x=[1, 2, 3], y=[4, 5, 2], z=[0, 2], data=None, method='pearson'
        )

    with pytest.raises(KeyError):
        _TriplewiseCorrelation(
            x='x', y='y', z='ZZ', data=pd.DataFrame(
                data=np.random.rand(10, 3), columns=['x', 'y', 'z']
            )
        )

    with pytest.raises(ValueError):
        _TriplewiseCorrelation(
            x=[], y='y', z='z', data=pd.DataFrame(
                columns=['x', 'y', 'z']
            )
        ).fit()


def test_partial_correleation_output():
    m = rst.correlations.models.PartCorrelation(method='pearson',
                                 **TRIPLEWISE_CORRS_XYZ)

    r_res = r(
        """
        library(ppcor)
        SATV <-  c(500, 550, 450, 400, 600, 650, 700, 550, 650, 550)
        HSGPA <- c(3.0, 3.2, 2.8, 2.5, 3.2, 3.8, 3.9, 3.8, 3.5, 3.1)
        FGPA <-  c(2.8, 3.0, 2.8, 2.2, 3.3, 3.3, 3.5, 3.7, 3.4, 2.9)
        spcor.test(c{x}, c{y}, c{z})
        """.format(**TRIPLEWISE_CORRS_XYZ)
    )

    pd.testing.assert_frame_equal(m._results._get_r_output_df(), r_res)


def test_part_output():
    m = rst.correlations.models.PartialCorrelation(**TRIPLEWISE_CORRS_XYZ, method='pearson'
                                    )
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

    pd.testing.assert_frame_equal(m._results._get_r_output_df(), r_res, check_dtype=False)


def test_bayes_correleation():
    m = rst.correlations.models.BayesCorrelation(x='Sepal.Width', y='Sepal.Length',
                                  data=IRIS_DATA)
    r_res = r(
        """
        library(BayesFactor)
        library(tibble)

        r_res = rownames_to_column(
            data.frame(correlationBF(y = iris$Sepal.Length,
            x = iris$Sepal.Width))[, c('bf', 'error')], 'model')
        """
    )
    pd.testing.assert_frame_equal(m._results._get_r_output_df(), r_res)
