"""
The correlations module contains several classes for calculating correlation
coefficients.
"""
import typing
import warnings

import numpy as np
import pandas as pd

from . import results
from .. import pyr
from ..misc import base

__all__ = ['ChiSquare', 'Correlation', 'PartCorrelation',
           'PartialCorrelation', 'BayesCorrelation']

CORRELATION_METHODS = ('pearson', 'spearman', 'kendall')
DEFAULT_CORRELATION_METHOD = 'pearson'
REDUNDANT_BAYES_RESULT_COLS = ['time', 'code']
DEFAULT_CORRELATION_NULL_INTERVAL = pyr.rinterface.NULL


class _PairwiseCorrelation(base.BaseModel):
    """
    Parameters
    ----------
    x : [str, array-like]
        Input to test. Either strings with the names of columns on the
        `data` dataframe or two array-like objects. Each array-like object
        must contain only two unique values.
    y : [str, array-like]
        Input to test. Either strings with the names of columns on the
        `data` dataframe or two array-like objects. Each array-like object
        must contain only two unique values.
    data : pd.DataFrame, optional
        Dataframe of data to test, with the columns specieid in `x` and `y`.
        Each of the relevant columns must contain only two unique values.
    fit : bool, optional
        Whether to run the statistical test upon object creation. Default is True.


    Raises
    ------
    ValueError
        If x and y (strings) are not names of columns in data.
        If x and y (array-like) are not of the same length.
        # TODO - be able to handle the following:
        if x and y are not of the same type.
        if x and y are not of the same type.


    """

    def __init__(self,
                 # TODO - modify the 'x' and 'y' types to be non-mapping iterables
                 x: typing.Iterable = None,
                 y: typing.Iterable = None,
                 data: typing.Optional[pd.DataFrame] = None,
                 fit: bool = True,
                 nan_action: str = 'raise',
                 **kwargs):
        self.data = data
        self.x = x
        self.y = y
        self.nan_action = nan_action
        self._results = None
        self._fitted = False

        super().__init__()

        if fit:
            self.fit()

    def _pre_process(self):
        """
        Pre-process the input arguments prior to fitting the model.
        @return:
        """
        self._set_model_controllers()
        self._select_input_data()
        self._validate_input_data()
        self._transform_input_data()

    def _select_input_data(self):
        _data = None

        if self.data is None:
            if isinstance(self.x, str) and isinstance(self.y, str):
                raise ValueError('Specify dataframe and enter `x` and `y`'
                                 ' as strings')
            if self.x.size == 0 or self.y.size == 0:
                raise ValueError('`x` or ``y` are empty')
            if self.x.size != self.y.size:
                raise ValueError(
                    'Possibly `x` and ``y` are not of the same length')

            _data = pd.DataFrame(columns=['x', 'y'],
                                 data=np.array([self.x, self.y]).T)

        elif (isinstance(self.data, pd.DataFrame)
              and isinstance(self.x, str) and isinstance(self.y, str)):
            if {self.x, self.y}.issubset(set(self.data.columns)):
                _data = self.data[[self.x, self.y]].copy()
            else:
                raise KeyError(f"Either `x` or ({self.x}),`y` ({self.y})"
                               f" are not columns in data")

        if _data is None:  # Failed to parse data from input
            raise ValueError('Either enter `data` as a pd.DataFrame'
                             'and `x` and `y` as two column names, or enter'
                             '`x` and `y` as np.arrays')

        self._input_data = _data

    def _validate_input_data(self):
        if self._input_data.isnull().values.any():
            if self.nan_action == 'raise':
                raise ValueError('NaN in data, either specify action or '
                                 'remove')
            if self.nan_action == 'drop':
                self._input_data.dropna(inplace=True)
            if self.nan_action == 'replace':
                if self.nan_action in ('mean', 'median', 'mode'):
                    raise NotImplementedError
                    self._input_data.fillna(self._input_data.apply(nan_action),
                                            inplace=True)

    def fit(self):
        """
        Fit the statistical model.

        Returns
        -------
        None

        Raises
        ------
        RunTimeError
            If model was already fitted, raises RunTimeError.
        """
        if self._fitted is True:
            raise RuntimeError("Model was already run. Use `reset()` method"
                               " prior to calling `fit()` again!")
        self._fitted = True
        self._pre_process()
        self._analyze()

    def _analyze(self):
        raise NotImplementedError

    def _transform_input_data(self):
        pass

    def _set_model_controllers(self):
        pass

    def reset(self, refit=True, **kwargs):
        """
        Updates the Model object state and removes current test results.

        Parameters
        ----------
        refit
            Whether to fit the statistical test after resetting parameters. Default is True.
        kwargs
            Any keyword arguments of parameters to be updated.

        Returns
        -------
        None
        """

        # What else?
        vars(self).update(**kwargs)

        self._fitted = False
        self._results = None

        if refit is True:
            self.fit()


# @custom_inherit.doc_inherit(_PairwiseCorrelationModel, "numpy_with_merge")
class ChiSquare(_PairwiseCorrelation):
    """ Run a frequentist Chi-Square \u03C7\u00B2 test of independence.

    .. _Implemented R function stats::t.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/chisq.test

    Parameters
    ---------
    correction : bool
        Whether to apply Yates' correction (True) or not (False). Default is
        False.

    Warns
    ----
    RuntimeWarning
        In case the crosstabulation of the data contains cells with less
        than 5 observeations and apply_correction is set to False.
    """

    def __init__(self, apply_correction: bool = False,
                 **kwargs):
        self.apply_correction = apply_correction
        super().__init__(**kwargs)

    def _select_input_data(self):
        super(ChiSquare, self)._select_input_data()
        self._crosstab_data()

    def _validate_input_data(self):
        self._test_crosstab_frequencies()
        super()._validate_input_data()

    def _test_crosstab_frequencies(self):
        if self.crosstabulated_data.min().min() < 5 and not self.apply_correction:
            warnings.warn(
                'Less than 5 observations in some cell(s). Assumptions'
                'may be violated. Use `apply_correction=True`')

    def _crosstab_data(self):
        """Returns the crosstabulation of `x` and `y` categories."""
        self.crosstabulated_data = pd.crosstab(*self._input_data.values.T)

    def _analyze(self):
        """Runs a Chi-Square test"""
        self._results = results.ChiSquareResults(
            pyr.rpackages.stats.chisq_test(
                self.crosstabulated_data,
                correct=self.apply_correction))


# @custom_inherit.doc_inherit(_PairwiseCorrelationModel, "numpy_with_merge")
class Correlation(_PairwiseCorrelation):
    """Calculate correlation coefficient in one of several methods.

    Parameters
    ----------
    method : str
        Type of correlation coefficient. Possible values are
        'pearson', 'spearman' or 'kendall'. Default is 'pearson'.

    """

    def __init__(self, method: str = 'pearson', alternative: str = 'two.sided',
                 **kwargs):
        self.method = method
        self.tail = alternative
        super().__init__(**kwargs)

    def _validate_input_data(self):
        if self.method not in CORRELATION_METHODS:
            raise ValueError('Invalid correlation coefficient method - specify'
                             ' either `pearson`, `spearman` or `kendall`')
        super()._validate_input_data()

    def _analyze(self):
        self._results = results.CorrelationResults(
            pyr.rpackages.stats.cor_test(
                *self._input_data.values.T,
                method=self.method,
                tail=self.tail,
            ))


# @custom_inherit.doc_inherit(Correlation, "numpy_with_merge")
class _TriplewiseCorrelation(Correlation):
    """
    A base class for correlation between two variables while controlling for
    some or all of the effect of a third vriable. Used as base for
    PartCorrelation and PartialCorrelation.

    Parameters
    ----------
    z : Union[str, array-like]
        The control variable for the correlation between x and y. The Either
        name of column in data or array-like object of values.

    Raises
    ------
    ValueError
        If x, y and z (strings) are not names of columns in data.
        If x, y and z (array-like) are not of the same length.
        if x, y, and z are not of the same type.
    """

    def __init__(self, z: typing.Union[str,], **kwargs):
        self.z = z
        super().__init__(**kwargs)

    def _select_input_data(self):
        _data = None

        if self.data is None:
            if sum([isinstance(i, str) for i in [self.x, self.y, self.z]]) == 3:
                raise ValueError('Specify dataframe and enter `x`, `y` and `z`'
                                 ' as strings.')
            try:
                _data = pd.DataFrame(columns=['x', 'y', 'z'],
                                     data=np.array([self.x, self.y, self.z]).T)
            except ValueError:
                raise ValueError('`x`, `y` and `z` are not of the same length')

        elif sum([isinstance(i, str) for i in [self.x, self.y, self.z]]) == 3:
            try:
                _data = self.data[[self.x, self.y, self.z]].copy()
            except KeyError:
                raise KeyError(f"Either `x` ({self.x}),`y` ({self.y}) or `z`"
                               f" {self.z} are not columns in data")

        if _data is None:
            raise ValueError('Either enter `data` as a pd.DataFrame'
                             'and `x`, `y` and `z` as two column names, or enter'
                             '`x`, `y` and `z` as np.arrays')

        self._input_data = _data

    def _analyze(self):
        raise NotImplementedError


# @custom_inherit.doc_inherit(_TriplewiseCorrelation, "numpy_with_merge")
class PartialCorrelation(_TriplewiseCorrelation):
    """
    Calculates partial correlation.

    Part correlation is the correlation between x and y while
    the correlation between both `x and z` and `y and z` is controlled for.

    R implementation - https://www.rdocumentation.org/packages/ppcor/versions/1.1/topics/pcor.test
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze(self):
        self._results = results.PartialCorrelationResults(
            pyr.rpackages.ppcor.pcor_test(
                *self._input_data.values.T,
                method=self.method))


# @custom_inherit.doc_inherit(_TriplewiseCorrelation, "numpy_with_merge")
class PartCorrelation(_TriplewiseCorrelation):
    """
    Calculates part (Semi-partial) correlation.

    Part correlation is the correlation between x and y while the correlation
    between `y and z` is controlled for.

    R implementation - https://www.rdocumentation.org/packages/ppcor/versions/1.1/topics/spcor.test
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _analyze(self):
        self._results = results.PartCorrelationResults(
            pyr.rpackages.ppcor.spcor_test(
                *self._input_data.values.T,
                method=self.method))


# TODO - see what other
# @custom_inherit.doc_inherit(_PairwiseCorrelationModel, "numpy_with_merge")
class BayesCorrelation(_PairwiseCorrelation):
    """
    Calculates Bayes factor or returns posterior samples for correlation.

    R implementation - https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/correlationBF

    Parameters
    ----------
    rscale_prior: Union[str, float]
        Controls the scale of the prior distribution. Default value is 1.0 which
        yields a standard Cauchy prior. It is also possible to pass 'medium',
        'wide' or 'ultrawide' as input arguments instead of a float (matching
        the values of :math:`\\frac{\\sqrt{2}}{2}, 1, \\sqrt{2}`,
        respectively).
    null_interval: Array-like, optional
        Predicted interval for correlation coefficient to test against the null
        hypothesis. Optional values for a 'simple' tests are
        [-1, 1], (H1: r != 0), [-1, 0] (H1: r < 0) and
        [0, 1] (H1: ES > 0).
        Default value is [-np.inf, np.inf].
    sample_from_posterior : bool, optional
        If True return samples from the posterior, if False returns Bayes
        factor. Default is False.

    Notes
    -----

    It is better practice to confine null_interval specification to a narrower
    range than the one used by default, if you have a prior belief
    regarding the expected effect size.
    """

    def __init__(self,
                 rscale_prior: typing.Union[str, float] = 'medium',
                 null_interval: typing.Optional[typing.List[int]] = None,
                 sample_from_posterior: bool = False, **kwargs):
        self.rscale_prior = rscale_prior
        self.sample_from_posterior = sample_from_posterior

        if null_interval is None:
            self.null_interval = DEFAULT_CORRELATION_NULL_INTERVAL

        super().__init__(**kwargs)

    def _analyze(self):
        self._results = results.BayesCorrelationResults(
            pyr.rpackages.BayesFactor.correlationBF(
                *self._input_data.values.T,
                nullInterval=self.null_interval,
                rscale_prior=self.rscale_prior,
                posterior=self.sample_from_posterior,
            ))
