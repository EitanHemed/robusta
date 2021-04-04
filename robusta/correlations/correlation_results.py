"""
The correlations module contains several classes for calculating correlation
coefficients.
"""

# import custom_inherit # ?

from .. import pyr
from ..misc import utils, base

REDUNDANT_BAYES_RESULT_COLS = ['time', 'code']


class _PairwiseCorrelationResults(base.BaseResults):
    pass


# @custom_inherit.doc_inherit(_PairwiseCorrelationResults, "numpy_with_merge")
class ChiSquareResults(base.BaseResults):
    pass


class CorrelationResults(_PairwiseCorrelationResults):
    pass


# @custom_inherit.doc_inherit(CorrelationResults, "numpy_with_merge")
class _TriplewiseCorrelationResults(CorrelationResults):

    def _tidy_results(self):
        return utils.convert_df(
            self.r_results
        )


# @custom_inherit.doc_inherit(_TriplewiseCorrelationResults, "numpy_with_merge")
class PartialCorrelationResults(_TriplewiseCorrelationResults):
    pass


# @custom_inherit.doc_inherit(_TriplewiseCorrelationResults, "numpy_with_merge")
class PartCorrelationResults(_TriplewiseCorrelationResults):
    pass


# TODO - see what other
# @custom_inherit.doc_inherit(_PairwiseCorrelationResults, "numpy_with_merge")
class BayesCorrelationResults(_PairwiseCorrelationResults):

    def _tidy_results(self):
        return utils.convert_df(
            pyr.rpackages.base.data_frame(self.r_results),
            'model').drop(
            columns=REDUNDANT_BAYES_RESULT_COLS)
