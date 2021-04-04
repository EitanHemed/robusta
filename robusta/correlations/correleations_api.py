from .correlations_models import *

__all__ = [
    'chisquare',
    'correlation',
    'bayes_correlation',
    'part_correlation',
    'partial_correlation',
]


def chisquare(**kwargs):
    return ChiSquareModel(**kwargs)


def correlation(**kwargs):
    return CorrelationModel(**kwargs)


def bayes_correlation(**kwargs):
    return BayesCorrelationModel(**kwargs)


def part_correlation(**kwargs):
    return PartCorrelationModel(**kwargs)


def partial_correlation(**kwargs):
    return PartialCorrelationModel(**kwargs)
