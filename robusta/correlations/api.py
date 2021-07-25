from .models import *

__all__ = [
    'chisquare',
    'correlation',
    'bayes_correlation',
    'part_correlation',
    'partial_correlation',
]


def chisquare(**kwargs):
    return ChiSquare(**kwargs)


def correlation(**kwargs):
    return Correlation(**kwargs)


def bayes_correlation(**kwargs):
    return BayesCorrelation(**kwargs)


def part_correlation(**kwargs):
    return PartCorrelation(**kwargs)


def partial_correlation(**kwargs):
    return PartialCorrelation(**kwargs)
