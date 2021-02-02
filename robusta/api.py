from . import groupwise

__all__ = [
    "t1sample",
    "t1sample_bayes",
    "t2samples",
    "t2samples_bayes"
]


def t1sample(**kwargs):
    return groupwise.T1SampleModel(**kwargs)


def t1sample_bayes(**kwargs):
    return groupwise.BayesT1SampleModel(**kwargs)


def t2samples(**kwargs):
    return groupwise.T2SamplesModel(**kwargs)


def t2samples_bayes(**kwargs):
    return groupwise.BayesT2SamplesModel(**kwargs)
