from . import groupwise

__all__ = [
    "t1sample",
    "bayes_t1sample",
    "t2samples",
    "bayes_t2samples",
    "anova",
    "bayes_anova"
]


def t1sample(**kwargs):
    return groupwise.T1SampleModel(**kwargs)


def bayes_t1sample(**kwargs):
    return groupwise.BayesT1SampleModel(**kwargs)


def t2samples(**kwargs):
    return groupwise.T2SamplesModel(**kwargs)


def bayes_t2samples(**kwargs):
    return groupwise.BayesT2SamplesModel(**kwargs)


def anova(**kwargs):
    return groupwise.AnovaModel(**kwargs)


def bayes_anova(**kwargs):
    return groupwise.BayesAnovaModel(**kwargs)
