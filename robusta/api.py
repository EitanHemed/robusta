from . import groupwise

__all__ = [
    "t1sample",
    "bayes_t1sample",
    "t2samples",
    "bayes_t2samples",
    "anova",
    "bayes_anova",
    "wilcoxon_1sample",
    "wilcoxon_2samples",
    "kruskal_wallis_test",
    "friedman_test",
    "aligned_ranks_test"
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

def wilcoxon_1sample(**kwargs):
    return groupwise.Wilcoxon1SampleModel(**kwargs)

def wilcoxon_2samples(**kwargs):
    return groupwise.Wilcoxon2SamplesModel(**kwargs)

def kruskal_wallis_test(**kwargs):
    return groupwise.KruskalWallisTestModel(**kwargs)

def friedman_test(**kwargs):
    return groupwise.FriedmanTestModel(**kwargs)

def aligned_ranks_test(**kwargs):
    return groupwise.AlignedRanksTestModel(**kwargs)