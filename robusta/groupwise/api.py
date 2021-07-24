from .models import *

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
    return T1Sample(**kwargs)


def bayes_t1sample(**kwargs):
    return BayesT1Sample(**kwargs)


def t2samples(**kwargs):
    return T2Samples(**kwargs)


def bayes_t2samples(**kwargs):
    return BayesT2Samples(**kwargs)


def anova(**kwargs):
    return Anova(**kwargs)


def bayes_anova(**kwargs):
    return BayesAnova(**kwargs)


def wilcoxon_1sample(**kwargs):
    return Wilcoxon1Sample(**kwargs)


def wilcoxon_2samples(**kwargs):
    return Wilcoxon2Samples(**kwargs)


def kruskal_wallis_test(**kwargs):
    return KruskalWallisTest(**kwargs)


def friedman_test(**kwargs):
    return FriedmanTest(**kwargs)


def aligned_ranks_test(**kwargs):
    return AlignedRanksTest(**kwargs)