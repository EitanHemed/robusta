import robusta as rst
import inspect


def list_examples():
    pass


def anova_between():
    """Output of a between-subject anova, using the mtcars dataset"""
    mtcars = rst.datasets.load('mtcars')
    mtcars['model'] = range(len(mtcars))
    anova = rst.Anova(data=mtcars, between=['gear', 'vs'],
                      subject='model', dependent='qsec')
    return anova


def ttest_independent():
    """Output of an independent samples t-test, using the ToothGrowth"""
    tooth = rst.datasets.load('ToothGrowth')
    anova = rst.Anova(data=mtcars, between=['gear', 'vs'],
                      subject='model', dependent='qsec')
    return anova
