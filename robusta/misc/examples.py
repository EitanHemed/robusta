import robusta as rst


def list_examples():
    pass


def anova_between():
    """Output of a between-subject anova, using the mtcars dataset"""
    mtcars = robusta.misc.datasets.load('mtcars')
    mtcars['model'] = range(len(mtcars))
    anova_freq = rst.AnovaBS(data=mtcars,
        between=['gear', 'vs'], subject='model', dependent='qsec')
    anova_bayes = rst.BayesAnovaBS(data=mtcars,
                             between=['gear', 'vs'], subject='model',
                             dependent='qsec')
    return anova_freq, anova_bayes


def ttest_independent():
    """Output of an independent samples t-test, using the ToothGrowth"""
    data = robusta.misc.datasets.load('ToothGrowth')
    ttest_freq = rst.T2IndSamples(
        data=data, independent='supp', subject='row_names', dependent='len')
    ttest_bayes = rst.BayesT2IndSamples(
        data=data, independent='supp', subject='row_names', dependent='len')
    return ttest_freq, ttest_bayes
