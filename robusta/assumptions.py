import robusta as rst  # So we can get the PyR singleton


# TODO add NORMALITY test of assumptions
# TODO add Skewness TEST
# TODO add kurtosis test

def normality(a):
    """Run a Shapiro-Wilk test of normality.

    Parameters
    ----------

    Returns
    -------

    """
    return rst.utils.convert_df(
        rst.pyr.rpackages.broom.tidy_htest(
            rst.pyr.rpackages.stats.shapiro_test(a)))
