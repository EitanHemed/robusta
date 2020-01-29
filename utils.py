import robusta  # So we can get the PyR singleton
import pandas as pd
import numpy as np
from rpy2 import robjects


def tidy(df, result_type) -> pd.core.frame.DataFrame:
    if result_type == 'Anova':
        return convert_df(_tidy_anova(df))

def _tidy_anova(df):
    return robusta.pyr.rpackages.afex.nice(df, es='pes')

def _tidy_ttest(df):
    return convert_df(robusta.pyr.afex.nice())

def verify_float(a):
    try:
        np.array(a).astype(float)
    except ValueError:
        print("data type can't be coerced to float")


def verify_levels(a: object, max_levels: object = None) -> None:
    cur_levels = len(pd.Series(a).unique())
    if cur_levels < 2:
        print('Not a variable! number of levels should be >= 2')
    if max_levels is not None and cur_levels > max_levels:
        print('Number of levels should be not be more'
              'than {}, but it is currently {}'.format(max_levels, cur_levels))


def convert_df(df):
    """A utility for safe conversion
    @type df: An R or Python DataFrame.
    """
    if type(df) == pd.core.frame.DataFrame:
        return robjects.pandas2ri.py2ri(df)
    elif type(df) == robjects.vectors.DataFrame:
        return pd.DataFrame(robjects.pandas2ri.ri2py(df))
    else:
        print("Input can only be R/Python DataFrame object")
