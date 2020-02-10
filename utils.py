import robusta as rst  # So we can get the PyR singleton
import pandas as pd
import numpy as np
from rpy2 import robjects
import itertools


def to_list(values):
    """Return a list of string from a list which may have lists, strings or None objects"""
    return [i for i in list(itertools.chain.from_iterable(
        itertools.repeat(x, 1) if isinstance(x, str) else x for x in values)
    ) if i != '']


def tidy(df, result_type) -> pd.DataFrame:
    if result_type == 'Anova':
        df = convert_df(_tidy_anova(df))
        # Format Degrees of freedom into separate columns
        df['df_1'], df['df_2'] = np.concatenate(df['df'].str.split(',').values).reshape((len(df), 2)).T
        # and remove the previous columns
        df.drop(columns=['df'], inplace=True)
        # Remove significane strings from the F-value column (e.g, ***)
        df['F'] = df['F'].apply(lambda s: np.round(float(s.split(' ')[0]), 2)).values
        df.rename(columns={'p.value': 'p_value'}, inplace=True)
        df[df.columns.difference(['Effect'])] = df[df.columns.difference(['Effect'])].apply(pd.to_numeric).values
        return df

    if result_type == 'TTest':
        return convert_df(_tidy_ttest(df))
    if result_type == 'BayesTTest':
        return convert_df(df)[['bf', 'error']]


def _tidy_anova(df):
    return rst.pyr.rpackages.afex.nice(df, es='pes')


def _tidy_ttest(df):
    return rst.pyr.rpackages.broom.tidy_htest(df)


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


def build_anova_formula(
                dependent,
                subject,
                between=None,
                within=None,
            ):
    ind_vars = "*".join(to_list([between, within]))
    return f'{dependent} ~ {ind_vars}'
