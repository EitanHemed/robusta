import re
import robusta as rst  # So we can get the PyR singleton
import pandas as pd
import numpy as np
from rpy2 import robjects
import itertools
from collections.abc import Iterable


def to_list(values):
    """Return a list of string from a list which may have lists, strings or None objects"""
    return np.hstack([values])


def tidy(df, result_type) -> pd.DataFrame:
    if result_type == 'Anova':
        df = convert_df(_tidy_anova(df))
        # Format Degrees of freedom into separate columns

        return df

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
    return rst.pyr.rpackages.broom.tidy_anova(df)


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


def build_lm4_style_formula(
        dependent,
        subject,
        between=None,
        within=None,
):
    between = "*".join(to_list(between))
    within = "|".join(to_list(within))
    if within:
        within = f'({within}|{subject})'
    else:
        within = f'(1|{subject})'
    frml = f'{dependent} ~ {between} + {within}'
    return frml, rst.pyr.rpackages.stats.formula(frml)


def parse_variables_from_lm4_style_formula(frml):
    # TODO - support much more flexible parsing of the between/within/interactions, as currently we can't
    #   seperate excluded interactions etc. This is good for ANOVA, but is very limiting for actual linear mixed models.
    #   The current implementation assumes the following form: 'y ~ b1*b2 + (w1|w2|id)'
    #   Generally, everything here could benefit from a regex implementation.

    dependent, frml = frml.split('~')  # to remove the dependent variable from the formula
    dependent = dependent[:-1]  # trim the trailing whitespace
    frml = frml[1:]  # Trim the leading whitespace
    between, frml = frml.split('+')
    between = between[:-1].split('*')  # Drop the trailing whitespace and split to separate between subject variables
    *within, subject = re.findall(r'\((.*?)\)', frml)[0].split('|')
    if within == ['1']:
        within = []

    return dependent, between, within, subject

# r2 = aov_ez(dv='len', within=c('v', 't'), id='id', between=c('dose', 'supp'), data=df3)
# r = aov_4(len ~ dose*supp + (v|t|id), data=df3)
