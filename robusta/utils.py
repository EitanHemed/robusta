import re
from itertools import chain

import numpy as np
import pandas as pd
import typing
from rpy2 import robjects

import robusta as rst  # So we can get the PyR singleton


# TODO - change this to return a list of strings instead of a generator
def to_list(values):
    """Return a list of string from a list which may have lists, strings or
     None objects"""
    return list(
        chain.from_iterable(
            item if isinstance(
                item, (list, tuple)) and not isinstance(
                item, str) else [
                item] for item in values))


def tidy(df, result_type) -> pd.DataFrame:
    # TODO - fix this to be more specific. Convert all of the checks to regex.
    if result_type == 'BayesAnova':
        return df[['model', 'bf', 'error']]
    if result_type == 'Anova':
        return convert_df(_tidy_anova(df))
    if result_type in ['BayesT2Samples',
                       'BayesT1Sample']:
        return convert_df(df)[['bf', 'error']]
    if result_type in ['T2Samples', 'T1Sample']:
        return convert_df(_tidy_ttest(df))


def _tidy_anova(df):
    return rst.pyr.rpackages.broom.tidy_anova(df)


def _tidy_ttest(df):
    return rst.pyr.rpackages.broom.tidy_htest(df)


def verify_float(a):
    try:
        np.array(a).astype(float)
    except ValueError:
        print("data type can't be coerced to float")


def verify_levels(df: pd.DataFrame, min_levels: int = 2,
                  max_levels: object = None) -> None:

    levels = df.apply(
        lambda s: s.unique().size)

    if (levels < min_levels).any():
        non_varying_levels = levels[levels < min_levels].index.values
        composed = '\n'.join([f'- Variable {variable} - has {num} levels'
                    for variable, num in
                    zip(
                        non_varying_levels,
                        levels[non_varying_levels].values)])
        raise RuntimeError("Non-varying independent variables encountered.\n"
                           "Make sure that each independent variable has\n"
                           f"at least {min_levels} levels. Invalid variables: \n"
                           f"{composed}")

    if max_levels is not None and (levels > max_levels).any():
        too_many_levels_variables = levels[levels > max_levels].index.values
        composed = "\n".join([f'- Variable {variable} - has {num} levels'
                    for variable, num in
                    zip(
                        too_many_levels_variables,
                        levels[too_many_levels_variables].values)])
        raise RuntimeError("Variables with more levels than allowed encountered.\n"
                           "Make sure that each independent variable has\n"
                           f"at most {max_levels} levels. Invalid variables:\n"
                           f"{composed}")


def convert_df(df, rownames_to_column_name: typing.Union[None, str]=None):
    """A utility for safe conversion
    @type df: An R or Python DataFrame.
    """
    if type(df) == pd.core.frame.DataFrame:
        _cat_cols = df.dtypes.reset_index()
        _cat_cols = _cat_cols.loc[
            _cat_cols[0] == 'category', 'index'].values.tolist()
        _df = robjects.pandas2ri.py2ri(df)
        for cn in filter(lambda s: s in _cat_cols, _df.names):
            _df[_df.names.index(cn)] = rst.pyr.rpackages.base.factor(
                _df[_df.names.index(cn)])
        return _df
    elif type(df) == robjects.vectors.DataFrame:
        if rownames_to_column_name is None:
            return pd.DataFrame(robjects.pandas2ri.ri2py(df))
        else:
            if not isinstance(rownames_to_column_name, str):
                raise ValueError("Column name of previous row names must be a str")
            df = rst.pyr.rpackages.tibble.rownames_to_column(
                    df, var=rownames_to_column_name)
            df = pd.DataFrame(robjects.pandas2ri.ri2py(df))
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
            return df
    else:
        raise RuntimeError("Input can only be R/Python DataFrame object")


def bayes_style_formula(dependent, between, within, subject=None):
    independent = "*".join(to_list([between, within]))
    if subject is None:
        return f'{dependent} ~ {independent}'
    else:
        return f'{dependent} ~ {independent} + {subject}'


def build_general_formula(dependent, independent):
    independent = "*".join(to_list(independent))
    return f'{dependent} ~ {independent}'


def parse_variables_from_general_formula(frml):
    dependent, frml = frml.split(
        '~')  # to remove the dependent variable from the formula
    if dependent[-1] == ' ':
        dependent = dependent[:-1]  # trim the trailing whitespace
    if frml[0] == ' ':
        frml = frml[1:]  # Trim the leading whitespace
    independent, frml = frml.split('+')
    independent = list(filter(independent[:-1].split('*'),
                              ''))  # Drop the trailing whitespace and split to separate between subject variables
    subject = re.findall(r'\((.*?)\)', frml)[0].split('|')[
        1]  # We expect something along the lines of (1|ID)

    return dependent, independent, subject


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
    #   This also works - `res = rst.Anova(data=rst.datasets.load('sleep'), formula='extra ~ +(group|ID)')`
    #   The following will fail miserably - 'y~b1*b2+(w1|id)'
    #   Generally, everything here could benefit from a regex implementation.
    #   Also, we want variable names to not include spaces, but only underscores.
    dependent, frml = frml.split(
        '~')  # to remove the dependent variable from the formula
    dependent = dependent[:-1]  # trim the trailing whitespace
    frml = frml[1:]  # Trim the leading whitespace
    between, frml = frml.split('+')
    between = between[:-1].split(
        '*')  # Drop the trailing whitespace and split to separate between subject variables
    *within, subject = re.findall(r'\((.*?)\)', frml)[0].split('|')
    if within == ['1']:
        within = []
    return dependent, between, within, subject

