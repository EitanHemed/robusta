import re
from itertools import chain
from collections.abc import Iterable
import numpy as np
import pandas as pd
import typing
import rpy2.robjects as ro

import robusta as rst  # So we can get the PyR singleton


# TODO - change this to return a list of strings instead of a generator
def to_list(values):
    # TODO - refactor this. e.g., what happens on a dict?

    """Return a list of string from a list which may have lists, strings or
     None objects"""
    if isinstance(values, str) or not isinstance(values, Iterable):
        if values == '' or values is None:
            return []
        return [values]
    return list(
        chain.from_iterable(
            item if isinstance(
                item, (list, tuple, set, np.ndarray)) and not isinstance(
                item, str) else [
                item] for item in values))

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


def convert_df(df, rownames_to_column_name: typing.Union[None, str] = None):
    """A utility for safe conversion
    @type df: An R or Python DataFrame.
    """

    if type(df) == pd.core.frame.DataFrame:
        with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
            _df = ro.conversion.py2rpy(df.copy())
            return _df
    elif type(df) == np.recarray:
        # Yes, currently a recursion until we refactor this function,
        # Send it as python dataframe into convert_df to get back an R dataframe with the column names, then
        # it will be returned as a python dataframe with the rownames as an adiditional column
        _df = pd.DataFrame(convert_df(convert_df(pd.DataFrame(df), rownames_to_column_name=rownames_to_column_name)))
        return _df
    else:
        with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
            # Make sure it is coerced into data frame
            _df = rst.pyr.rpackages.base.data_frame(df)
            if rownames_to_column_name is not None:
                _df = _df.rename_axis(rownames_to_column_name).reset_index()
            return _df

def bayes_style_formula_from_vars(dependent, between, within, subject=None):
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
