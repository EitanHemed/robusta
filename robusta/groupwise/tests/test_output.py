import numpy as np
import pandas as pd
import pytest
import robusta as rst

from collections import namedtuple

MTCARS = rst.load_dataset('mtcars')
PLANTS = rst.load_dataset('PlantGrowth')
SELFESTEEM = rst.load_dataset('selfesteem').set_index(
    ['id']).filter(
    regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_1': 'time'})
WEIGHTLOSS = rst.load_dataset('weightloss').set_index(
    ['id', 'diet', 'exercises']).filter(
    regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_3': 'time'}
)


PERFORMANCE = rst.load_dataset('performance').set_index(
    ['id', 'gender', 'stress']).filter(
    regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_3': 'time'}
)

def test_t2samples_between():
    pass


def test_t1sample():
    pass


def test_bayes_t2samples():
    pass


def test_bayes_t1sample():
    pass


def test_wilcoxon_2samples():
    pass


def test_wilcoxon_1sample():
    pass


def test_oneway_between_anova_output():
    m = rst.groupwise.Anova(data=PLANTS, dependent='weight', between='group', subject='dataset_rownames')
    m.fit()
    assert m.report_text() == 'group [F(2, 27) = 4.85, p = 0.016, Partial Eta-Sq. = 0.26]'


def test_oneway_within_anova_output():
    m = rst.groupwise.Anova(data=SELFESTEEM, dependent='score', within='time', subject='id')
    m.fit()
    assert m.report_text() == 'time [F(1, 12) = 55.47, p = 0.001, Partial Eta-Sq. = 0.86]'


def test_twoway_between_anova_output():
    m = rst.groupwise.Anova(dependent='mpg', between=['vs', 'am'], subject='dataset_rownames', data=MTCARS)
    m.fit()
    assert m.report_text() == (
        'vs [F(1, 28) = 31.73, p = 0.001, Partial Eta-Sq. = 0.53]. '
        'am [F(1, 28) = 23.54, p = 0.001, Partial Eta-Sq. = 0.46]. '
        'vs:am [F(1, 28) = 1.33, p = 0.259, Partial Eta-Sq. = 0.04]')


def test_threeway_within_anova_output():
    m = rst.groupwise.Anova(formula='score~diet|exercises|time|id', data=WEIGHTLOSS)
    m.fit()
    assert m.report_text() == (
        'diet [F(1, 11) = 6.02, p = 0.032, Partial Eta-Sq. = 0.35]. '
        'exercises [F(1, 11) = 58.93, p = 0.001, Partial Eta-Sq. = 0.84]. '
        'time [F(2, 22) = 110.94, p = 0.001, Partial Eta-Sq. = 0.91]. '
        'diet:exercises [F(1, 11) = 75.36, p = 0.001, Partial Eta-Sq. = 0.87]. '
        'diet:time [F(1, 15) = 0.60, p = 0.501, Partial Eta-Sq. = 0.05]. '
        'exercises:time [F(2, 17) = 20.83, p = 0.001, Partial Eta-Sq. = 0.65]. '
        'diet:exercises:time [F(2, 21) = 14.25, p = 0.001, Partial Eta-Sq. = 0.56]')

def test_threeway_mixed_anova_output():
    m = rst.groupwise.Anova(formula='score~gender*stress+(time|id)', data=PERFORMANCE)
    m.fit()
    assert m.report_text() == (
        'gender [F(1, 54) = 2.41, p = 0.127, Partial Eta-Sq. = 0.04]. '
        'stress [F(2, 54) = 21.17, p = 0.001, Partial Eta-Sq. = 0.44]. '
        'gender:stress [F(2, 54) = 1.55, p = 0.221, Partial Eta-Sq. = 0.05]. '
        'time [F(1, 54) = 0.06, p = 0.803, Partial Eta-Sq. = 0.00]. '
        'gender:time [F(1, 54) = 4.73, p = 0.034, Partial Eta-Sq. = 0.08]. '
        'stress:time [F(2, 54) = 1.82, p = 0.172, Partial Eta-Sq. = 0.06]. '
        'gender:stress:time [F(2, 54) = 6.10, p = 0.004, Partial Eta-Sq. = 0.18]'
    )