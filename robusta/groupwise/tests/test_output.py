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


