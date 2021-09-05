import numpy as np
import pandas as pd
import pytest
import robusta as rst

from robusta.groupwise.models import R_FREQUENTIST_TEST_TAILS_SPECS

from collections import namedtuple

SLEEP = rst.load_dataset('sleep')
MTCARS = rst.load_dataset('mtcars')
PLANTS = rst.load_dataset('PlantGrowth')
CHICK_WEIGHT = rst.load_dataset('chickwts')

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
MICE2 = rst.load_dataset('mice2').set_index('id')[['before', 'after']].stack().reset_index().rename(
    columns={'level_1': 'time', 0: 'weight'})

PERFORMANCE = rst.load_dataset('performance').set_index(
    ['id', 'gender', 'stress']).filter(
    regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_3': 'time'}
)


@pytest.mark.parametrize('tail_specs', R_FREQUENTIST_TEST_TAILS_SPECS.items())
def test_tail_specification(tail_specs):
    m = rst.groupwise.T2Samples(data=MICE2, formula='weight~time|id', tail=tail_specs[0])
    
    output = m.report_text()
    m.reset(tail=tail_specs[1])
    
    assert output == m.report_text()


def test_t2samples_paired_output():
    m = rst.groupwise.T2Samples(data=MICE2, formula='weight~time|id')
    
    assert m.report_text() == 't(9) = 25.55, p < 0.001'

    x, y = MICE2.groupby('time')['weight'].apply(lambda s: s.values)
    m = rst.groupwise.T2Samples(x=x, y=y, tail='x > y', paired=True)
    
    assert m.report_text() == 't(9) = 25.55, p < 0.001'


def test_t2samples_unpaired_output():
    m = rst.groupwise.T2Samples(data=MTCARS, formula='wt~am+1|dataset_rownames', tail='x > y')
    
    assert m.report_text() == "t(29) = 5.49, p < 0.001"

    x, y = MTCARS.groupby('am')['wt'].apply(lambda s: s.values)
    m = rst.groupwise.T2Samples(x=x, y=y, tail='x > y', paired=False)
    
    assert m.report_text() == "t(29) = 5.49, p < 0.001"


def test_t1sample_output():
    m = rst.groupwise.T1Sample(data=MTCARS, formula='wt~am+1|dataset_rownames', tail='x < y', mu=3.5)
    
    assert m.report_text() == "t(31) = -1.63, p = 0.056"

    m = rst.groupwise.T1Sample(x=MTCARS['wt'].values, tail='x < y', mu=3.5)
    
    assert m.report_text() == "t(31) = -1.63, p = 0.056"


def test_bayes_t2samples_output():
    _data = CHICK_WEIGHT.assign(feed=CHICK_WEIGHT['feed'].astype(str).values).loc[
        CHICK_WEIGHT['feed'].isin(["horsebean", "linseed"])]
    m = rst.groupwise.BayesT2Samples(data=_data, formula='weight~feed+1|dataset_rownames', paired=False)
    
    assert m.report_text() == 'Alt., r=0.707 [BF1:0 = 5.98, Error = 0.001%]'

    x, y = _data.groupby(['feed'])['weight'].apply(lambda s: s.values)
    m = rst.groupwise.BayesT2Samples(x=x, y=y, paired=False)
    
    assert m.report_text() == 'Alt., r=0.707 [BF1:0 = 5.98, Error = 0.001%]'


def test_bayes_t1sample_output():
    m = rst.groupwise.BayesT1Sample(
        data=MTCARS.assign(wt=2.5 - MTCARS['wt'].values), formula='wt~am+1|dataset_rownames',
        tail='x < y')  # , tail=[-np.Inf, 0])
    
    assert m.report_text() == ('Alt., r=0.707 -Inf<d<0 [BF1:0 = 230.25, Error = 0.001%]. '
                               'Alt., r=0.707 !(-Inf<d<0) [BF1:0 = 0.04, Error = 0.001%]')

    m = rst.groupwise.BayesT1Sample(
        x=(2.5 - MTCARS['wt'].values),
        tail='x < y')  # , tail=[-np.Inf, 0])
    
    assert m.report_text() == ('Alt., r=0.707 -Inf<d<0 [BF1:0 = 230.25, Error = 0.001%]. '
                               'Alt., r=0.707 !(-Inf<d<0) [BF1:0 = 0.04, Error = 0.001%]')


def test_wilcoxon_2samples_output():
    x = np.array([0.80, 0.83, 1.89, 1.04, 1.45, 1.38, 1.91, 1.64, 0.73, 1.46])
    y = np.array([1.15, 0.88, 0.90, 0.74, 1.21])
    group = np.concatenate([np.zeros(x.size), np.ones(y.size)])
    df = pd.DataFrame(data=np.array([np.concatenate([x, y]), group]).T,
                      columns=['score', 'group']).reset_index()

    m = rst.groupwise.Wilcoxon2Samples(formula='score~group + 1|index', tail="x > y", data=df, mu=0.1)
    
    assert m.report_text() == 'Z = 35.00, p = 0.127'

    m = rst.groupwise.Wilcoxon2Samples(x=x, y=y, mu=0.1, paired=False, tail="x > y")
    
    assert m.report_text() == 'Z = 35.00, p = 0.127'


def test_wilcoxon_1sample_output():
    x = (1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
    y = (0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)
    weight_diff = np.array(x) - np.array(y)
    group = np.repeat(0, len(weight_diff))
    df = pd.DataFrame(data=np.array([weight_diff, group]).T,
                      columns=['score', 'group']).reset_index()

    m = rst.groupwise.Wilcoxon1Sample(formula='score~(group|index)', tail="x > y", data=df, mu=0)
    
    assert m.report_text() == 'Z = 40.00, p = 0.020'

    m = rst.groupwise.Wilcoxon1Sample(x=weight_diff, mu=0, tail="x > y")
    
    assert m.report_text() == 'Z = 40.00, p = 0.020'


def test_oneway_between_anova_output():
    m = rst.groupwise.Anova(data=PLANTS, dependent='weight', between='group', subject='dataset_rownames')
    
    assert m.report_text() == 'group [F(2, 27) = 4.85, p = 0.016, Partial Eta-Sq. = 0.26]'


def test_oneway_within_anova_output():
    m = rst.groupwise.Anova(data=SELFESTEEM, dependent='score', within='time', subject='id')
    
    assert m.report_text() == 'time [F(1, 12) = 55.47, p = 0.001, Partial Eta-Sq. = 0.86]'


def test_twoway_between_anova_output():
    m = rst.groupwise.Anova(dependent='mpg', between=['vs', 'am'], subject='dataset_rownames', data=MTCARS)
    
    assert m.report_text() == (
        'vs [F(1, 28) = 31.73, p = 0.001, Partial Eta-Sq. = 0.53]. '
        'am [F(1, 28) = 23.54, p = 0.001, Partial Eta-Sq. = 0.46]. '
        'vs:am [F(1, 28) = 1.33, p = 0.259, Partial Eta-Sq. = 0.04]')


def test_threeway_within_anova_output():
    m = rst.groupwise.Anova(formula='score~diet|exercises|time|id', data=WEIGHTLOSS)
    
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
    
    assert m.report_text() == (
        'gender [F(1, 54) = 2.41, p = 0.127, Partial Eta-Sq. = 0.04]. '
        'stress [F(2, 54) = 21.17, p = 0.001, Partial Eta-Sq. = 0.44]. '
        'gender:stress [F(2, 54) = 1.55, p = 0.221, Partial Eta-Sq. = 0.05]. '
        'time [F(1, 54) = 0.06, p = 0.803, Partial Eta-Sq. = 0.00]. '
        'gender:time [F(1, 54) = 4.73, p = 0.034, Partial Eta-Sq. = 0.08]. '
        'stress:time [F(2, 54) = 1.82, p = 0.172, Partial Eta-Sq. = 0.06]. '
        'gender:stress:time [F(2, 54) = 6.10, p = 0.004, Partial Eta-Sq. = 0.18]'
    )


def test_oneway_between_bayes_anova_output():
    m = rst.groupwise.BayesAnova(formula='weight~feed+1|dataset_rownames', data=CHICK_WEIGHT)
    
    assert m.report_text() == 'feed [BF1:0 = 1.41E+07, Error = 0.001]'


def test_mixed_bayes_anova_output():
    m = rst.groupwise.BayesAnova(formula='score~gender*stress+(time|id)', data=PERFORMANCE,
                                        which_models='bottom')
    
    assert m.report_text() == (
        'gender [BF1:0 = 4.14E-01, Error = 0.001]. stress [BF1:0 = 2.09E+05, Error = 0.001]. '
        'time [BF1:0 = 1.98E-01, Error = 0.001]. gender:stress [BF1:0 = 2.97E-01, Error = 0.001]. '
        'gender:time [BF1:0 = 7.89E-01, Error = 0.001]. stress:time [BF1:0 = 3.20E-01, Error = 0.001]. '
        'gender:stress:time [BF1:0 = 3.27E+00, Error = 0.001]')
