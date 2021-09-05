import numpy as np
import pandas as pd
import pytest
from rpy2.robjects import r

import robusta as rst

from robusta.groupwise.results import BF_COLUMNS
# TODO - add tests against rst.groupwise.models.TEST_TAIL_DICT.keys ('x<y', etc.)
from robusta.groupwise.models import (R_BAYES_TEST_TAILS_DICT, R_FREQUENTIST_TEST_TAILS_SPECS)

SLEEP_DATASET = rst.load_dataset('sleep')
MTCARS_DATASET = rst.load_dataset('mtcars')
IRIS_DATASET = rst.load_dataset('iris')
TOOTH_GROWTH_DATASET = rst.load_dataset('ToothGrowth')
PLANT_GROWTH_DATASET = rst.load_dataset('PlantGrowth')
ANXIETY_DATASET = rst.load_dataset('anxiety').set_index(
    ['id', 'group']).filter(
    regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_2': 'time'})
SELF_ESTEEM_DATASET = rst.load_dataset('selfesteem').set_index(
    'id').filter(
    regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score', 'level_2': 'time'})

class FauxInf:
    """The sole purpose of this class is to have an object that it's __repr__
    return Inf (or -Inf) when you format the R string (in case you need an
    Inf value when specifying prior distribution)."""
    def __init__(self, sign: str = ''):
        if sign not in ['-', '']:
            raise ValueError
        self.sign = sign

    def __repr__(self):
        return f'{self.sign}Inf'


# TODO - this is messy, separate into two tests
# @pytest.mark.parametrize('assume_equal_variance', [True, False])
@pytest.mark.parametrize('paired', [True, False])
@pytest.mark.parametrize('tail', R_FREQUENTIST_TEST_TAILS_SPECS.items())
def test_t2samples(paired, tail):
    m = rst.groupwise.T2Samples(data=SLEEP_DATASET,
                                independent='group',
                                dependent='extra',
                                subject='ID',
                                paired=paired,
                                tail=tail[0],
                                assume_equal_variance=False
                                )
    

    r_res = rst.misc.utils.convert_df(r(
        f"""
        library(broom)
        data.frame(tidy(t.test(
            extra~group, data=sleep, paired={'TRUE' if paired else 'FALSE'}, 
            alternative='{tail[1]}',
            var.equal=FALSE)
        ))
        """))
    pd.testing.assert_frame_equal(r_res, m._results._get_r_output_df())

    m.reset(formula='extra~group+1|ID')
    

    pd.testing.assert_frame_equal(r_res, m._results._get_r_output_df())


@pytest.mark.parametrize('mu', [0, 2.33, 4.66])
@pytest.mark.parametrize('tail', R_FREQUENTIST_TEST_TAILS_SPECS.items())
def test_t1sample(mu, tail):
    m = rst.groupwise.T1Sample(data=SLEEP_DATASET.loc[SLEEP_DATASET['group'] == '1'],
                               dependent='extra', subject='ID',
                               independent='group', mu=mu, tail=tail[0])
    
    res = m._results._get_r_output_df()
    r_res = rst.misc.utils.convert_df(r(
        f"""
        library(broom)
        data.frame(tidy(t.test(
            x=sleep[sleep$group == 1, 'extra'],
            mu={mu},
            alternative='{tail[1]}')))
        """
    ))
    pd.testing.assert_frame_equal(res, r_res)

    m.reset(formula='extra~group+1|ID')
    
    res = m._results._get_r_output_df()
    pd.testing.assert_frame_equal(res, r_res)


# TODO - the main problem remaining is that you have different column names
#  in the returned dataframe when you sample_from_posterior vs. not.

@pytest.mark.integtest
@pytest.mark.parametrize('iterations', [1e4, 1e5])
@pytest.mark.parametrize('prior_scale', [0.5, 0.707])
@pytest.mark.parametrize('null_interval', R_BAYES_TEST_TAILS_DICT.items())
# [[-np.inf, 0], [-np.inf, np.inf], [0, np.inf]])
@pytest.mark.parametrize('sample_from_posterior', [False])  # , True])
def test_bayes_t2samples_independent(
        null_interval,
        prior_scale,
        sample_from_posterior,
        iterations):

    m = rst.groupwise.BayesT2Samples(
        data=MTCARS_DATASET, subject='dataset_rownames',
        dependent='mpg', independent='am',
        tail=null_interval[0],
        prior_scale=prior_scale,
        sample_from_posterior=sample_from_posterior,
        iterations=iterations, paired=False)
    
    res = m._results._get_r_output_df()

    r_res = rst.misc.utils.convert_df(r("""   
    library(BayesFactor)
    library(tibble)
    
    # test data
    dat = split.data.frame(mtcars, mtcars$am)
    x = dat[[1]]$mpg
    y = dat[[2]]$mpg
    
    # test parameters
    
    nullInterval = c{}
    rscale = {}
    posterior = {}
    iterations = {}

    
    # additional_col = ifelse(posterior, 'iteration', 'model')
    # return_cols = ifelse(posterior, 
    #     c('mu', 'beta..x...y', 'sig2', 'delta', 'g'), 
    #     c('model', 'bf', 'error'))
    
    r = data.frame(
        ttestBF(x=x, y=y, nullInterval=nullInterval,
            rscale=rscale,
            posterior=posterior,
            iterations=iterations))
            
    if (!posterior){{
        r = data.frame(rownames_to_column(r, 'model'))
    }}
    r
    """.format(
        tuple([{-np.inf: FauxInf('-'), np.inf: FauxInf()}.get(i, 0) for i in
               null_interval[1]]),
        prior_scale,
        'TRUE' if sample_from_posterior else 'FALSE',
        iterations,
    )))

    # TODO - we need a better solution to compare the sampled values
    if sample_from_posterior:
        res = res.describe()
        r_res = r_res.describe()

    pd.testing.assert_frame_equal(
        res,
        r_res[res.columns.tolist()].reset_index(drop=True),
        # test only the columns that we are intrested in
        check_exact=(not sample_from_posterior),
        check_less_precise=5)


@pytest.mark.parametrize('iterations', [1e4, 1e5])
@pytest.mark.parametrize('prior_scale', [0.5, 0.707])
@pytest.mark.parametrize('null_interval', R_BAYES_TEST_TAILS_DICT.items())
# [[-np.inf, 0], [-np.inf, np.inf], [0, np.inf]])
@pytest.mark.parametrize('sample_from_posterior', [False])  # , True])
@pytest.mark.parametrize('mu', [0, -2, 3.5])
def test_bayes_t2samples_dependent(
        iterations,
        prior_scale,
        null_interval,
        sample_from_posterior,
        mu):
    m = rst.groupwise.BayesT2Samples(
        data=SLEEP_DATASET, subject='ID',
        dependent='extra', independent='group',
        tail=null_interval[0],
        prior_scale=prior_scale,
        sample_from_posterior=sample_from_posterior,
        iterations=iterations, paired=True, mu=mu)
    
    res = m._results._get_r_output_df()

    r_res = r("""
    library(BayesFactor)
    library(tibble)

    # test data
    dat = split.data.frame(sleep, sleep$group)
    x = dat[[1]]$extra
    y = dat[[2]]$extra

    # test parameters

    nullInterval = c{}
    rscale = {}
    posterior = {}
    iterations = {}
    mu = {}

    # additional_col = ifelse(posterior, 'iteration', 'model')
    # return_cols = ifelse(posterior,
    #     c('mu', 'beta..x...y', 'sig2', 'delta', 'g'),
    #     c('model', 'bf', 'error'))

    r = data.frame(
        ttestBF(x=x, y=y, nullInterval=nullInterval,
            rscale=rscale,
            posterior=posterior,
            iterations=iterations, mu=mu, paired=TRUE))

    if (!posterior){{
        r = data.frame(rownames_to_column(r, 'model'))
    }}
    r
    """.format(
        tuple([{-np.inf: FauxInf('-'), np.inf: FauxInf()}.get(i, 0) for i in
               null_interval[1]]),
        prior_scale,
        'TRUE' if sample_from_posterior else 'FALSE',
        iterations,
        mu
    ))

    # TODO - we need a better solution to compare the sampled values
    if sample_from_posterior:
        res = res.describe()
        r_res = r_res.describe()
        pd.testing.assert_frame_equal(
            res,
            rst.misc.utils.convert_df(r_res),
            check_exact=False,
            check_less_precise=5)

    else:
        pd.testing.assert_frame_equal(
            res.reset_index(drop=True),
            rst.misc.utils.convert_df(r_res, 'model')[BF_COLUMNS].reset_index(drop=True),
            check_exact=True,
            check_less_precise=5.)


@pytest.mark.parametrize('iterations', [1e4, 1e5])
@pytest.mark.parametrize('prior_scale', [0.5, 0.707])
@pytest.mark.parametrize('null_interval', R_BAYES_TEST_TAILS_DICT.items())
# [[-np.inf, 0], [-np.inf, np.inf], [0, np.inf]])
@pytest.mark.parametrize('sample_from_posterior', [False])  # , True])
@pytest.mark.parametrize('mu', [0, -2, 3.5])
def test_bayes_t1sample(
        iterations,
        prior_scale,
        null_interval,
        sample_from_posterior,
        mu):
    data = IRIS_DATASET.loc[IRIS_DATASET['Species'] == 'setosa']

    m = rst.groupwise.BayesT1Sample(
        data=data, subject='dataset_rownames',
        dependent='Sepal.Width', independent='Species',
        tail=null_interval[0],
        prior_scale=prior_scale,
        sample_from_posterior=sample_from_posterior,
        iterations=iterations, mu=mu)
    
    res = m._results._get_r_output_df()

    r_res = r("""
    library(BayesFactor)
    library(tibble)

    # test data
    data = iris
    data = data[data$Species == 'setosa', ]

    # test parameters
    nullInterval = c{}
    rscale = {}
    posterior = {}
    iterations = {}
    mu = {}

    r = data.frame(
        ttestBF(x=data$Sepal.Width,
            nullInterval=nullInterval,
            rscale=rscale,
            posterior=posterior,
            iterations=iterations, mu=mu))

    if (!posterior){{
        r = data.frame(rownames_to_column(r, 'model'))
    }}
    r
    """.format(
        tuple([{-np.inf: FauxInf('-'), np.inf: FauxInf()}.get(i, 0) for i in
               null_interval[1]]),
        prior_scale,
        'TRUE' if sample_from_posterior else 'FALSE',
        iterations,
        mu
    ))

    # TODO - we need a better solution to compare the sampled values
    if sample_from_posterior:
        res = res.describe()
        r_res = r_res.describe()
        pd.testing.assert_frame_equal(
            res,
            rst.misc.utils.convert_df(r_res),
            check_exact=False,
            check_less_precise=5)

    else:
        pd.testing.assert_frame_equal(
            res.reset_index(drop=True),
            rst.misc.utils.convert_df(r_res, 'model')[BF_COLUMNS].reset_index(drop=True),
            # test only the columns that we are intrested in
            check_exact=(not sample_from_posterior),
            check_less_precise=5)


@pytest.mark.parametrize('between_vars', [
    ['supp', "supp"],
    [['supp', 'dose'], "supp*dose"],
    [['dose'], "dose"]
])
def test_anova_between(between_vars):
    m = rst.groupwise.Anova(
        data=TOOTH_GROWTH_DATASET,
        dependent='len', subject='dataset_rownames',
        between=between_vars[0])

    r_res = rst.misc.utils.convert_df(r(
        f"""
        library(broom)
        library(afex)
        library(tibble)
        rownames_to_column(ToothGrowth, 'dataset_rownames')
    
        data.frame(
            nice(aov_4(data=rownames_to_column(ToothGrowth, 'dataset_rownames'), formula=len ~ {between_vars[1]} 
            + (1|dataset_rownames)), es='pes'))
        """))

    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, r_res)

    m.reset(
        formula=f"len ~ {between_vars[1]} + (1|dataset_rownames)")
    
    res = m._results._get_r_output_df()

    # m = rst.anova(
    #    data=rst.datasets.data('ToothGrowth'),
    #    formula=f"len ~ {'*'.join(rst.utils.to_list(between_vars[0]))} + 1|dataset_rownames")

    pd.testing.assert_frame_equal(res, r_res)


def test_anova_within():
    m = rst.groupwise.Anova(data=ANXIETY_DATASET, within='time',
                            dependent='score', subject='id')
    
    res = m._results._get_r_output_df()

    r_res = rst.misc.utils.convert_df(r(
        """
        library(afex)
        library(broom)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)

        data.frame(nice(aov_4(score ~ (time|id), data=data_long), es='pes'))
        """
    ))

    pd.testing.assert_frame_equal(res, r_res)

    m.reset(formula='score~time|id')
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, r_res)


def test_anova_mixed():
    # Initialize r objects to be tested
    r(
        """
        library(emmeans)
        library(afex)
        library(broom)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)

        a1 = aov_4(score~(time|id) + group, data=data_long)
        anova_table = data.frame(nice(a1, es='pes'))
        interaction_margins = data.frame(emmeans(a1, specs=c('group', 'time'),
         type='response'))
        time_margins = data.frame(emmeans(a1, specs=c('time'),
         type='response'))
        group_margins = data.frame(emmeans(a1, specs=c('group'),
         type='response'))
        """
    )

    m = rst.groupwise.Anova(data=ANXIETY_DATASET, within='time', between='group', dependent='score', subject='id')
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, rst.misc.utils.convert_df(r('anova_table')))

    m.reset(formula='score ~ group + (time|id)')
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, rst.misc.utils.convert_df(r('anova_table')))

    pd.testing.assert_frame_equal(
        m.get_margins(['group', 'time']),
        rst.misc.utils.convert_df(r('interaction_margins'))
    )
    pd.testing.assert_frame_equal(
        m.get_margins(margins_terms='group', by_terms='time'),
        rst.misc.utils.convert_df(r('interaction_margins'))
    )
    pd.testing.assert_frame_equal(
        m.get_margins(margins_terms='time'),
        rst.misc.utils.convert_df(r('time_margins'))
    )
    pd.testing.assert_frame_equal(
        m.get_margins(margins_terms='group'),
        rst.misc.utils.convert_df(r('group_margins'))
    )


# TODO - we need to fix the R code for testing the model with and without the
#  the subject term
@pytest.mark.parametrize('between_vars',
                         [['supp', "len ~ supp"],
                          [['supp', 'dose'], "len ~ supp*dose"],
                          [['dose'], "len ~ dose"]]
                         )
@pytest.mark.parametrize('include_subject',
                         [False]  # , True]
                         )
def test_bayes_anova_between(between_vars, include_subject):
    m = rst.groupwise.BayesAnova(
        data=TOOTH_GROWTH_DATASET,
        dependent='len', subject='dataset_rownames',
        between=between_vars[0], iterations=1e4,
        include_subject=include_subject)
    
    res = m._results._get_r_output_df()

    formula = between_vars[1]
    if include_subject:
        formula += ' + dataset_rownames'

    r_res = rst.misc.utils.convert_df(r(
        f"""
        library(BayesFactor)
        library(tibble)

        data = rownames_to_column(ToothGrowth, 'dataset_rownames')
        cols = c('dose', 'dataset_rownames')
        data[cols] <- lapply(data[cols], factor)  
        
        if (grepl('dataset_rownames', {formula}, fixed = TRUE)){{
            whichRandom = NULL
        }} else {{
            whichRandom = 'dataset_rownames'
        }}       
        

        data.frame(
            rownames_to_column(
                    data.frame(
                        anovaBF(formula={formula}, data=data,
                        whichRandom=whichRandom,
                        iterations=1e4))[ , c('bf', 'error')],
            'model'
            )
        )
        """))

    # TODO - as we would upgrade the project to a more recent pandas version we
    #  can provide better testing of the dataframe's values, now it would
    #   be too cumbersome as some values may be too large/small to simply use
    #   `check_less_exact' (e.g., see below)

    # E[left]: [1.198756782290705, 4983636409073.187, 287711229722084.75,
    #           776685585093109.9]
    # E[right]: [1.198756782290705, 4983636409073.187, 285885307062654.3,
    #            784100477177213.6]

    pd.testing.assert_frame_equal(
        res.head(2), r_res.head(2).reset_index(drop=True))

    m.reset(
        formula=f"{between_vars[1]} + (1|dataset_rownames)",
        include_subject=include_subject)
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res.head(2), r_res.head(2).reset_index(drop=True))


@pytest.mark.parametrize('within_vars',
                         [['time', "score ~ time"]]
                         )
@pytest.mark.parametrize('include_subject', [False])
def test_bayes_anova_within(within_vars, include_subject):
    m = rst.groupwise.BayesAnova(data=ANXIETY_DATASET, within='time',
                                 dependent='score', subject='id')
    
    res = m._results._get_r_output_df()

    formula = within_vars[1]
    if include_subject:
        formula += ' + id'

    r_res = rst.misc.utils.convert_df(r(
        f"""
        library(BayesFactor)
        library(tibble)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)

        if (grepl('id', {formula}, fixed = TRUE)){{
            whichRandom = NULL
        }} else {{
            whichRandom = 'id'
        }}       

        data.frame(
            rownames_to_column(
            data.frame(anovaBF({formula}, data=data_long,
                iterations=1e4,
                ))[ , c('bf', 'error')],
            'model'
            )
        )
        """))

    # TODO - as we would upgrade the project to a more recent pandas version we
    #  can provide better testing of the dataframe's values, now it would
    #   be too cumbersome as some values may be too large/small to simply use
    #   `check_less_exact' (e.g., see below)

    # E[left]: [1.198756782290705, 4983636409073.187, 287711229722084.75,
    #           776685585093109.9]
    # E[right]: [1.198756782290705, 4983636409073.187, 285885307062654.3,
    #            784100477177213.6]

    pd.testing.assert_frame_equal(
        res, r_res.reset_index(drop=True))

    m.reset(
        formula=f"{within_vars[1]} + (1|id)",
        include_subject=include_subject)
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, r_res.reset_index(drop=True))


@pytest.mark.parametrize('include_subject', [False])
def test_bayes_anova_mixed(include_subject):
    m = rst.groupwise.BayesAnova(data=ANXIETY_DATASET, within='time', between='group',
                                 dependent='score', subject='id')
    
    res = m._results._get_r_output_df()

    formula = "score ~ group + time | id"

    r_res = rst.misc.utils.convert_df(r(
        f"""
        library(BayesFactor)
        library(tibble)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)

        if (grepl('id', {formula}, fixed = TRUE)){{
            whichRandom = NULL
        }} else {{
            whichRandom = 'id'
        }}       

        data.frame(
            rownames_to_column(
            data.frame(anovaBF(score ~ group + time, data=data_long,
                iterations=1e4,
                ))[ , c('bf', 'error')],
            'model'
            )
        )
        """))

    # TODO - as we would upgrade the project to a more recent pandas version we
    #  can provide better testing of the dataframe's values, now it would
    #   be too cumbersome as some values may be too large/small to simply use
    #   `check_less_exact' (e.g., see below)

    pd.testing.assert_frame_equal(res.head(2), r_res.head(2).reset_index(drop=True))

    m.reset(
        formula=formula,
        include_subject=include_subject)
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res.head(2), r_res.head(2).reset_index(drop=True))


@pytest.mark.parametrize('p_exact', [False, True])
@pytest.mark.parametrize('p_correction', [False, True])
@pytest.mark.parametrize('mu', [-10, -3])
@pytest.mark.parametrize('tail', R_FREQUENTIST_TEST_TAILS_SPECS.items())
def test_wilcoxon_1sample(p_exact, p_correction, mu, tail):
    x = (38.9, 61.2, 73.3, 21.8, 63.4, 64.6, 48.4, 48.8, 48.5)
    y = (67.8, 60, 63.4, 76, 89.4, 73.3, 67.3, 61.3, 62.4)
    weight_diff = np.array(x) - np.array(y)
    group = np.repeat(0, len(weight_diff))
    df = pd.DataFrame(data=np.array([weight_diff, group]).T,
                      columns=['weight', 'group']).reset_index()
    m = rst.groupwise.Wilcoxon1Sample(data=df, independent='group',
                                      subject='index',
                                      dependent='weight', mu=mu, tail=tail[0],
                                      p_exact=p_exact, p_correction=p_correction)
    
    res = m._results._get_r_output_df()

    r_res = rst.misc.utils.convert_df(r(
        f"""
        # Example from http://www.sthda.com/english/wiki/unpaired-two-samples-wilcoxon-test-in-r
        library(broom)
        x <- c(38.9, 61.2, 73.3, 21.8, 63.4, 64.6, 48.4, 48.8, 48.5)
        y <- c(67.8, 60, 63.4, 76, 89.4, 73.3, 67.3, 61.3, 62.4)
        weight_diff <- x - y
        data.frame(tidy(wilcox.test(weight_diff,
            exact={str(p_exact)[0]}, 
            correct={str(p_correction)[0]}, mu={mu}, alternative='{tail[1]}')))
        """))

    pd.testing.assert_frame_equal(res, r_res)

    m.reset(formula='weight ~ group + 1|index')
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, r_res)


@pytest.mark.parametrize('p_exact', [False, True])
@pytest.mark.parametrize('p_correction', [False, True])
@pytest.mark.parametrize('tail', R_FREQUENTIST_TEST_TAILS_SPECS.items())
@pytest.mark.parametrize('paired', [True, False])
def test_wilcoxon_2samples(p_exact, p_correction, tail, paired):
    before = (200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2,
              193.7)
    after = (392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3,
             352.2)
    weight = np.concatenate([before, after])
    group = np.repeat([0, 1], 10)
    sid = np.repeat(range(len(before)), 2)

    df = pd.DataFrame(data=np.array([weight, group, sid]).T,
                      columns=['weight', 'group', 'sid'])

    m = rst.groupwise.Wilcoxon2Samples(
        data=df, independent='group', paired=paired,
        dependent='weight', subject='sid',
        tail=tail[0], p_correction=p_correction, p_exact=p_exact
    )
    
    res = m._results._get_r_output_df()

    r_res = r(f"""
        # Example from http://www.sthda.com/english/wiki/paired-samples-wilcoxon-test-in-r
        library(broom)
        before = c(200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2,
                    193.7)
        after = c(392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3,
                   352.2)
        # Create a data frame
        my_data <- data.frame(
                group = as.factor(rep(c("before", "after"), each = 10)),
                weight = c(before,  after)
                )
        data.frame(tidy(
            wilcox.test(before, after, paired = {str(paired)[0]},
            alternative='{tail[1]}',
            exact={str(p_exact)[0]}, correct={str(p_correction)[0]})))
        """)

    pd.testing.assert_frame_equal(res, rst.misc.utils.convert_df(r_res))

    m.reset(formula='weight ~ group|sid')
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(res, rst.misc.utils.convert_df(r_res))


def test_kruskal_wallis_test():
    r_res = rst.misc.utils.convert_df(r("""
    # Example from http://www.sthda.com/english/wiki/kruskal-wallis-test-in-r
    library(broom)
    data.frame(tidy(kruskal.test(weight ~ group, data = PlantGrowth)))
    """))
    m = rst.groupwise.KruskalWallisTest(
        data=PLANT_GROWTH_DATASET,
        between='group',
        dependent='weight', subject='dataset_rownames')
    
    res = m._results._get_r_output_df()
    pd.testing.assert_frame_equal(res, r_res)

    m.reset(formula='weight~group+1|dataset_rownames')
    
    res = m._results._get_r_output_df()
    pd.testing.assert_frame_equal(res, r_res)


def test_friedman_test():
    with pytest.raises(rst.pyr.rinterface.embedded.RRuntimeError):
        # Until we get rstatix on the environment
        r_res = rst.misc.utils.convert_df(r("""
        # Example from https://www.datanovia.com/en/lessons/friedman-test-in-r/
        library(rstatix)
        library(broom)
        library(datarium)
        library(tidyr)
        data_long <- gather(selfesteem, 'time', 'score', t1:t3, factor_key=TRUE)
        data.frame(tidy(friedman_test(score ~ time |id, data=data_long)))
        """))

        m = rst.groupwise.FriedmanTest(data=SELF_ESTEEM_DATASET, within='time', dependent='score',
                                       subject='id')
        
        res = m._results._get_r_output_df()
        pd.testing.assert_frame_equal(res, r_res)

        m.reset(formula='score ~ time|id')
        
        res = m._results._get_r_output_df()
        pd.testing.assert_frame_equal(res, r_res)


def test_aligned_ranks_test():
    # initialize r

    r("""
        library(ARTool)
        data(Higgins1990Table5)
        res = anova(
            art(DryMatter ~ Moisture*Fertilizer + (1|Tray), 
            data=Higgins1990Table5))
        """)
    m = rst.groupwise.AlignedRanksTest(
        data=rst.load_dataset('Higgins1990Table5'),
        formula='DryMatter ~ Moisture*Fertilizer + (1|Tray)')
    
    res = m._results._get_r_output_df()

    pd.testing.assert_frame_equal(
        res, rst.misc.utils.convert_df(r("res"))
    )

    m.reset(between=['Moisture', 'Fertilizer'], dependent='DryMatter',
            subject='Tray')
    
    res = m._results._get_r_output_df()
    pd.testing.assert_frame_equal(
        res, rst.misc.utils.convert_df(r("res"))
    )
