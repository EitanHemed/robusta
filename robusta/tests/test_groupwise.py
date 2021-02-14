import sys

import numpy as np
import pandas as pd
import pytest

import robusta as rst

sys.path.append('./')

TAILS = ["two.sided", "less", "greater"]


class FauxInf:
    def __init__(self, sign=''):
        self.sign = sign

    def __repr__(self):
        return f'{self.sign}Inf'


@pytest.mark.integtest
# TODO - this is messy, seperate into two tests
# @pytest.mark.parametrize('assume_equal_variance', [True, False])
@pytest.mark.parametrize('paired', [True, False])
@pytest.mark.parametrize('alternative', TAILS)
def test_t2samples(paired, alternative):
    data = rst.datasets.data('sleep')
    m = rst.t2samples(data=data,
                      independent='group',
                      dependent='extra',
                      subject='ID',
                      paired=paired,
                      tail=alternative,
                      assume_equal_variance=False
                      )
    res = m.fit()

    r_res = rst.pyrio.r(
        f"""
        library(broom)
        data.frame(tidy(t.test(
            extra~group, data=sleep, paired={'TRUE' if paired else 'FALSE'}, 
            alternative='{alternative}',
            var.equal=FALSE)
        ))
        """)
    pd.testing.assert_frame_equal(r_res, res.get_df())

    m.reset(formula='extra~group+1|ID')
    res = m.fit()

    pd.testing.assert_frame_equal(r_res, res.get_df())


@pytest.mark.integtest
@pytest.mark.parametrize('mu', [0, 2.33, 4.66])
@pytest.mark.parametrize('alternative', TAILS)
def test_t1sample(mu, alternative):
    sleep = rst.datasets.data('sleep')
    m = rst.t1sample(data=sleep.loc[sleep['group'] == '1'],
                     dependent='extra', subject='ID',
                     independent='group', mu=mu, tail=alternative)
    res = m.fit().get_df()
    r_res = rst.pyrio.r(
        f"""
        library(broom)
        data.frame(tidy(t.test(
            x=sleep[sleep$group == 1, 'extra'],
            mu={mu},
            alternative='{alternative}')))
        """
    )
    pd.testing.assert_frame_equal(res, r_res)

    m.reset(formula='extra~group+1|ID')

    res = m.fit().get_df()
    pd.testing.assert_frame_equal(res, r_res)


# TODO - the main problem remaining is that you have different column names
#  in the returned dataframe when you sample_from_posterior vs. not.

@pytest.mark.integtest
@pytest.mark.parametrize('iterations', [1e4, 1e5])
@pytest.mark.parametrize('prior_scale', [0.5, 0.707])
@pytest.mark.parametrize('null_interval',
                         [[-np.inf, 0], [-np.inf, np.inf], [0, np.inf]])
@pytest.mark.parametrize('sample_from_posterior', [False])  # , True])
def test_bayes_t2samples_independent(
        null_interval,
        prior_scale,
        sample_from_posterior,
        iterations):
    data = rst.datasets.data('mtcars')

    m = rst.bayes_t2samples(
        data=data, subject='dataset_rownames',
        dependent='mpg', independent='am',
        null_interval=null_interval,
        prior_scale=prior_scale,
        sample_from_posterior=sample_from_posterior,
        iterations=iterations, paired=False)

    res = m.fit().get_df()

    r_res = rst.pyrio.r("""   
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
               null_interval]),
        prior_scale,
        'TRUE' if sample_from_posterior else 'FALSE',
        iterations,
    ))

    # TODO - we need a better solution to compare the sampled values
    if sample_from_posterior:
        res = res.describe()
        r_res = r_res.describe()

    pd.testing.assert_frame_equal(
        res,
        r_res[res.columns.tolist()],
        # test only the columns that we are intrested in
        check_exact=(not sample_from_posterior),
        check_less_precise=5)


@pytest.mark.integtest
@pytest.mark.parametrize('iterations', [1e4, 1e5])
@pytest.mark.parametrize('prior_scale', [0.5, 0.707])
@pytest.mark.parametrize('null_interval',
                         [[-np.inf, 0], [-np.inf, np.inf], [0, np.inf]])
@pytest.mark.parametrize('sample_from_posterior', [False])  # , True])
@pytest.mark.parametrize('mu', [0, -2, 3.5])
def test_bayes_t2samples_dependent(
        iterations,
        prior_scale,
        null_interval,
        sample_from_posterior,
        mu):
    data = rst.datasets.data('sleep')

    m = rst.bayes_t2samples(
        data=data, subject='ID',
        dependent='extra', independent='group',
        null_interval=null_interval,
        prior_scale=prior_scale,
        sample_from_posterior=sample_from_posterior,
        iterations=iterations, paired=True, mu=mu)

    res = m.fit().get_df()

    r_res = rst.pyrio.r("""
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
            iterations=iterations, mu=mu, paired=T))

    if (!posterior){{
        r = data.frame(rownames_to_column(r, 'model'))
    }}
    r
    """.format(
        tuple([{-np.inf: FauxInf('-'), np.inf: FauxInf()}.get(i, 0) for i in
               null_interval]),
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
        r_res[res.columns.tolist()],
        # test only the columns that we are intrested in
        check_exact=(not sample_from_posterior),
        check_less_precise=5)


@pytest.mark.integtest
@pytest.mark.parametrize('iterations', [1e4, 1e5])
@pytest.mark.parametrize('prior_scale', [0.5, 0.707])
@pytest.mark.parametrize('null_interval',
                         [[-np.inf, 0], [-np.inf, np.inf], [0, np.inf]])
@pytest.mark.parametrize('sample_from_posterior', [False])  # , True])
@pytest.mark.parametrize('mu', [0, -2, 3.5])
def test_bayes_t1sample(
        iterations,
        prior_scale,
        null_interval,
        sample_from_posterior,
        mu):
    data = rst.datasets.data('iris')
    data = data.loc[data['Species'] == 'setosa']

    m = rst.bayes_t1sample(
        data=data, subject='dataset_rownames',
        dependent='Sepal.Width', independent='Species',
        null_interval=null_interval,
        prior_scale=prior_scale,
        sample_from_posterior=sample_from_posterior,
        iterations=iterations, mu=mu)

    res = m.fit().get_df()

    r_res = rst.pyrio.r("""
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
               null_interval]),
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
        r_res[res.columns.tolist()],
        # test only the columns that we are intrested in
        check_exact=(not sample_from_posterior),
        check_less_precise=5)


@pytest.mark.parametrize('between_vars', [
    ['supp', "'supp'"],
    [['supp', 'dose'], "c('supp', 'dose')"],
    [['dose'], "c('dose')"]
])
def test_anova_between(between_vars):
    m = rst.anova(
        data=rst.datasets.data('ToothGrowth'),
        dependent='len', subject='dataset_rownames',
        between=between_vars[0])

    r_res = rst.pyrio.r(
        f"""
        library(broom)
        library(afex)
        library(tibble)
        data.frame(
            tidy(
                anova(
                    aov_ez(
                    data=rownames_to_column(ToothGrowth, 'dataset_rownames'),
                    id='dataset_rownames', between={between_vars[1]},
                            dv='len', es='pes')
                    )
                )
        )
        """)

    pd.testing.assert_frame_equal(m.fit().get_df(), r_res)

    m.reset(
        formula=f"len ~ {'*'.join(rst.utils.to_list(between_vars[0]))} + 1|dataset_rownames")

    # m = rst.anova(
    #    data=rst.datasets.data('ToothGrowth'),
    #    formula=f"len ~ {'*'.join(rst.utils.to_list(between_vars[0]))} + 1|dataset_rownames")

    pd.testing.assert_frame_equal(m.fit().get_df(), r_res)


def test_anova_within():
    df = rst.datasets.data('anxiety').set_index(
        ['id', 'group']).filter(
        regex='^t[1-3]$').stack().reset_index().rename(
        columns={0: 'score',
                 'level_2': 'time'})
    m = rst.anova(data=df, within='time',
                  dependent='score', subject='id')
    r_res = rst.pyrio.r(
        """
        library(afex)
        library(broom)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)

        data.frame(tidy(anova(aov_4(score ~ (time|id), data=data_long,
                ))))
        """
    )

    pd.testing.assert_frame_equal(m.fit().get_df(), r_res)

    m.reset(formula='score~time|id')
    pd.testing.assert_frame_equal(m.fit().get_df(), r_res)


def test_anova_mixed():
    df = rst.datasets.data('anxiety').set_index(
        ['id', 'group']).filter(
        regex='^t[1-3]$').stack().reset_index().rename(
        columns={0: 'score',
                 'level_2': 'time'})
    m = rst.anova(data=df, within='time', between='group',
                  dependent='score', subject='id')

    r_res = rst.pyrio.r(
        """
        library(afex)
        library(broom)
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)

        data.frame(tidy(anova(aov_4(score~(time|id) + group,
            data=data_long))))
        """
    )

    pd.testing.assert_frame_equal(m.fit().get_df(), r_res)

    m.reset(formula='group + score~time|id')
    pd.testing.assert_frame_equal(m.fit().get_df(), r_res)



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
    m = rst.bayes_anova(
        data=rst.datasets.data('ToothGrowth'),
        dependent='len', subject='dataset_rownames',
        between=between_vars[0], iterations=1e4,
        include_subject=include_subject)

    formula = between_vars[1]
    if include_subject:
        formula += ' + dataset_rownames'

    r_res = rst.pyrio.r(
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
        """)

    # TODO - as we would upgrade the project to a more recent pandas version we
    #  can provide better testing of the dataframe's values, now it would
    #   be too cumbersome as some values may be too large/small to simply use
    #   `check_less_exact' (e.g., see below)

    # E[left]: [1.198756782290705, 4983636409073.187, 287711229722084.75,
    #           776685585093109.9]
    # E[right]: [1.198756782290705, 4983636409073.187, 285885307062654.3,
    #            784100477177213.6]

    pd.testing.assert_frame_equal(
        m.fit().get_df().head(2), r_res.head(2))

    m.reset(
        formula=f"{between_vars[1]} + (1|dataset_rownames)",
        include_subject=include_subject)

    pd.testing.assert_frame_equal(m.fit().get_df().head(2), r_res.head(2))

# class TestAnova(unittest.TestCase):

#     def test_margins(self):
#         df = rst.datasets.data('anxiety').set_index(
#             ['id', 'group']).filter(
#             regex='^t[1-3]$').stack().reset_index().rename(columns={0: 'score',
#                                                                     'level_2': 'time'})
#         anova_time = rst.Anova(data=df, within='time',
#                                dependent='score', subject='id')
#         margins_time = anova_time.get_margins('time')
#
#         anova_time_by_group = rst.Anova(
#             data=df, between='group', within='time',
#             dependent='score', subject='id')
#         margins_time_by_group = anova_time_by_group.get_margins(
#             ('time', 'group'))
#
#         # TODO - do this in one call to r (i.e., return a vector
#         #  of two dataframes).
#         r_res_time = rst.pyrio.r(
#             """
#             library(afex)
#             library(emmeans)
#             library(tidyr)
#             library(datarium)
#             data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)
#
#             r_res_time = data.frame(emmeans(aov_ez(data=data_long,
#                     within='time', dv='score', id='id'), 'time', type='response'))
#             """
#         )
#         r_res_time_by_group = rst.pyrio.r(
#             """
#             r_res_time_by_group = data.frame(emmeans(aov_ez(data=data_long, between='group',
#                 within='time', id='id', dv='score'), c('time', 'group'),
#                 type='response'))
#             """
#         )
#
#         pd.testing.assert_frame_equal(
#             margins_time, r_res_time)
#         pd.testing.assert_frame_equal(
#             margins_time_by_group, r_res_time_by_group)
#
#
# class TestBayesAnova(unittest.TestCase):
#
#     def test_bayes_anova(self):
#         # TODO write a better test case - as Bayes factors can get obscenely
#         #  large or small rounding errors can fail the test.
#         #  also the fact that
#
#         anova = rst.BayesAnova(data=rst.datasets.data('ToothGrowth'),
#                                dependent='len', subject='dataset_rownames',
#                                between=['supp', 'dose'], iterations=100000
#                                ).get_results()  # .round(4)
#         r_res = rst.pyrio.r(
#             """
#             ToothGrowth$dose = as.factor(ToothGrowth$dose)
#             rownames_to_column(
#                 data.frame(
#                     anovaBF(len ~ supp*dose, data=ToothGrowth),
#                     iterations=10000)[ , c('bf', 'error')], 'model')
#             """
#         )
#
#         # TODO - currently there is very partial testing of the values magnitude
#         #    as they are either very large or very small.
#
#         # The partial testing is because of the great/small magnitude of
#         # Bayes factors and their error terms.
#         pd.testing.assert_frame_equal(anova.iloc[:2], r_res.iloc[:2])
#
#
# class TestWilcoxon1Sample(unittest.TestCase):
#
#     def test_wilcox1sample(self):
#         x = (38.9, 61.2, 73.3, 21.8, 63.4, 64.6, 48.4, 48.8, 48.5)
#         y = (67.8, 60, 63.4, 76, 89.4, 73.3, 67.3, 61.3, 62.4)
#         weight_diff = np.array(x) - np.array(y)
#         group = np.repeat(0, len(weight_diff))
#
#         df = pd.DataFrame(data=np.array([weight_diff, group]).T,
#                           columns=['weight', 'group']).reset_index()
#         res = rst.Wilcoxon1Sample(data=df, independent='group',
#                                   subject='index',
#                                   dependent='weight', mu=-10).get_results()
#
#         r_res = rst.pyrio.r(f"""
#         # Example from http://www.sthda.com/english/wiki/unpaired-two-samples-wilcoxon-test-in-r
#         library(broom)
#         x <- c(38.9, 61.2, 73.3, 21.8, 63.4, 64.6, 48.4, 48.8, 48.5)
#         y <- c(67.8, 60, 63.4, 76, 89.4, 73.3, 67.3, 61.3, 62.4)
#         weight_diff <- x - y
#         data.frame(tidy(wilcox.test(weight_diff,
#             exact = TRUE, correct=TRUE, mu=-10)))
#         """)
#
#         pd.testing.assert_frame_equal(
#             res, r_res)
#
# class TestWilcoxon2Sample(unittest.TestCase):
#
#     def test_wilcox2sample_dependent(self):
#         before = (200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2,
#                     193.7)
#         after = (392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3,
#                    352.2)
#         weight = np.concatenate([before, after])
#         group = np.repeat([0, 1], 10)
#         id = np.repeat(range(len(before)), 2)
#
#         df = pd.DataFrame(data=np.array([weight, group, id]).T,
#                           columns=['weight', 'group', 'id'])
#
#         res = rst.Wilcoxon2Sample(data=df, independent='group', paired=True,
#                                   dependent='weight', subject='id').get_results()
#
#         r_res = rst.pyrio.r(f"""
#         # Example from http://www.sthda.com/english/wiki/paired-samples-wilcoxon-test-in-r
#         library(broom)
#         before = c(200.1, 190.9, 192.7, 213, 241.4, 196.9, 172.2, 185.5, 205.2,
#                     193.7)
#         after = c(392.9, 393.2, 345.1, 393, 434, 427.9, 422, 383.9, 392.3,
#                    352.2)
#         # Create a data frame
#         my_data <- data.frame(
#                 group = as.factor(rep(c("before", "after"), each = 10)),
#                 weight = c(before,  after)
#                 )
#         data.frame(tidy(
#             wilcox.test(before, after, paired = TRUE,
#             exact=TRUE, correct=TRUE)))
#         """)
#         pd.testing.assert_frame_equal(
#             res, r_res)
#
# class TestKruskalWallisTest(unittest.TestCase):
#
#     def test_kruskalwallistest(self):
#
#         res = rst.KruskalWallisTest(
#             data=rst.datasets.data('PlantGrowth'),
#             between='group', paired=True,
#             dependent='weight', subject='dataset_rownames').get_results()
#         r_res = rst.pyrio.r("""
#         # Example from http://www.sthda.com/english/wiki/kruskal-wallis-test-in-r
#         library(broom)
#         data.frame(tidy(kruskal.test(weight ~ group, data = PlantGrowth)))
#         """)
#         pd.testing.assert_frame_equal(
#             res, r_res)
#
# class FriedmanTest(unittest.TestCase):
#
#     def test_friedman_test(self):
#
#         with self.assertRaises(rst.pyr.rinterface.RRuntimeError):
#             # Until we get rstatix on the environment
#
#             r_res = rst.pyrio.r("""
#             # Example from https://www.datanovia.com/en/lessons/friedman-test-in-r/
#             library(rstatix)
#             library(broom)
#             library(datarium)
#             library(tidyr)
#             data_long <- gather(selfesteem, 'time', 'score', t1:t3, factor_key=TRUE)
#             data.frame(tidy(friedman_test(score ~ time |id, data=data_long)))
#             """)
#
#             df = rst.datasets.data('selfesteem').set_index(
#                 ['id', 'group']).filter(
#                 regex='^t[1-3]$').stack().reset_index().rename(
#                 columns={0: 'score', 'level_2': 'time'})
#             res = rst.FriedmanTest(data=df, within='time', dependent='score',
#                                    subject='id')
#
#             pd.testing.assert_frame_equal(res, r_res)
#
# if __name__ == '__main__':
#     unittest.main()
