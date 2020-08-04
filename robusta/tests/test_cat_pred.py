import unittest
import sys

import numpy as np
import pandas as pd

sys.path.append('./')

import robusta as rst


class TestT1Sample(unittest.TestCase):

    def test_t1sample_get_results(self):
        sleep = rst.datasets.data('sleep')
        res = rst.T1Sample(data=sleep.loc[sleep['group'] == '1'],
                           dependent='extra', subject='ID',
                           independent='group').get_results()
        """# Now in R...
        library(broom)
        data.frame(tidy(t.test(x=sleep[sleep$group == 1, 'extra'])))
        """
        r_res = pd.read_csv(pd.compat.StringIO("""
        estimate statistic p.value parameter conf.low conf.high method alternative
    0.75 1.32571 0.2175978  9 -0.5297804  2.02978 "One Sample t-test" two.sided"""
                                               ), delim_whitespace=True,
                            dtype={'parameter': 'float64'})
        pd.testing.assert_frame_equal(res, r_res, check_exact=False,
                                      check_less_precise=5)


class TestBayesT1Sample(unittest.TestCase):

    def test_bayes1sample(self):
        sleep = rst.datasets.data('sleep')
        data = sleep.loc[sleep['group'] == '1'].assign(
            diff_scores=sleep.groupby('ID')['extra'].apply(
                lambda s: s.diff(-1).max()).values)
        res = rst.BayesT1Sample(data=data,
                                dependent='diff_scores', subject='ID',
                                independent='group',
                                null_interval=[-np.inf, 0]).get_results()
        """
        library(BayesFactor)
        diff = (sleep[sleep$group == 1, 'extra'] 
            - sleep[sleep$group == 2, 'extra'])
        res = data.frame(
            ttestBF(
                diff, nullInterval=c(-Inf, 0)))[, c('bf', 'er')]
                                          bf        error
        Alt., r=0.707 -Inf<d<0    34.4169360 2.549779e-07
        Alt., r=0.707 !(-Inf<d<0)  0.1008246 6.210317e-04
        """
        r_res = pd.DataFrame(columns=['bf', 'error'],
                             data=np.array([[34.416936, 2.549779e-07],
                                            [0.100825, 6.210317e-04]]))
        pd.testing.assert_frame_equal(
            res[['bf', 'error']],
            r_res
        )


class TestAnova(unittest.TestCase):

    def test_between_anova(self):
        anova = rst.Anova(data=rst.datasets.data('ToothGrowth'),
                          dependent='len', subject='dataset_rownames',
                          between=['supp', 'dose']).get_results()
        """
        library(broom)
        library(afex)
        library(tibble)
        data.frame(tidy(anova(aov_ez(rownames_to_column(ToothGrowth, 'dataset_rownames'),
        id='dataset_rownames', between=c('supp', 'dose'),
        dv='len', es='pes'))))
        """
        r_res = pd.read_csv(pd.compat.StringIO(
        """
               term num.Df den.Df      MSE statistic       ges      p.value
                  supp      1     54 13.18715 15.571979 0.2238254 2.311828e-04
                  dose      2     54 13.18715 91.999965 0.7731092 4.046291e-18
             supp:dose      2     54 13.18715  4.106991 0.1320279 2.186027e-02"""),
        delim_whitespace=True, dtype={'num.Df': 'float64', 'den.Df': 'float64'}
        )

        pd.testing.assert_frame_equal(anova, r_res, check_exact=False,
                                      check_less_precise=5)

    def test_margins(self):
        df = rst.datasets.data('anxiety').set_index(
            ['id', 'group']).stack().reset_index().rename(columns={0: 'score',
                                                                   'level_2': 'time'})
        anova_time = rst.Anova(data=df, within='time',
                               dependent='score', subject='id')
        margins_time = anova_time.get_margins('time')

        anova_time_by_group = rst.Anova(
            data=df, between='group', within='time',
            dependent='score', subject='id')
        margins_time_by_group = anova_time_by_group.get_margins(
            ('time', 'group'))
        """
        # Now in R...
        library(tidyr)
        library(datarium)
        data_long <- gather(anxiety, 'time', 'score', t1:t3, factor_key=TRUE)
        margins_time = data.frame(emmeans(aov_ez(data=data_long, 
                within='time', dv='score', id='id'), 'time', type='response'))

        margins_time_by_group = data.frame(emmeans(aov_ez(data=data_long, between='group', 
            within='time', id='id', dv='score'), c('time', 'group'), 
            type='response'))
        """
        r_res_time = pd.read_csv(pd.compat.StringIO("""
               time   emmean        SE       df lower.CL upper.CL
            t1 16.91556 0.2612364 55.02618 16.39203 17.43908
            t2 16.13556 0.2612364 55.02618 15.61203 16.65908
            t3 15.19778 0.2612364 55.02618 14.67425 15.72130"""),
                                 delim_whitespace=True,
                                 dtype={'time': 'category'})
        r_res_time_by_group = pd.read_csv(pd.compat.StringIO("""
        time group   emmean        SE       df lower.CL upper.CL
               t1  grp1 17.08667 0.4025229 43.99952 16.27543 17.89790
               t2  grp1 16.92667 0.4025229 43.99952 16.11543 17.73790
               t3  grp1 16.50667 0.4025229 43.99952 15.69543 17.31790
               t1  grp2 16.64667 0.4025229 43.99952 15.83543 17.45790
               t2  grp2 16.46667 0.4025229 43.99952 15.65543 17.27790
               t3  grp2 15.52667 0.4025229 43.99952 14.71543 16.33790
               t1  grp3 17.01333 0.4025229 43.99952 16.20210 17.82457
               t2  grp3 15.01333 0.4025229 43.99952 14.20210 15.82457
               t3  grp3 13.56000 0.4025229 43.99952 12.74877 14.37123"""),
                                          delim_whitespace=True,
                                          dtype={'time': 'category',
                                                 'group': 'category'})
        pd.testing.assert_frame_equal(
            margins_time, r_res_time, check_exact=False, check_less_precise=5)
        pd.testing.assert_frame_equal(
            margins_time_by_group, r_res_time_by_group, check_exact=False,
            check_less_precise=5)


class TestBayesAnova(unittest.TestCase):

    def test_bayes_anova(self):
        # TODO write a better test case - as Bayes factors can get obscenely
        #  large or small rounding errors can fail the test.
        #  also the fact that

        anova = rst.BayesAnova(data=rst.datasets.data('ToothGrowth'),
                               dependent='len', subject='dataset_rownames',
                               between=['supp', 'dose'], iterations=100000
                               ).get_results()  # .round(4)
        """#Now in R...
        ToothGrowth$dose = as.factor(ToothGrowth$dose)
        rownames_to_column(
            data.frame(
                anovaBF(len ~ supp*dose, data=ToothGrowth),
                iterations=10000)[ , c('bf', 'error')], 'model')
        """
        r_res = pd.read_csv(pd.compat.StringIO("""
                                model           bf        error
                                supp 1.198757e+00 8.941079e-05
                                dose 4.983636e+12 1.189630e-08
                         "supp + dose" 2.823349e+14 1.579653e-02
             "supp + dose + supp:dose" 7.830497e+14 1.781969e-02"""),
                            delim_whitespace=True)
        # The partial testing is because of the great/small magnitude of
        # Bayes factors and their error terms.
        pd.testing.assert_frame_equal(anova.iloc[:2], r_res.iloc[:2])


if __name__ == '__main__':
    unittest.main()
