"""
ttest_and_anova contains many classes capable of running common t-test and
analysis of variance. Each analysis is accompanied by a matching Bayesian
analysis.

- Anova, BayesAnovaBS: Between/Within/Mixed n-way analysis of variance.
- T2Samples, BayesT2Samples: Independent/Dependent samples t-test.
- T1Sample, BayesT1Sample: One-Sample t-test.

The class instances have the following methods:
- get_df(): Returns a pandas dataframe with the results of the analysis.
- get_text_report(): Returns a textual report of the analysis.
- get_latex_report(): Returns a latex formatted report of the analysis.
"""

import typing
import textwrap
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pandas_flavor import register_dataframe_accessor
import robusta as rst

__all__ = [
    "Anova", "BayesAnova",
    "T1Sample", "T2Samples",
    "BayesT1Sample", "BayesT2Samples"
]


@dataclass
class _BaseParametric:
    """
    A basic class to handle pre-requisites of T-Tests and ANOVAs.

    """
    data: typing.Type[pd.DataFrame]
    subject: typing.Union[str, None]
    between = typing.Union[str, list, None]
    within = typing.Union[str, list, None]
    dependent: typing.Union[str, None]
    formula: typing.Union[str, None]
    agg_func: typing.Union[str, typing.Callable]
    _perform_aggregation: bool
    _perform_aggregation: bool
    max_levels = None

    def __init__(
            self,
            data=None,
            dependent=None,
            between=None,
            within=None,
            subject=None,
            formula=None,
            agg_func='mean',
            **kwargs
    ):
        self.agg_func = agg_func
        self.data = data
        self.max_levels = kwargs.get('max_levels', None)
        self._get_variables_and_formula(formula, dependent, between, within,
                                        subject)
        self.data = self._select_data()
        if self._perform_aggregation:
            self.data = self._aggregate_data()
        self.data.dropna(inplace=True)

    def _get_variables_and_formula(self, formula, dependent, between, within,
                                   subject):
        # Parse variables from formula or build formula from entered variables
        if formula is None:
            # Variables setting routine
            self._set_variables(dependent, between, within, subject)
            # Build model formula from entered variables
            self.formula, self._r_formula = self._get_formula_from_vars()
        elif subject is None:
            # Parse models from entered formula
            self.formula, self._r_formula = formula, rst.pyr.rpackages.stats.formula(
                formula)
            dependent, between, within, subject = self._get_vars_from_formula()
            # Variables setting routine
            self._set_variables(dependent, between, within, subject)
        else:
            raise RuntimeError("")

    def _set_variables(self, dependent, between, within, subject):
        # Verify identity variable integrity
        self.subject, self._perform_aggregation = self._set_subject(subject)
        # Verify dependent variable integrity
        self.dependent = self._set_dependent_var(dependent)
        # To test whether DV can be coerced to float
        rst.utils.verify_float(self.data[self.dependent].values)
        # Verify independent variables integrity
        self._between = self._set_independent_vars(between)
        self._within = self._set_independent_vars(within)
        self.independent = self._between + self._within
        self._verify_independent_vars()

    def _get_formula_from_vars(self):
        return None, None

    def _get_vars_from_formula(self):
        return None, [], [], None

    def _set_subject(self, subject):
        if subject not in self.data.columns:
            raise KeyError(
                'Subject ID variable {} not found in data'.format(subject))
        return subject, len(self.data[subject].unique()) < len(self.data)

    def _set_independent_vars(self, ind_vars: object) -> list:
        """Make sure that the you have a list of independent variables"""
        if isinstance(ind_vars, list):
            return ind_vars
        if ind_vars is None:
            return []
        if isinstance(ind_vars, str):
            if ind_vars == '':
                return []
            return [ind_vars]
        if isinstance(self, T1Sample):
            return []
        raise TypeError

    def _verify_independent_vars(self):
        # _ind_vars = self._between + self._within
        for i in self.independent:
            rst.utils.verify_levels(self.data[i].values, self.max_levels)

    def _set_dependent_var(self, dependent):
        if dependent not in self.data.columns:
            raise KeyError("Dependent variable not found in data")
        return dependent

    def _select_data(self):
        data = self.data[
            self.independent +
            [self.subject, self.dependent]].copy()
        # Make sure we are handing R a DataFrame with factor variables,
        # as some packages do not coerce factor type.
        data[self.independent] = data[
            self.independent].astype('category')
        return data

    def _aggregate_data(self):
        return self.data.groupby(
            rst.utils.to_list(
                [self.subject] + self.independent)).agg(
            self.agg_func).reset_index()

    def _run_analysis(self):
        pass

    def _finalize_results(self):
        pass
        # return rst.utils.tidy(self._r_results, type(self).__name__)

    def get_df(self, **kwargs):
        return self.results.copy()

    def get_text_report(self):
        pass

    def get_latex_report(self, **kwargs):
        pass


@register_dataframe_accessor("t2samples")
class T2Samples(_BaseParametric):
    """
    Run a frequentist dependent or independent samples t-test.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFY FORM HERE).


    .. _Implemented R function stats::t.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/t.test

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns
        (usually not in a 'long file' format).
        TODO - see if data can become a non-default.
    paired : bool
        Whether the test is dependent/paired-samples (True) or
        independent-samples (False). Default is True.
    dependent : str, optional
        The name of the column identifying the dependent variable in the data.
        The column data type should be numeric.
    independent : str, optional
        The name of the column identifying the independent variable in the data.
        The column could be either numeric or object, but can contain up to two
        unique values. Alias for `within` for paired, `between` for unpaired.
    subject : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        TODO fill this out
    tail: str, optional
        Direction of the tested hypothesis. Optional values are 'two.sided'
        (H1: x != y), 'less' (H1: x < y) and 'greater' (H1: x > y).
        Default value is 'two.sided'.
    assume_equal_variance : bool, optional
        Applicable only to independent samples t-test. Whether to assume that
        the two samples have equal variance. If True runs regular two-sample,
        if False runs Welch two-sample test. Default is True.

    """

    def __init__(self,
                 paired=True,
                 independent=None,
                 tail='two.sided',
                 assume_equal_variance=True,
                 **kwargs):
        self.paired = paired
        self.tail = tail
        self.assume_equal_variance = assume_equal_variance
        kwargs['max_levels'] = 2

        if independent is None:
            try:
                independent = kwargs[{False: 'between', True: 'within'}[paired]]
            except KeyError as e:
                if paired:
                    print(f'Specify `independent` or `within`: {e}')
                else:
                    print(f'Specify `independent` or `between`: {e}')
                raise e
        else:
            kwargs[{False: 'between', True: 'within'}[paired]] = independent

        super().__init__(**kwargs)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _select_data(self):
        data = self.data[
            rst.utils.to_list(
                [self.dependent, self.independent, self.subject])]
        self.x, self.y = data.groupby(
            getattr(self, 'independent'))[getattr(self, 'dependent')].apply(
            lambda s: s.values)
        return data

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.t_test(
            **{'x': self.x, 'y': self.y, 'paired': self.paired,
               'alternative': self.alternative, 'tail': self.tail,
               'var.equal': self.assume_equal_variance})

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.broom.tidy_htest(self._r_results))

    def get_text_report(self):
        params = self.results
        t_clause = self.results['t']


@register_dataframe_accessor("bayes_t2samples")
class BayesT2Samples(T2Samples):
    """
    Run a frequentist independent-samples t-test.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFY FORM HERE).

    .. _Implemented R function BayesFactor::ttestBF: https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/ttestBF

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns
        (usually not in a 'long file' format).
    dependent : str, optional
        The name of the column identifying the dependent variable in the data.
        The column data type should be numeric.
    independent : str, optional
        The name of the column identifying the independent variable in the data.
        The column could be either numeric or object, but can contain up to two
        unique values.
    subject : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        TODO fill this out
    null_interval: tuple, optional
        Predicted interval for standardized effect size to test against the null
        hypothesis. Optional values for a 'simple' directional test are
        [-np.inf, np.inf], (H1: ES != 0), [-np.inf, 0] (H1: ES < 0) and
        [0, np.inf] (H1: ES > 0). However, it is better to confine null_interval
        specification to a narrower range (e.g., [0.2, 0.5], if you expect
        a positive small standardized effect size), if you have a prior belief
        regarding the expected effect size.
        Default value is [-np.inf, np.inf].
    return_both : bool, optional
        If True returns Bayes factor for both H1 and H0, else, returns only for
        H1. Used only on directional hypothesis. Default is False.
        TODO - implement return of both bayes factors.
    scale_prior : float, optional
        Controls the scale of the prior distribution. Default value is 1.0 which
        yields a standard Cauchy prior. It is also possible to pass 'medium',
        'wide' or 'ultrawide' as input arguments instead of a float (matching
        the values of :math:`\\frac{\\sqrt{2}}{2}, 1, \\sqrt{2}`,
        respectively).
        TODO - limit str input values.
    sample_from_posterior : bool, optional
        If True return samples from the posterior, if False returns Bayes
        factor. Default is False.
    iterations : int, optional
        Number of samples used to estimate Bayes factor or posterior. Default
        is 1000.
        # mu (on dependent Bayesian t-test)
    """

    def __init__(
            self,
            null_interval=(-np.inf, np.inf),
            prior_r_scale: str = 'medium',
            sample_from_posterior: bool = False,
            iterations: int = 10000,
            mu: float = 0,
            **kwargs
    ):
        self.null_interval = np.array(null_interval)
        self.prior_r_scale = prior_r_scale
        self.sample_from_posterior = sample_from_posterior
        self.iterations = iterations
        self.mu = mu
        super().__init__(**kwargs)

    def _run_analysis(self):
        return rst.pyr.rpackages.base.data_frame(
            rst.pyr.rpackages.bayesfactor.ttestBF(
                x=self.x,
                y=self.y,
                paired=self.paired,
                nullInterval=self.null_interval,
                iterations=self.iterations,
                posterior=self.sample_from_posterior,
                rscalse=self.prior_r_scale,
                simple=True,
                mu=self.mu
            ))

    def _finalize_results(self):
        return rst.utils.convert_df(self._r_results)


@register_dataframe_accessor("t1sample")
class T1Sample(T2Samples):
    """
    Run a frequentist one-sample t-test.

    To run a model either specify the dependent and subject
    variables, or enter a formula (SPECIFY FORM HERE).

    .. _Implemented R function stats::t.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/t.test

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns
        (usually not in a 'long file' format).
    dependent : str, optional
        The name of the column identifying the dependent variable in the data.
        The column data type should be numeric.
    subject : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        TODO fill this out based on whether we will depractate the formula
            functionality or not.
    mu :  float, optional
        Population mean value to text x against. Default value is 0.
    tail: str, optional
        Direction of the tested hypothesis. Optional values are 'two.sided'
        (H1: x != mu), 'less' (H1: x < mu) and 'greater' (H1: x > mu).
        Default value is 'two.sided'.
        TODO allow translation of x != y, x > y, x < y
    """

    def __init__(self,
                 tail='two.sided',
                 mu=0,
                 **kwargs):
        self.mu = mu
        kwargs['paired'] = True
        self.max_levels = 1
        super().__init__(**kwargs)

    def _select_data(self):
        data = self.data[
            rst.utils.to_list(
                [self.dependent, self.independent, self.subject])]
        self.x = data[getattr(self, 'dependent')].values
        return data

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.t_test(
            x=self.x,
            mu=self.mu,
            alternative=self.tail
        )


@register_dataframe_accessor("bayes_t1sample")
class BayesT1Sample(T1Sample):
    """
    Run a frequentist independent-samples t-test.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFY FORM HERE).

    .. _Implemented R function BayesFactor::ttestBF: https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/ttestBF

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns
        (usually not in a 'long file' format).
    dependent : str, optional
        The name of the column identifying the dependent variable in the data.
        The column data type should be numeric.
    independent : str, optional
        The name of the column identifying the independent variable in the data.
        The column could be either numeric or object, but can contain up to two
        unique values.
    subject : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        TODO fill this out
    null_interval: tuple, optional
        Predicted interval for standardized effect size to test against the null
        hypothesis. Optional values for a 'simple' directional test are
        [-np.inf, np.inf], (H1: ES != 0), [-np.inf, 0] (H1: ES < 0) and
        [0, np.inf] (H1: ES > 0). However, it is better to confine null_interval
        specification to a more confined range (e.g., [0.2, 0.5], if you expect
        a positive small standardized effect size), if you have a prior belief
        regarding the expected effect size.
        Default value is [-np.inf, np.inf].
    return_both : bool, optional
        If True returns Bayes factor for both H1 and H0, else, returns only for
        H1. Used only on directional hypothesis. Default is False.
        TODO - implement return of both bayes factors.
    scale_prior : float, optional
        Controls the scale of the prior distribution. Default value is 1.0 which
        yields a standard Cauchy prior. It is also possible to pass 'medium',
        'wide' or 'ultrawide' as input arguments instead of a float (matching
        the values of :math:`\\frac{\\sqrt{2}}{2}, 1, \\sqrt{2}`, respectively).
        TODO - limit str input values.
    sample_from_posterior : bool, optional
        If True return samples from the posterior, if False returns Bayes
        factor. Default is False.
    iterations : int, optional
        Number of samples used to estimate Bayes factor or posterior. Default
        is 1000.
    mu : float, optional
        The null (H0) predicted value for the mean difference compared with the
        population's mean (mu), in un-standardized effect size (e.g., raw
        measure units). Default value is 0.
    """

    def __init__(
            self,
            null_interval=(-np.inf, np.inf),
            prior_r_scale: str = 'medium',
            sample_from_posterior: bool = False,
            iterations: int = 10000,
            mu: float = 0.0,
            **kwargs):
        self.null_interval = np.array(null_interval)
        self.prior_r_scale = prior_r_scale
        self.sample_from_posterior = sample_from_posterior
        self.iterations = iterations
        kwargs['mu'] = mu
        super().__init__(**kwargs)

    def _run_analysis(self):
        return rst.pyr.rpackages.base.data_frame(
            rst.pyr.rpackages.bayesfactor.ttestBF(
                x=self.x,
                nullInterval=self.null_interval,
                iterations=self.iterations,
                posterior=self.sample_from_posterior,
                rscalse=self.prior_r_scale,
                mu=self.mu
            ))

    def _finalize_results(self):
        return rst.utils.convert_df(self._r_results)

@register_dataframe_accessor("anova")
class Anova(_BaseParametric):
    """
    Run a mixed (between + within) frequentist ANOVA.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFY FORM HERE).

    .. _Implemented R function afex::aov_4: https://www.rdocumentation.org/packages/afex/versions/0.27-2/topics/aov_car

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the subject, dependent and independent variable(s) as
        columns.
    dependent : str, optional
        The name of the column identifying the dependent variable in the data.
        The column data type should be numeric.
    between, within : list[str], optional
        For both `between` and `within` - Either a string or a list with the
        name(s) of the independent variables in the data. The column data type
         should be preferebly be string or category.
    subject : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        An lme4-like formula specifying the model, in the form of
        dependent ~ (within | subject). See examples.
        TODO - give a more verbose example
    effect_size: str, optional
        Optional values are 'ges', (:math:`generalized-{\\eta}^2`) 'pes'
        (:math:`partial-{\\eta}^2`) or 'none'. Default value is 'pes'.
    correction: str, optional
        Sphericity correction method. Possible values are "GG"
        (Greenhouse-Geisser), "HF" (Hyunh-Feldt) or "none". Default is 'none'.
    """

    def __init__(self, **kwargs):
        self.effect_size = kwargs.get('effect_size', 'pes')
        self.sphericity_correction = kwargs.get('sphericity_correction', 'none')

        super().__init__(**kwargs)
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _get_formula_from_vars(self):
        return rst.utils.build_lm4_style_formula(
            dependent=self.dependent,
            between=self._between,
            within=self._within,
            subject=self.subject,
            es=self.effect_size,
            correction=self.sphericity_correction
        )

    def _get_vars_from_formula(self):
        return rst.utils.parse_variables_from_lm4_style_formula(self.formula)

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.anova(
            rst.pyr.rpackages.afex.aov_4(
                formula=self._r_formula,
                data=rst.utils.convert_df(self.data)))
        # TODO: Add reliance on aov_ez aggregation functionality.
        # TODO: Add this functionality - sig_symbols= rst.pyr.vectors.StrVector(["", "", "", ""]))

    def _finalize_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.broom.tidy_anova(self._r_results))


@register_dataframe_accessor("bayes_anova")
class BayesAnova(Anova):
    # TODO - Formula specification will be using the lme4 syntax and variables
    #  will be parsed from it
    """
    Run a mixed (between + within) Bayesian ANOVA.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFY FORM HERE).

    .. _Implemented R function BayesFactor::anovaBF: https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/anovaBF

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the subject, dependent and independent variable(s) as
        columns.
    dependent : str, optional
        The name of the column identifying the dependent variable in the data.
        The column data type should be numeric.
    between, within : list[str], optional
        For both `between` and `within` - Either a string or a list with the
        name(s) of the independent variables in the data. The column data type
         should be preferebly be string or category.
    subject : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        An lme4-like formula specifying the model, in the form of
        dependent ~ (within | subject). See examples.
        TODO - give a more verbose example
    which_models : str, optional
        Setting which_models to 'all' will test all models that can be
        created by including or not including a main effect or interaction.
        'top' will test all models that can be created by removing or
        leaving in a main effect or interaction term from the full model.
        'bottom' creates models by adding single factors or interactions to
        the null model. 'withmain' will test all models, with the constraint
        that if an interaction is included, the corresponding main effects
        are also included. Default value is 'withmain'.
    iterations : int, optional
        Number of iterations used to estimate Bayes factor. Default value is
         10000.
    scale_prior_fixed : [float, str], optional
        Controls the scale of the prior distribution for fixed factors.
        Default value is 1.0  which yields a standard Cauchy prior.
        It is also possible to pass 'medium', 'wide' or 'ultrawide' as input
         arguments instead of a float (matching the values of
        :math:`\\frac{\\sqrt{2}}{2}, 1, \\sqrt{2}`, respectively).

    scale_prior_random : [float, str], optional
        Similar to `scale_prior_fixed` but applies to random factors and can
        except all values specified above and also 'nuisance' - variance
        in the data that may stem from variables which are irrelevant to the
        model, such as participants. Default value is 'nuisance'.
        r_scale_effects=rst.pyr.rinterface.NULL,
    mutli_core : bool, optional
        Whether to use multiple cores for estimation. Not available on
        Windows. Default value is False.
    method : str, optional
        The method used to estimate the Bayes factor depends on the method
        argument. "simple" is most accurate for small to moderate sample
        sizes, and uses the Monte Carlo sampling method described in Rouder
        et al. (2012). "importance" uses an importance sampling algorithm
        with an importance distribution that is multivariate normal on
        log(g). "laplace" does not sample, but uses a Laplace approximation
        to the integral. It is expected to be more accurate for large sample
        sizes, where MC sampling is slow. If method="auto", then an initial
        run with both samplers is done, and
        the sampling method that yields the least-variable samples is
        chosen. The number of initial test iterations is determined by
        options(BFpretestIterations).
     no_sample : bool, optional
        Will prevent sampling when possible (i.e., rely on calculation of
        BayesFactor). Default value is False. Currently `True` is not
        implemented and may lead to an error.
        TODO test how we would handle `no_sample = True`

    """

    def __init__(
            self,
            which_models="withmain",
            iterations=10000,
            scale_prior_fixed="medium",
            scale_prior_random="nuisance",
            r_scale_effects=rst.pyr.rinterface.NULL,
            multi_core=False,
            method="auto",
            no_sample=False,
            include_subject=False,
            # TODO - document `include_subject`
            **kwargs):
        self.which_models = which_models
        self.iterations = iterations
        self.r_scale_fixed = scale_prior_fixed
        self.r_scale_random = scale_prior_random
        self.r_scale_effects = r_scale_effects
        self.multi_core = multi_core
        self.method = method
        self.no_sample = no_sample
        self.include_subject = include_subject

        if not ('between' in kwargs and 'within' in kwargs):
            raise ValueError(
                "A Mixed Anova was selected. Specify both `between`"
                "and `within`")
        super().__init__(**kwargs)

    def _get_formula_from_vars(self):
        frml = rst.utils.bayes_style_formula(self.dependent, self._between,
                                             self._within,
                                             subject={
                                                 False: None,
                                                 True: self.subject
                                             }[self.include_subject])
        return frml, rst.pyr.rpackages.stats.formula(frml)

    def _get_vars_from_formula(self):
        dependent, between, within, subject = rst.utils.parse_variables_from_lm4_style_formula(
            self.formula)
        # And now we can update the formula, as BayesFactor packages requires
        self._update_formula()
        return dependent, between, within, subject

    def _update_formula(self):
        self.formula = rst.utils.bayes_style_formula(self.formula)
        self._r_formula = rst.pyr.rpackages.stats.formula(self.formula)

    def _run_analysis(self):
        return rst.pyr.rpackages.bayesfactor.anovaBF(
            self._r_formula,
            rst.utils.convert_df(self.data),
            whichRandom=self.subject,
            whichModels=self.which_models,
            iterations=self.iterations,
            rscaleFixed=self.r_scale_fixed,
            rscaleRandom=self.r_scale_random,
            rscaleEffects=self.r_scale_effects,
            multicore=self.multi_core,
            method=self.method,
            # noSample=self.no_sample
        )

    def _finalize_results(self):
        self._r_results = rst.pyr.rpackages.bayesfactor.extractBF(
            self._r_results)
        return rst.utils.convert_df(self._r_results).join(
            pd.DataFrame(np.array(self._r_results.rownames),
                         columns=['model'])
        )[['model', 'bf', 'error']]


class PairwiseComparison:
    """To be implemented - runs all pairwise comparisons between two groups
    in a dataframe"""

    def __init__(self, data, correction_method):
        self.data = data
        self.correction_method = correction_method


class MixedModel:
    pass
    # TODO: This! ^
    # Here is a working example using the mtcars dataset.
    #  m1 = rst.pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # rst.utils.convert_df(rst.pyr.rpackages.afex.nice(m1))
