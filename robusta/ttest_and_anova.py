import typing
import pandas as pd
import numpy as np
from dataclasses import dataclass
import robusta as rst
from pandas_flavor import register_dataframe_accessor

__all__ = [
    "AnovaBS", "AnovaWS", "AnovaMixed",
    "BayesAnovaBS", "BayesAnovaWS", "BayesAnovaMixed",
    "T1Sample", "T2DepSamples", "T2IndSamples",
    "BayesT1Sample", "BayesT2DepSamples", "BayesT2IndSamples"
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
        if isinstance(self, OneSampleTTest):
            return []
        raise TypeError

    def _verify_independent_vars(self):
        #_ind_vars = self._between + self._within
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
            self.independent].astype('category').values
        return data

    def _aggregate_data(self):
        return self.data.groupby(
            rst.utils.to_list(
                [self.subject] + self.independent)).agg(
            self.agg_func).reset_index()

    def _run_analysis(self):
        pass

    def _finalize_results(self):
        return rst.utils.tidy(self._r_results, type(self).__name__)



class _T2Samples(_BaseParametric):
    """
    A base class used to run a two sample t-test.
    """
    def __init__(self,
                 **kwargs):
        kwargs['max_levels'] = 2
        self.paired = kwargs['paired']
        self.tail = kwargs['tail']
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
            x=self.x,
            y=self.y,
            paired=self.paired, alternative=self.tail)


@register_dataframe_accessor("t_2_ind_samples")
class T2IndSamples(_T2Samples):
    """
    Run a frequentist independent-samples t-test.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFIY FORM HERE).

    Implemented R function - stats::t.test
    (https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/t.test)

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
    dependent : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        TODO fill this out
    tail: str, optional
        Direction of the tested hypothesis. Optional values are 'two.sided'
        (H1: x != y), 'less' (H1: x < y) and 'greater' (H1: x > y).
        Default value is 'two.sided'.
        TODO allow translation of x != y, x > y, x < y
    """
    def __init__(self,
                 independent=None,
                 tail='two.sided',
                 **kwargs):
        kwargs['between'] = independent
        kwargs['paired'] = False
        kwargs['tail'] = tail
        super().__init__(**kwargs)


@register_dataframe_accessor("t_2_dep_samples")
class T2DepSamples(_T2Samples):
    """
    Run a frequentist dependent-samples t-test.

    To run a model either specify the dependent, independent and subject
    variables, or enter a formula (SPECIFIY FORM HERE).

    Implemented R function - stats::t.test
    (https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/t.test)

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
    dependent : str, optional
        The name of the column identifying the subject variable in the data.
    formula : str, optional
        TODO fill this out
    """

    def __init__(self,
                 independent=None,
                 tail='two.sided',
                 **kwargs):
        kwargs['between'] = independent
        kwargs['paired'] = True
        kwargs['tail'] = tail
        super().__init__(**kwargs)

@register_dataframe_accessor("t_1_sample")
class T1Sample(_T2Samples):
    def __init__(self,
                 paired=None,
                 tail='two.sided',
                 mu=0,
                 **kwargs):
        self.mu = mu
        kwargs['paired'] = paired
        kwargs['tail'] = tail
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

class _BayesT2Samples(_T2Samples):

    def __init__(
            self,
            null_interval=(-np.inf, np.inf),
            prior_r_scale: str = 'medium',
            sample_from_posterior: bool = False,
            iterations: int = 10000,
            mu: float = 0.0,
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
                rscalse=self.prior_r_scale
            ))

@register_dataframe_accessor("bayes_t_2_ind_samples")
class BayesT2IndSamples(_BayesT2Samples):

    def __init__(self, **kwargs):
        kwargs['paired'] = False
        super().__init__(**kwargs)

@register_dataframe_accessor("bayes_t_2_dep_samples")
class BayesT2DepSamples(_BayesT2Samples):

    def __init__(self, **kwargs):
        kwargs['paired'] = True
        super().__init__(**kwargs)

@register_dataframe_accessor("bayes_t_1_sample")
class BayesT1Sample(T1Sample):
    def __init__(
            self,
            null_interval = (-np.inf, np.inf),
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

class _Anova(_BaseParametric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.test_type = self._infer_test_type()
        self._r_results = self._run_analysis()
        self.results = self._finalize_results()

    def _get_formula_from_vars(self):
        return rst.utils.build_lm4_style_formula(
            dependent=self.dependent,
            between=self._between,
            within=self._within,
            subject=self.subject
        )

    def _get_vars_from_formula(self):
        return rst.utils.parse_variables_from_lm4_style_formula(self.formula)

    def _infer_test_type(self):
        # Mainly for the user, not really doing anything in terms of code
        if not self._between:
            self._between = []
            return 'Within'
        if not self._within:
            self._within = []
            return 'Between'
        return 'Mixed'

    def _run_analysis(self):
        return rst.pyr.rpackages.stats.anova(
            rst.pyr.rpackages.afex.aov_4(
                formula=self._r_formula,
                data=rst.utils.convert_df(self.data)))
        # TODO: Add reliance on aov_ez aggregation functionality.
        # TODO: Add this functionality - sig_symbols= rst.pyr.vectors.StrVector(["", "", "", ""]))


@register_dataframe_accessor("anova_ws")
class AnovaWS(_Anova):
    def __init__(self, **kwargs):
        if 'between' in kwargs:
            raise ValueError(
                "A Within-Subject Anova was selected. Either remove `between` or"
                " use BetweenSubjectsAnova / MixedAnova")
        super().__init__(**kwargs)


@register_dataframe_accessor("anova_bs")
class AnovaBS(_Anova):
    def __init__(self, **kwargs):
        if 'within' in kwargs:
            raise ValueError(
                "A Within-Subject Anova was selected. Either remove `between`"
                "or use BetweenSubjectsAnova / MixedAnova")
        super().__init__(**kwargs)


@register_dataframe_accessor("anova_mixed")
class AnovaMixed(_Anova):
    def __init__(self, **kwargs):
        if not ('between' in kwargs and 'within' in kwargs):
            raise ValueError(
                "A Mixed Anova was selected. Specify both `between`"
                "and `within`")
        super().__init__(**kwargs)


class _BayesAnova(_Anova):
    # TODO - Formula specification will be using the lme4 syntax and variables will be parsed from it

    def __init__(
            self,
            which_models="withmain",  # TODO Find out all the possible options
            iterations=10000,
            r_scale_fixed="medium",  # TODO Find out all the possible options
            r_scale_random="nuisance",
            r_scale_effects=rst.pyr.rinterface.NULL,
            multi_core=False,
            method="auto",
            no_sample=False,
            **kwargs):
        self.which_models = which_models
        self.iterations = iterations
        self.r_scale_fixed = r_scale_fixed
        self.r_scale_random = r_scale_random
        self.r_scale_effects = r_scale_effects
        self.multi_core = multi_core
        self.method = method
        self.no_sample = no_sample
        # This is a scaffold. It will be removed when we would have formula compatibility for all tests.
        super().__init__(**kwargs)

    def _get_formula_from_vars(self):
        frml, _ = rst.utils.build_lm4_style_formula(
            dependent=self.dependent,
            between=self._between,
            within=self._within,
            subject=self.subject
        )
        frml = rst.utils.bayes_style_formula(frml)
        return frml, rst.pyr.rpackages.stats.formula(frml)

    def _get_vars_from_formula(self):
        dependent, between, within, subject = rst.utils.parse_variables_from_lm4_style_formula(self.formula)
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
            noSample=self.no_sample
        )

    def _finalize_results(self):
            self._r_results = rst.pyr.rpackages.bayesfactor.extractBF(self._r_results)
            return rst.utils.tidy(
                rst.utils.convert_df(self._r_results).join(
                    pd.DataFrame(np.array(self._r_results.rownames), columns=['model'])
                ), type(self).__name__)

@register_dataframe_accessor("bayes_anova_ws")
class BayesAnovaWS(_BayesAnova):
    def __init__(self, **kwargs):
        if 'between' in kwargs:
            raise ValueError(
                "A Within-Subject Anova was selected. Either remove `between` or"
                " use BetweenSubjectsAnova / MixedAnova")
        super().__init__(**kwargs)

@register_dataframe_accessor("bayes_anova_bs")
class BayesAnovaBS(_BayesAnova):
    def __init__(self, **kwargs):
        if 'within' in kwargs:
            raise ValueError(
                "A Within-Subject Anova was selected. Either remove `between` or"
                " use BetweenSubjectsAnova / MixedAnova")
        super().__init__(**kwargs)

@register_dataframe_accessor("bayes_anova_mixed")
class BayesAnovaMixed(_BayesAnova):
    def __init__(self, **kwargs):
        if not ('between' in kwargs and 'within' in kwargs):
            raise ValueError(
                "A Mixed Anova was selected. Specify both `between`"
                "and `within`")
        super().__init__(**kwargs)

class PairwiseComparison:
    def __init__(self, data, correction_method):
        self.data = data
        self.correction_method = correction_method


class MixedModel:
    pass
    # TODO: This! ^
    # Here is a working example using the mtcars dataset.
    #  m1 = rst.pyr.rpackages.afex.mixed('qsec ~ mpg + (mpg|am)', data=data.reset_index(drop=False))
    # rst.utils.convert_df(rst.pyr.rpackages.afex.nice(m1))
