# TODO - define __repr__ and __str__ for all classes

"""
ttest_and_anova contains classes used to run statistical tests in which a
central tendency measure of groups is compared:
- Anova, BayesAnovaBS: Between/Within/Mixed n-way analysis of variance
    (frequentist and Bayesian)
- T2Samples, BayesT2Samples: Independent/Dependent samples t-test
    (Frequentist and Bayesian)
- T1Sample, BayesT1Sample: One-Sample t-test (Frequentist and Bayesian)

Additionally, there are classes for running non-parametric tests such as:
- Wilcoxon1Sample, Wilcoxon2Sample: One- and two-sample variations of the
 Wilcoxon signed rank test
- KruskalWalisTest, FriedmanTest: The Kruskal-Walis test and Friedman test
    are used as non parametric analysis of variance for between-subject and
    within-subject designs, respectively.

All classes have at least these postestimation methods:
- get_results(): Returns a pandas dataframe with the results of the analysis.
- get_text_report(): Returns a textual report of the analysis.
"""

import typing
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

import robusta as rst
from . import base

BF_COLUMNS = ['model', 'bf', 'error']



__all__ = [
    "AnovaModel", "BayesAnovaModel",
    "T1SampleModel", "T2SamplesModel",
    "BayesT1SampleModel", "BayesT2SamplesModel",
    "Wilcoxon1SampleModel", "Wilcoxon2SamplesModel",
    "KruskalWallisTestModel", "FriedmanTestModel"
]

DEFAULT_GROUPWISE_NULL_INTERVAL = (-np.inf, np.inf)
PRIOR_SCALE_STR_OPTIONS = ('medium', 'wide', 'ultrawide')
DEFAULT_ITERATIONS = 10000
DEFAULT_MU = 0


@dataclass
class GroupwiseModel(base.BaseModel):
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
    max_levels = None
    min_levels = None

    def __init__(
            self,
            data: pd.DataFrame = None,
            dependent: str = None,
            between: typing.Optional[str] = None,
            within: typing.Optional[str] = None,
            subject: str = None,
            formula: typing.Optional[str] = None,
            na_action: typing.Optional[str] = None,
            agg_func: typing.Optional[
                typing.Union[str, typing.Callable]] = np.mean,
            **kwargs
    ):

        vars(self).update(
            data=data,
            formula=formula,
            dependent=dependent,
            between=between,
            within=within,
            subject=subject,
            agg_func=agg_func,
            na_action=na_action,
            max_levels=kwargs.get('max_levels', None)
        )
        self.data.dropna(inplace=True)

        # TODO - on the new structure nothing is run on init, so do we really
        #  need the call to super here?
        super().__init__()

    def _pre_process(self):
        self._set_controllers()
        self._select_input_data()
        self._validate_input_data()
        self._transform_input_data()

    def _set_controllers(self):
        """
        This function set
        @param: self
        @return: None
        """
        # Parse variables from formula or build formula from entered variables
        if self.formula is None:
            # Variables setting routine
            self._set_variables()
            # Build model formula from entered variables
            self._set_formula_from_vars()
        elif self.formula is not None:
            # Parse models from entered formula
            self._r_formula = self.formula, rst.pyr.rpackages.stats.formula(
                self.formula)
            self._set_vars_from_formula()
            # Variables setting routine
            self._set_variables()

    def _set_formula_from_vars(self):
        fp = rst.formula_tools.FormulaParser(
            self.dependent, self.between, self.within, self.subject
        )
        self.formula = fp.get_formula()
        self._r_formula = rst.pyr.rpackages.stats.formula(self.formula)

    def _set_vars_from_formula(self):
        vp = rst.formula_tools.VariablesParser(self.formula)
        (self.dependent, self.between, self.within,
         self.subject) = vp.get_variables()

    def _set_variables(self):
        # Verify independent variables integrity
        self.between = self._convert_independent_vars_to_list(self.between)
        self.within = self._convert_independent_vars_to_list(self.within)
        self.independent = self.between + self.within

    def _convert_independent_vars_to_list(self, ind_vars: object) -> list:
        """Make sure that the you have a list of independent variables"""
        if isinstance(ind_vars, list):
            return ind_vars
        if ind_vars is None:
            return []
        if isinstance(ind_vars, str):
            if ind_vars == '':
                return []
            return [ind_vars]
        if isinstance(self, T1SampleModel):
            return []
        raise TypeError

    def _select_input_data(self):
        try:
            data = self.data[
                self.independent +
                [self.subject, self.dependent]].copy()
        except KeyError:
            cols = self.data.columns
            vars = [i for i in [
                self.independent +
                [self.subject, self.dependent]] if i not in cols]
            raise RuntimeError(f"Variables not in data: \n {','.join(vars)}")
        self._input_data = data

    def _validate_input_data(self):
        # To test whether DV can be coerced to float
        rst.utils.verify_float(self._input_data[self.dependent].values)

        # Verify ID variable uniqueness
        self._perform_aggregation = (
                self._input_data.groupby(
                    [self.subject] + self.independent).size().any() > 1)

        # Make sure that you have at least two levels on each independent
        # variable
        self._verify_independent_vars()

    def _transform_input_data(self):
        # Make sure we are handing R a DataFrame with factor variables,
        # as some packages do not coerce factor type.

        if self._perform_aggregation:
            self._input_data = self._aggregate_data()

        self._input_data.loc[:, self.independent] = self._input_data[
            self.independent].astype('category')
        self._r_input_data = rst.utils.convert_df(self._input_data.copy())

    def _verify_independent_vars(self):

        for i in self.independent:
            rst.utils.verify_levels(self._input_data[self.independent],
                                    self.min_levels, self.max_levels)

    def _aggregate_data(self):
        return self._input_data.groupby(
            rst.utils.to_list(
                [self.subject] + self.independent)).agg(
            self.agg_func).reset_index()

    def _analyze(self):
        pass

    def fit(self):
        """
        This method runs the model defined by the input.
        @return:
        """
        self._pre_process()
        # returns the results objects that is created with the (r) results object
        return self._analyze()


class GroupwiseResults(base.BaseResults):

    def get_text(self, mode: str = 'df'):
        raise NotImplementedError
        if mode == 'df':
            return rst.pyr.rpackages.report.as_data_frame_report(
                self._r_results)
        if mode == 'verbose':
            return rst.pyr.rpackages.report.report(self._r_results)

    # def get_df(self):
    #    return self._tidy_results()

    # def _tidy_results(self):
    #    return rst.utils.convert_df(
    #        rst.pyr.rpackages.generics.tidy(self.r_results))


class T2SamplesModel(GroupwiseModel):
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

    alternative: str

    def __init__(self,
                 paired: bool = None,
                 independent: bool = None,
                 tail: str = 'two.sided',
                 assume_equal_variance: bool = True,
                 x=None,
                 y=None,
                 correct=False,
                 **kwargs):
        self.paired = paired
        self.tail = tail
        self.assume_equal_variance = assume_equal_variance
        kwargs['max_levels'] = 2
        kwargs['min_levels'] = 2

        if kwargs.get('formula') is None:
            if independent is None:
                try:
                    independent = kwargs[{
                        False: 'between', True: 'within'}[paired]]
                except KeyError as e:
                    if paired:
                        raise TypeError(
                            f'Specify `independent` or `within`: {e}')
                    else:
                        raise TypeError(
                            f'Specify `independent` or `between`: {e}')
            else:
                kwargs[{False: 'between', True: 'within'}[paired]] = independent

        super().__init__(**kwargs)

    def _select_input_data(self):
        super()._select_input_data()
        self.x, self.y = self._input_data.groupby(
            getattr(self, 'independent'))[getattr(self, 'dependent')].apply(
            lambda s: s.values)

    def _analyze(self):
        return T2SamplesResults(
            rst.pyr.rpackages.stats.t_test(
                **{'x': self.x, 'y': self.y,
                   'paired': self.paired,
                   'alternative': self.tail,
                   'var.equal': self.assume_equal_variance,
                   })
        )


class T2SamplesResults(GroupwiseResults):

    def get_text_report(self):
        params = self.r_results
        t_clause = self.results['t']


class BayesT2SamplesModel(T2SamplesModel):
    """
    Run a Bayesian independent-samples t-test.

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
            null_interval=DEFAULT_GROUPWISE_NULL_INTERVAL,
            prior_scale: str = 'medium',
            sample_from_posterior: bool = False,
            iterations: int = DEFAULT_ITERATIONS,
            mu: float = 0,
            **kwargs
    ):
        self.null_interval = np.array(null_interval)
        self.prior_scale = prior_scale
        self.sample_from_posterior = sample_from_posterior
        self.iterations = iterations
        self.mu = mu
        if kwargs['paired'] is False:
            if mu != 0: raise ValueError
        super().__init__(**kwargs)

    # def re_analyze(self, x=np.nan, y=np.nan, paired=np.nan,
    #                sample_from_posterior=np.nan):
    #     kwargs = {
    #         'x': self.x if x is np.nan else x,
    #         'y': self.y if y is np.nan else y,
    #         'paired': self.paired if paired is np.nan else paired,
    #         'nullInterval': self.null_interval,
    #         'iterations': self.iterations,
    #         'posterior': self.sample_from_posterior if sample_from_posterior is np.nan else sample_from_posterior,
    #         'rscale': self.prior_scale,
    #         'mu': self.mu
    #     }
    #
    #     rst.pyr.rpackages.base.data_frame(
    #         rst.pyr.rpackages.bayesfactor.ttestBF(
    #             **kwargs
    #         ))

    def _analyze(self):
        return BayesT2SamplesResults(
            rst.pyr.rpackages.base.data_frame(
                rst.pyr.rpackages.bayesfactor.ttestBF(
                    x=self.x,
                    y=self.y,
                    paired=self.paired,
                    nullInterval=self.null_interval,
                    iterations=self.iterations,
                    posterior=self.sample_from_posterior,
                    rscale=self.prior_scale,
                    mu=self.mu
                )),
            # TODO - refactor
            mode='posterior' if self.sample_from_posterior else 'bf'
        )


class BayesT2SamplesResults(T2SamplesResults):
    # TODO Validate if correctly inherits init from parent

    def __init__(self, r_results, mode='bf'):
        self.mode = mode

        super().__init__(r_results)

    def _tidy_results(self):
        if self.mode == 'bf':
            return rst.utils.convert_df(self.r_results, 'model')[BF_COLUMNS]
        else:
            return rst.utils.convert_df(self.r_results)



class T1SampleModel(T2SamplesModel):
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
                 mu=DEFAULT_MU,
                 **kwargs):
        kwargs['paired'] = True
        kwargs['tail'] = tail
        self.mu = mu
        self.max_levels = 1
        self.min_levels = 1
        super().__init__(**kwargs)

    def _select_input_data(self):
        # Skip the method implemented by parent (T2Samples)
        GroupwiseModel._select_input_data(self)
        self.x = self._input_data[getattr(self, 'dependent')].values

    def _analyze(self):
        return T1SampleResults(rst.pyr.rpackages.stats.t_test(
            x=self.x,
            mu=self.mu,
            alternative=self.tail
        )
        )


class T1SampleResults(T2SamplesResults):
    pass


class BayesT1SampleModel(T1SampleModel):
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
            prior_scale: typing.Union[str, float] = 1 / np.sqrt(2),
            sample_from_posterior: bool = False,
            iterations: int = 10000,
            mu: float = 0.0,
            **kwargs):
        self.null_interval = np.array(null_interval)
        self.prior_scale = prior_scale
        self.sample_from_posterior = sample_from_posterior
        self.iterations = iterations
        kwargs['mu'] = mu
        super().__init__(**kwargs)

    def _analyze(self):
        return BayesT1SampleResults(
            rst.pyr.rpackages.base.data_frame(
                rst.pyr.rpackages.bayesfactor.ttestBF(
                x=self.x,
                nullInterval=self.null_interval,
                iterations=self.iterations,
                posterior=self.sample_from_posterior,
                rscale=self.prior_scale,
                mu=self.mu
            )),
            mode='posterior' if self.sample_from_posterior else 'bf')



# TODO - this is fairly similar to the two-sample bayesian t-test, so consider
#  merging them
class BayesT1SampleResults(T1SampleResults):

    def __init__(self, r_results, mode='bf'):
        self.mode = mode

        super().__init__(r_results)

    def _tidy_results(self):
        if self.mode == 'bf':
            return rst.utils.convert_df(self.r_results, 'model')[BF_COLUMNS]
        else:
            return rst.utils.convert_df(self.r_results, 'iteration')


class AnovaModel(GroupwiseModel):
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

    def __init__(self, margins_term=None, **kwargs):
        self.effect_size = kwargs.get('effect_size', 'pes')
        self.sphericity_correction = kwargs.get(
            'sphericity_correction', 'none')
        self.margins_term = margins_term

        super().__init__(**kwargs)

        self.margins_results = {}

        if self.margins_term is not None:
            self.get_margins()

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.afex.aov_4(
            formula=self._r_formula,
            data=self._r_input_data,
            es=self.effect_size,
            correction=self.sphericity_correction)
        # TODO: Add reliance on aov_ez aggregation functionality.
        # TODO: Add this functionality - sig_symbols=
        # rst.pyr.vectors.StrVector(["", "", "", ""]))


class AnovaResults(GroupwiseResults):

    def _tidy_results(self):
        return rst.utils.convert_df(
            rst.pyr.rpackages.generics.tidy(
                rst.pyr.rpackages.stats.anova(self.r_results)))

    def get_margins(
            self,
            margins_term: typing.List[str] = None,
            by: typing.List[str] = rst.pyr.rinterface.NULL,
            ci: int = 95,
            overwrite_margins_results: bool = False):
        # TODO Look at this - emmeans::as_data_frame_emm_list
        # TODO - Documentation
        # TODO - Issue the 'balanced-design' warning etc.
        # TODO - allow for `by` option
        # TODO - implement between and within CI calculation

        # if by is not None:
        #    raise NotImplementedError
        if margins_term is None:
            if self.margins_term is None:
                raise RuntimeError('No margins term defined')
            margins_term = self.margins_term

        if margins_term in self.margins_results and not overwrite_margins_results:
            # TODO - either way this is only a temporary implementation. It is
            #  insufficient in the case the a user wants to run a margins analysis with
            #   a similar margins term but with different parameters (e.g.
            #   different adjust, type, etc.)
            raise RuntimeError(
                'Margins already defined. To re-run, get_margins'
                'with `overwrite_margins_results` kwarg set '
                'to True')

        # TODO Currently this does not support integer arguments if we would
        #  get those. Also we need to make sure that we get a list or array
        #  as RPy2 doesn't take tuples. Probably should go with array as this
        #  might save time on checking whether the margins term is in the model,
        #  without making sure we are not comparing a string and a list.
        if isinstance(margins_term, str):
            margins_term = [margins_term]
        margins_term = np.array(margins_term)
        if not all(
                term in self.get_results()['term'].values for term in
                margins_term):
            raise RuntimeError(
                f'Margins term: {[i for i in margins_term]}'
                'not included in model')

        _r_margins = rst.pyr.rpackages.emmeans.emmeans(
            self.r_results,
            specs=margins_term,
            type='response',
            level=ci,
            # by=by
        )
        margins = rst.utils.convert_df(
            rst.pyr.rpackages.emmeans.as_data_frame_emmGrid(
                _r_margins))

        self.margins_results[tuple(margins_term)] = {
            'margins': margins, 'r_margins': _r_margins}
        return margins

    def get_pairwise(self,
                     margins_term: typing.Optional[typing.Union[str]] = None,
                     overwrite_pairwise_results: bool = False):
        # TODO - Documentation.
        # TODO - Testing.
        # TODO - implement pairwise options (e.g., `infer`)
        warnings.warn('Currently under development. Expect bugs.')

        if not isinstance(str, margins_term):
            if self.margins_term is None:
                raise RuntimeError("No margins_term defined")
            else:
                margins_term = self.margins_term
        if '*' in margins_term:  # an interaction
            raise RuntimeError(
                'get_pairwise cannot be run using an interaction'
                'margins term')
        if margins_term in self.margins_results:
            if ('pairwise' in self.margins_results[margins_term]
                    and not overwrite_pairwise_results):
                raise RuntimeError(
                    'Margins already defined. To re-run, get_margins'
                    'with `overwrite_margins_results` kwarg set '
                    'to True')
        return rst.utils.convert_df(rst.pyr.rpackages.emmeans.pairs(
            self.margins_results[margins_term]['r_margins']))


class BayesAnovaModel(AnovaModel):
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
            iterations=DEFAULT_ITERATIONS,
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

        super().__init__(**kwargs)

    def _set_formula_from_vars(self):
        frml = rst.utils.bayes_style_formula(self.dependent, self.between,
                                             self.within,
                                             subject={
                                                 False: None,
                                                 True: self.subject
                                             }[self.include_subject])
        self.formula = frml
        self._r_formula = rst.pyr.rpackages.stats.formula(frml)

    def _analyze(self):
        self._r_results = rst.pyr.rpackages.bayesfactor.anovaBF(
            self._r_formula,
            self._r_input_data,
            whichRandom=self.subject,
            whichModels=self.which_models,
            iterations=self.iterations,
            rscaleFixed=self.r_scale_fixed,
            rscaleRandom=self.r_scale_random,
            rscaleEffects=self.r_scale_effects,
            multicore=self.multi_core,
            method=self.method,
            progress=False
            # noSample=self.no_sample
        )


class BayesAnovaResults(AnovaResults):

    def _tidy_results(self):
        raise NotImplementedError
        #return rst.utils.tidy_bayes_factor_object(self.r_results)
        #results = rst.pyr.rpackages.bayesfactor.extractBF(
        #    self._r_results)
        #return rst.utils.convert_df(
        #    results, 'model')[['model', 'bf', 'error']]

    def get_margins(self):
        raise NotImplementedError


class Wilcoxon1SampleModel(T1SampleModel):

    def __init__(self, **kwargs):
        super().__init__(paired=True, **kwargs)

    """Mann-Whitney"""

    def _analyze(self):
        return Wilcoxon1SampleResults(rst.pyr.rpackages.stats.wilcox_test(
            x=self.x, mu=self.mu, alternative=self.tail,
            exact=True, correct=True
        ))


class Wilcoxon1SampleResults(T1SampleResults):
    pass


class Wilcoxon2SamplesModel(T2SamplesModel):

    # def __init__(self, **kwargs):
    #    super().__init__(**kwargs)

    def _analyze(self):
        return Wilcoxon2SamplesResults(rst.pyr.rpackages.stats.wilcox_test(
            x=self.x, y=self.y, paired=self.paired,
            alternative=self.tail,
            exact=True, correct=True
        ))


class Wilcoxon2SamplesResults(T2SamplesResults):
    pass


class KruskalWallisTestModel(AnovaModel):
    """Runs a Kruskal-Wallis test, similar to a non-parametric between
    subject anova.
    """

    def _validate_input_data(self):
        super()._validate_input_data()
        if len(self.between) != 1:
            raise RuntimeError('More than one between subject factors defined')
        if self.within != []:
            raise RuntimeError('A within subject factor has been defined')

    def _set_formula_from_vars(self):
        super()._set_formula_from_vars()
        self.formula = f'{self.dependent} ~ {self.between[0]}'
        self._r_formula = rst.pyr.rpackages.stats.formula(self.formula)

    def _analyze(self):
        return KruskalWallisTestResults(rst.pyr.rpackages.stats.kruskal_test(
            self._r_formula, data=self._r_input_data
        ))


class KruskalWallisTestResults(AnovaResults):
    pass
    # def _tidy_results(self):
    #     self._results = rst.utils.convert_df(
    #         rst.pyr.rpackages.generics.tidy(self.r_results))


class FriedmanTestModel(AnovaModel):
    """Runs a Friedman test, similar to a non-parametric within-subject anova.
    """

    def _validate_input_data(self):
        super()._validate_input_data()
        if len(self.within) != 1:
            raise RuntimeError('More than one within subject factors defined')
        if self.between != []:
            raise RuntimeError('A between subject factor has been defined')

    def _set_formula_from_vars(self):
        super()._set_formula_from_vars()
        self.formula = f'{self.dependent} ~ {self.between[0]}'
        self._r_formula = rst.pyr.rpackages.stats.formula(self.formula)

    def _analyze(self):
        return FriedmanTestResults(rst.pyr.rpackages.stats.friedman_test(
            self._r_formula, data=self._r_input_data
        ))


class FriedmanTestResults(AnovaResults):
    pass


class PairwiseComparisons:
    """To be implemented - runs all pairwise comparisons between groups
    in a dataframe defines by a variable. Used for exploration"""

    def __init__(self, data, correction_method):
        raise NotImplementedError
