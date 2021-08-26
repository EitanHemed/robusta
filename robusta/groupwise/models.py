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

# TODO - define __repr__ and __str__ for all classes

import typing
import warnings
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from . import results, reports
from .. import pyr
from ..misc import utils, formula_tools, base

import rpy2.robjects as ro

BF_COLUMNS = ['model', 'bf', 'error']

DEFAULT_TTEST_VARIABLE_NAMES = dict(zip(['DEPENDENT', 'SUBJECT', 'INDEPENDENT'],
                                        ['DEPENDENT', 'SUBJECT', 'INDEPENDENT']))

__all__ = [
    "Anova", "BayesAnova",
    "T1Sample", "T2Samples",
    "BayesT1Sample", "BayesT2Samples",
    "Wilcoxon1Sample", "Wilcoxon2Samples",
    "KruskalWallisTest", "FriedmanTest",
    "AlignedRanksTest"
]

DEFAULT_GROUPWISE_NULL_INTERVAL = (-np.inf, np.inf)
PRIOR_SCALE_STR_OPTIONS = ('medium', 'wide', 'ultrawide')
DEFAULT_ITERATIONS: int = 10000
DEFAULT_MU: float = 0.0

# TODO - find a better variable name
TEST_TAIL_DICT = {'x<y': 'less', 'x>y': 'greater', 'x=y': 'two.sided'}



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
    min_levels = 2

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
            max_levels=kwargs.get('max_levels', None),
            min_levels=kwargs.get('min_levels', 2)
        )

        self._results = None
        self._fitted = False

    def _pre_process(self):
        """
        Pre-process the input arguments prior to fitting the model.
        @return:
        """
        self._set_model_controllers()
        self._select_input_data()
        self._validate_input_data()
        self._transform_input_data()

    def _set_model_controllers(self):
        """
        Set the model controllers (formula and data-variables) based on the
         input - either parse formula from entered variables, or parse variables
         from entered formula.
        """
        if self.formula is None:
            self._build_from_variables()
        elif self.formula is not None:
            self._build_from_formula()

    def _build_from_variables(self):
        """In case we get only variables and no formula as input arguments,
        we build the model's formula based on the entered variables.
        """
        # Variables setting routine
        self._set_variables()
        # Build model formula from entered variables
        self._set_formula_from_vars()

    def _build_from_formula(self):
        """In case we get only formula and no variables as input arguments,
        we identify the model's variables based on the entered formula
        """
        # Parse models from entered formula
        self._r_formula = pyr.rpackages.stats.formula(
            self.formula)
        self._set_vars_from_formula()
        # Variables setting routine
        self._set_variables()

    def _set_variables(self):
        self.between = utils.to_list(self.between)
        self.within = utils.to_list(self.within)
        self.independent = self.between + self.within

    def _set_formula_from_vars(self):
        fp = formula_tools.FormulaParser(
            self.dependent, self.between, self.within, self.subject
        )
        self.formula = fp.get_formula()
        self._r_formula = pyr.rpackages.stats.formula(self.formula)

    def _set_vars_from_formula(self):
        vp = formula_tools.VariablesParser(self.formula)
        (self.dependent, self.between, self.within,
         self.subject) = vp.get_variables()

    def _convert_independent_vars_to_list(self, ind_vars: object) -> list:
        # if isinstance(self, T1SampleModel):
        #    return []
        return utils.to_list(ind_vars)

    def _select_input_data(self):
        try:
            data = self.data[
                utils.to_list([self.independent, self.subject,
                               self.dependent])].copy()
        except KeyError:
            _ = [i for i in utils.to_list(
                [self.independent, self.subject, self.dependent])
                 if i not in self.data.columns]
            raise KeyError(f"Variables not in data: {', '.join(_)}")
        self._input_data = data

    def _validate_input_data(self):
        # To test whether DV can be coerced to float
        utils.verify_float(self._input_data[self.dependent].values)

        # Verify ID variable uniqueness
        self._is_aggregation_required = (
                self._input_data.groupby(
                    [self.subject] + self.independent).size().any() > 1)

        # Make sure that the independent variables have enough levels, but
        # not too many
        _n_levels = (
            self._input_data[self.independent].apply(
                lambda s: s.unique().size))
        low = _n_levels.loc[_n_levels.values < self.min_levels]
        # It is likely that there will be a max_levels argument
        high = (_n_levels.loc[self.max_levels < _n_levels]
                if self.max_levels is not None else pd.Series())
        _s = ''
        if not low.empty:
            pass
            _s += ("The following variable:levels pairs have"
                   f" less than {self.min_levels} levels: {low.to_dict()}.\n")
        if not high.empty:
            _s += (
                "The following {variable:levels} pairs have more"
                f" than {self.max_levels} levels: {high.to_dict()}.")
        if _s:
            # TODO - this should be a ValueError
            warnings.warn(_s)

    def _transform_input_data(self):
        # Make sure we are handing R a DataFrame with factor variables,
        # as some packages do not coerce factor type.

        if self._is_aggregation_required:
            self._input_data = self._aggregate_data()

        self._r_input_data = utils.convert_df(self._input_data.copy())

    def _aggregate_data(self):
        return self._input_data.groupby(
            utils.to_list(
                [self.subject] + self.independent)).agg(
            self.agg_func).reset_index()

    def _analyze(self):
        pass

    def fit(self):
        """
        This method runs the model defined by the input.
        @return:
        """
        if self._fitted is True:
            raise RuntimeError("Model was already run. Use `reset()` method"
                               " prior to calling `fit()` again!")
        self._fitted = True
        self._pre_process()
        self._analyze()

    def reset(self, **kwargs):

        if self.formula is not None and kwargs.get('formula', None) is None:
            # We assume that the user aimed to specify the model using variables
            # so we need to remove a pre-existing formula so it would be updated
            self.formula = None  # re.sub('[()]', '', self.formula)

        # What else?
        vars(self).update(**kwargs)

        self._fitted = False
        self._results = None

    def report_table(self):
        return self._results.get_df()

    def report_text(self):
        # TODO - remake this into a visitor pattern
        visitor = reports.Reporter()
        return visitor.report_text(self)

    # def report_table(self):
    #     # TODO - remake this into a visitor pattern
    #     visitor = groupwise_reports.Reporter()
    #     return visitor.report_table(self._results)


class T2Samples(GroupwiseModel):
    """
    Run a frequentist dependent or independent-samples t-test.

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
                 assume_equal_variance: bool = False,
                 x=None,
                 y=None,
                 correct=False,
                 **kwargs):
        self.x = x
        self.y = y
        self.paired = paired

        self.tail = tail
        self._r_tail = self._set_r_tail()

        self.assume_equal_variance = assume_equal_variance
        kwargs['max_levels'] = 2
        kwargs['min_levels'] = 2

        # TODO - refactor this

        if self.x is None:
            if kwargs.get('formula') is None:
                # If independent is specified, try to set between/within based on independent and paired
                # If no independent variable is specified, try to use paired and between/within to infer independent

                if paired is None:
                    raise RuntimeError("`paired` specification missing")
                else:
                    if independent is None:
                        kwargs['independent'] = kwargs[{False: 'between', True: 'within'}[paired]]
                    if independent is not None:
                        kwargs[{False: 'between', True: 'within'}[paired]] = independent

            else:
                # What happens if there is formula and no specification of paired/unpaired
                # TODO - refactor this
                self.paired = bool(not re.search(r'\+\s*1\s*\|', kwargs['formula']))

        super().__init__(**kwargs)

    def _set_model_controllers(self):

        if self.x is not None:
            self.dependent, self.subject, self.independent = DEFAULT_TTEST_VARIABLE_NAMES.values()

            if self.data is not None:
                # We assume that x and y are two columns in the dataframe
                self.x, self.y = self.data[[self.x, self.y]].values.T

            # No data, we need to build one from from the x and y entered arrays
            # TODO - check that x and y are arrays/lists and not strings
            self.data = self._form_dataframe()

        super()._set_model_controllers()

    def _form_dataframe(self):

        dependent = np.hstack([self.x, self.y])
        x_len, y_len = len(self.x), len(self.y)
        subject = np.hstack([range(x_len), range(y_len)])
        independent = np.repeat(['X', 'Y'], repeats=[x_len, y_len])

        df = pd.DataFrame(data=zip(dependent, subject, independent),
                          columns=DEFAULT_TTEST_VARIABLE_NAMES)
        df[DEFAULT_TTEST_VARIABLE_NAMES['INDEPENDENT']] = (
            df[DEFAULT_TTEST_VARIABLE_NAMES['INDEPENDENT']].astype('category').values)

        return df

    def _select_input_data(self):
        super()._select_input_data()

        try:
            self.x, self.y = self._input_data.groupby(
                getattr(self, 'independent'))[getattr(self, 'dependent')].apply(
                lambda s: s.values)
        except ValueError as e:
            print("Input data has more than two categories. Use pd.Series.cat.remove_unused_categories"
                  "or change `independent` column into non-category.")
            raise e

    def _set_variables(self):
        if self.x is not None:
            if self.paired:
                self.within = self.independent
            else:
                self.between = self.independent
        super()._set_variables()

    def _set_r_tail(self):

        original_tail = re.sub(r'\s+', '', self.tail)

        if original_tail in TEST_TAIL_DICT.keys():
            r_tail = TEST_TAIL_DICT[original_tail]
        # The fallback is specifying the alternative based on the conventions in R
        elif original_tail in TEST_TAIL_DICT.values():
            r_tail = original_tail
        else:
            raise ValueError(f"{self.tail} is not a valid tail specification. "
                             f"Specify using one of the following - {', '.join(TEST_TAIL_DICT.keys())}"
                             f" or {' ,'.join(TEST_TAIL_DICT.values())}")

        return r_tail


    def _analyze(self):
        self._results = results.TTestResults(
            pyr.rpackages.stats.t_test(
                **{'x': self.x, 'y': self.y,
                   'paired': self.paired,
                   'alternative': self._r_tail,
                   'var.equal': self.assume_equal_variance,
                   })
        )

    def reset(self, **kwargs):
        self.x = None
        self.y = None

        super().reset()


class BayesT2Samples(T2Samples):
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
            if mu != 0:
                raise ValueError
        super().__init__(**kwargs)

    def _analyze(self):
        b = pyr.rpackages.BayesFactor.ttestBF(
            x=self.x,
            y=self.y,
            paired=self.paired,
            nullInterval=self.null_interval,
            iterations=self.iterations,
            posterior=self.sample_from_posterior,
            rscale=self.prior_scale,
            mu=self.mu
        )

        self._results = results.BayesResults(b,
                                             # TODO - refactor
                                             mode='posterior' if self.sample_from_posterior else 'bf'
                                             )


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
                 mu=DEFAULT_MU,
                 x=None,
                 y=None,
                 **kwargs):
        kwargs['paired'] = True
        kwargs['tail'] = tail
        kwargs['max_levels'] = 1
        kwargs['min_levels'] = 1

        kwargs['x'] = x
        kwargs['y'] = y

        self.mu = mu

        # self.max_levels = 1
        # self.min_levels = 1
        super().__init__(**kwargs)

    def _select_input_data(self):
        # Skip the method implemented by parent (T2Samples)
        GroupwiseModel._select_input_data(self)
        self.x = self._input_data[getattr(self, 'dependent')].values

        if self.mu is not None:
            self.y = self.mu

    def _analyze(self):
        self._results = results.TTestResults(pyr.rpackages.stats.t_test(
            x=self.x,
            mu=self.y,
            alternative=self._r_tail
            )
        )

    def _form_dataframe(self):
        dependent = self.x
        x_len = len(self.x)
        subject = range(x_len)
        independent = np.repeat('X', repeats=[x_len])

        df = pd.DataFrame(data=zip(dependent, subject, independent),
                          columns=DEFAULT_TTEST_VARIABLE_NAMES)

        # To sildence R's warnings
        df[DEFAULT_TTEST_VARIABLE_NAMES['INDEPENDENT']] = df[DEFAULT_TTEST_VARIABLE_NAMES['INDEPENDENT']].astype(
            'category').values

        return df


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
        b = pyr.rpackages.BayesFactor.ttestBF(
            x=self.x,
            y=pyr.rinterface.NULL,
            nullInterval=self.null_interval,
            iterations=self.iterations,
            posterior=self.sample_from_posterior,
            rscale=self.prior_scale,
            mu=self.mu
        )
        self._results = results.BayesResults(b,
                                             mode='posterior' if self.sample_from_posterior else 'bf')


class Wilcoxon1Sample(T1Sample):

    def __init__(self,
                 p_exact: bool = True,
                 p_correction: bool = True,
                 ci: int = 95,
                 **kwargs):
        self.p_exact = p_exact
        self.p_correction = p_correction
        self.ci = ci

        super().__init__(paired=True, **kwargs)

    """Mann-Whitney"""

    def _analyze(self):
        self._results = results.WilcoxonResults(
            pyr.rpackages.stats.wilcox_test(
                x=self.x, mu=self.y, alternative=self._r_tail,
                exact=self.p_exact, correct=self.p_correction,
                conf_int=self.ci
            ))


class Wilcoxon2Samples(T2Samples):

    def __init__(self, paired=None, p_exact: bool = True,
                 p_correction: bool = True, ci: int = 95, **kwargs):
        self.paired = paired
        self.p_exact = p_exact
        self.p_correction = p_correction
        self.ci = ci
        super().__init__(paired=self.paired, **kwargs)

    def _analyze(self):
        self._results = results.WilcoxonResults(
            pyr.rpackages.stats.wilcox_test(
                x=self.x, y=self.y, paired=self.paired,
                alternative=self._r_tail,
                exact=self.p_exact, correct=self.p_correction,
                conf_int=self.ci
            ))


class Anova(GroupwiseModel):
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
        self.sphericity_correction = kwargs.get(
            'sphericity_correction', 'none')

        super().__init__(**kwargs)

    def _analyze(self):
        # TODO - if we want to use aov_4 than the error term needs to be
        #  encapsulated in parentheses. Maybe write something that will add the
        #  parantheses automatically.
        self._results = results.AnovaResults(pyr.rpackages.afex.aov_4(
            formula=self._r_formula,
            # dependent=self.dependent,
            # id=self.subject,
            # within=self.within,
            # between=self.between,
            data=self._r_input_data,
            correction=self.sphericity_correction))
        # TODO: Add reliance on aov_ez aggregation functionality.
        # TODO: Add this functionality - sig_symbols=
        # pyr.vectors.StrVector(["", "", "", ""]))

    def get_margins(
            self,
            margins_terms: typing.List[str] = None,
            by_terms: typing.List[str] = None,
            ci: int = 95,
            overwrite: bool = False
    ):
        # TODO Look at this - emmeans::as_data_frame_emm_list
        # TODO - Documentation
        # TODO - Issue the 'balanced-design' warning etc.
        # TODO - allow for `by` option
        # TODO - implement between and within CI calculation

        _terms_to_test = np.array(
            utils.to_list([margins_terms,
                           [] if by_terms is None else by_terms])
        )

        if not all(
                term in self.report_table()['Term'].values for term in
                _terms_to_test):
            raise RuntimeError(
                f'Margins term: {[i for i in _terms_to_test]}'
                'not included in model')

        margins_terms = np.array(utils.to_list(margins_terms))

        if by_terms is None:
            by_terms = pyr.rinterface.NULL
        else:
            by_terms = np.array(utils.to_list(by_terms))

        _r_margins = pyr.rpackages.emmeans.emmeans(
            self._results.r_results,
            specs=margins_terms,
            type='response',
            level=ci,
            by=by_terms
        )

        return _r_margins

        margins = utils.convert_df(
            pyr.rpackages.emmeans.as_data_frame_emmGrid(
                _r_margins))

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
        return utils.convert_df(pyr.rpackages.emmeans.pairs(
            self.margins_results[margins_term]['r_margins']))


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
        r_scale_effects=pyr.rinterface.NULL,
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
            r_scale_effects=pyr.rinterface.NULL,
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
        self.formula = utils.bayes_style_formula_from_vars(
            self.dependent, self.between, self.within,
            subject={False: None, True: self.subject}[self.include_subject])
        self._r_formula = pyr.rpackages.stats.formula(self.formula)

    def _set_model_controllers(self):
        super()._set_model_controllers()

        # The formula needs to comply with BayesFactor formulas, which might
        #  leave out the participant term. Therefore we need to re-set the
        #  formula based on the entered/parsed variables
        # TODO - consider leaving this or writing a more sophisticated solution.
        self._set_formula_from_vars()

    def _transform_input_data(self):
        # TODO - check if this can be permanently removed
        self._input_data.loc[:, self.independent] = self._input_data[
            self.independent].astype('category')

        super(BayesAnova, self)._transform_input_data()

    def _analyze(self):
        b = pyr.rpackages.BayesFactor.anovaBF(
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
        self._results = results.BayesResults(b)


class KruskalWallisTest(Anova):
    """Runs a Kruskal-Wallis test, similar to a non-parametric between
    subject anova.
    """

    def _validate_input_data(self):
        super()._validate_input_data()
        if len(self.between) != 1:
            raise ValueError('More than one between subject factors defined')
        if self.within != []:
            raise ValueError('A within subject factor has been defined')

    def _set_formula_from_vars(self):
        self.formula = utils.bayes_style_formula_from_vars(
            self.dependent, self.between, self.within,
            subject=None)
        self._r_formula = pyr.rpackages.stats.formula(self.formula)

    def _set_model_controllers(self):
        super()._set_model_controllers()

        # The formula needs to comply with BayesFactor formulas, which might
        #  leave out the participant term. Therefore we need to re-set the
        #  formula based on the entered/parsed variables
        # TODO - consider leaving this or writing a more sophisticated solution.
        self._set_formula_from_vars()

    def _analyze(self):
        self._results = results.KruskalWallisTestResults(
            pyr.rpackages.stats.kruskal_test(
                self._r_formula, data=self._r_input_data
            ))


class FriedmanTest(Anova):
    """Runs a Friedman test, similar to a non-parametric within-subject anova.
    """

    def _validate_input_data(self):
        super()._validate_input_data()
        if len(self.within) != 1:
            raise RuntimeError('More than one within subject factors defined')
        if self.between != []:
            raise RuntimeError('A between subject factor has been defined')

    def _set_formula_from_vars(self):
        self.formula = utils.bayes_style_formula_from_vars(
            self.dependent, self.between, self.within,
            subject=None)
        self._r_formula = pyr.rpackages.stats.formula(self.formula)

    def _set_model_controllers(self):
        super()._set_model_controllers()

        # The formula needs to comply with BayesFactor formulas, which might
        #  leave out the participant term. Therefore we need to re-set the
        #  formula based on the entered/parsed variables
        # TODO - consider leaving this or writing a more sophisticated solution.
        self._set_formula_from_vars()

    def _analyze(self):
        self._results = results.FriedmanTestResults(
            pyr.rpackages.stats.friedman_test(
                self._r_formula, data=self._r_input_data
            ))


class AlignedRanksTest(Anova):
    """N-way non-parametric Anova"""

    def _analyze(self):
        self._results = results.AlignedRanksTestResults(
            pyr.rpackages.ARTool.art(
                data=self._r_input_data,
                formula=self._r_formula
            )
        )


class PairwiseComparisons:
    """To be implemented - runs all pairwise comparisons between groups
    in a dataframe defines by a variable. Used for exploration"""

    def __init__(self, data, correction_method):
        raise NotImplementedError
