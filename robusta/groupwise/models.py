"""
Models for fitting statistical models where central tendencies of groups (eg., mean, median) are compared.
"""

# TODO add examples to individual classes
# TODO - define __repr__ and __str__ for all classes

import typing
import collections
import warnings
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
import numpy.typing as npt  # Odd but this seems to be the cannonical way, in NumPy docs.

from . import results, reports
from .. import pyr
from ..misc import utils, formula_tools, base

__all__ = [
    "Anova", "BayesAnova",
    "T1Sample", "T2Samples",
    "BayesT1Sample", "BayesT2Samples",
    "Wilcoxon1Sample", "Wilcoxon2Samples",
    "KruskalWallisTest", "FriedmanTest",
    "AlignedRanksTest"
]

# TODO - find a better variable name
ROBUSTA_TEST_TAILS_SPECS = ['x<y', 'x>y', 'x!=y']
R_FREQUENTIST_TEST_TAILS_SPECS = dict(zip(ROBUSTA_TEST_TAILS_SPECS, ['less', 'greater', 'two.sided']))
R_BAYES_TEST_TAILS_DICT = dict(zip(ROBUSTA_TEST_TAILS_SPECS, [[-np.inf, 0], [0, np.inf], [-np.inf, np.inf]]))

DEFAULT_GROUPWISE_TAIL = 'x!=y'
DEFAULT_GROUPWISE_NULL_INTERVAL = (-np.inf, np.inf)
PRIOR_SCALE_STR_OPTIONS = ('medium', 'wide', 'ultrawide')
DEFAULT_ITERATIONS: int = 10000
DEFAULT_MU: float = 0.0

BF_COLUMNS = ['model', 'bf', 'error']

DEFAULT_TTEST_VARIABLE_NAMES = dict(zip(['DEPENDENT', 'SUBJECT', 'INDEPENDENT'],
                                        ['DEPENDENT', 'SUBJECT', 'INDEPENDENT']))

DEFAULT_POPULATION_MEAN = 0


@dataclass
class GroupwiseModel(base.BaseModel):
    data: typing.Type[pd.DataFrame]
    subject: typing.Union[str, None]
    between = typing.Union[str, list, None]
    within = typing.Union[str, list, None]
    dependent: typing.Union[str, None]
    formula: typing.Union[str, None]
    agg_func: typing.Union[str, typing.Callable]
    _max_levels = None
    _min_levels = 2

    """
    A basic class to handle pre-requisites of T-Tests and ANOVAs.


    Parameters
    ----------

    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling
    fit : bool, optional
        Whether to run the statistical test upon object creation. Default is True. 
        
    Methods
    -------
    fit()
        Run the corresponding statistical test. 
    reset(**kwargs)
        Reset the Model object and remove the current test results. `kwargs` will be used to update the object 
        attributes, e.g., model's `formula`. 
    report_table()
        Return a tabular report formatted report of the statistical test's results.  
    report_text()
        Return a string reporting the results of the statistical test. 
    """

    def __init__(
            self,
            formula: typing.Optional[str] = None,
            data: pd.DataFrame = None,
            dependent: str = None,
            between: typing.Optional[str] = None,
            within: typing.Optional[str] = None,
            subject: str = None,
            na_action: typing.Optional[str] = None,
            agg_func: typing.Optional[
                typing.Union[str, typing.Callable]] = np.mean,
            fit=True,
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
            _max_levels=kwargs.get('_max_levels', None),
            _min_levels=kwargs.get('_min_levels', 2)
        )

        self._results = None
        self._fitted = False

        if fit:
            self.fit()

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
        low = _n_levels.loc[_n_levels.values < self._min_levels]
        # It is likely that there will be a _max_levels argument
        high = (_n_levels.loc[self._max_levels < _n_levels]
                if self._max_levels is not None else pd.Series())
        _s = ''
        if not low.empty:
            pass
            _s += ("The following variable:levels pairs have"
                   f" less than {self._min_levels} levels: {low.to_dict()}.\n")
        if not high.empty:
            _s += (
                "The following {variable:levels} pairs have more"
                f" than {self._max_levels} levels: {high.to_dict()}.")
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
        Fit the statistical model.

        Returns
        -------
        None

        Raises
        ------
        RunTimeError
            If model was already fitted, raises RunTimeError.
        """
        if self._fitted is True:
            raise RuntimeError("Model was already run. Use `reset()` method"
                               " prior to calling `fit()` again!")
        self._fitted = True
        self._pre_process()
        self._analyze()

    def reset(self, refit=True, **kwargs):
        """
        Updates the Model object state and removes current test results.

        Parameters
        ----------
        refit
            Whether to fit the statistical test after resetting parameters. Default is True.
        kwargs
            Any keyword arguments of parameters to be updated.

        Returns
        -------
        None
        """

        if self.formula is not None and kwargs.get('formula', None) is None:
            # We assume that the user aimed to specify the model using variables
            # so we need to remove a pre-existing formula so it would be updated
            self.formula = None  # re.sub('[()]', '', self.formula)

        # What else?
        vars(self).update(**kwargs)

        self._fitted = False
        self._results = None

        if refit is True:
            self.fit()

    def report_table(self):
        """

        Returns
        -------
        pd.DataFrame
            Formatted tabular report of the statistical test results.
        """
        return self._results.get_df()

    def report_text(self, ):
        """
        Returns
        -------
        str
            Formatted string report of statistical test results.
        """
        # TODO - remake this into a visitor pattern
        visitor = reports.Reporter()
        return visitor.report_text(self)

    # def report_table(self):
    #     # TODO - remake this into a visitor pattern
    #     visitor = groupwise_reports.Reporter()
    #     return visitor.report_table(self._results)

class T2Samples(GroupwiseModel):
    """Run a two-samples t-test, either dependent or independent.

    Parameters
    ----------
    x, y: keys in data or NumPy array of values, optional
        x and y can be used to specify. If str, both have to be keys to columns in the dataframe (`data`
        argument). If array-like, have to contain only objects that can be coerced into numeric. If not
        specified they are inferred based on the following arguments `formula`, and `between` or `within` (in
        this order).
    paired : bool
        Whether the test is dependent/paired-samples (True) or independent-samples (False).
        If not specified, robusta will try to infer based on other input arguments - `formula`, `indpependent`,
         `between` and `within` (in this order). Default is True.
    tail: str, optional
        Direction of the tested alternative hypothesis. Optional values are 'x!=y' (Two sided test; aliased
        by 'two.sided'), 'x<y' (lower tail; aliased by 'less') 'x>y' (upper tail; aliased by 'greater').
        Whitespace characters in the input are ignored. Default value is 'x != y'.
    assume_equal_variance : bool, default: True
        Whether to assume that the two samples have equal variance (Applicable only to independent samples
        t-test). If True runs regular two-sample, if False runs Welch two-sample test. Default is True.
    ci : int
        Width of confidence interval around the sample mean difference. Float between 0 and 100.
        Default value is 95.
    independent : str, optional
        The name of the column identifying the independent variable in the data. The column could be either
        numeric or object, but can contain up to two unique values. Alias for `within` for paired, `between`
        for unpaired.
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling
    fit : bool, optional
        Whether to run the statistical test upon object creation. Default is True.


    kwargs: mapping, optional
        Keyward arguments to be passed down to robusta.groupwise.models.GroupwiseModel.

    Notes
    -----
    R function - t.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/t.test
    """

    def __init__(self,
                 paired: bool = None,
                 x: typing.Union[str, npt.ArrayLike] = None,
                 y: typing.Union[str, npt.ArrayLike] = None,
                 independent: bool = None,
                 tail: str = 'x!=y',
                 assume_equal_variance: bool = False,
                 ci=95,
                 **kwargs):
        self.x = x
        self.y = y
        self.paired = paired
        # TODO - allow specification by group key (e.g., 'treatment>control'
        self.tail = tail
        self.ci = ci
        self._r_ci = self.ci / 100  # R uses a float in the range of 0 to 1 here.

        # TODO - refactor this
        if not hasattr(self, '_tails_dict'):
            self._tails_dict = R_FREQUENTIST_TEST_TAILS_SPECS

        self.assume_equal_variance = assume_equal_variance
        kwargs['_max_levels'] = 2
        kwargs['_min_levels'] = 2

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

    def _pre_process(self):
        self._set_r_tail()
        super()._pre_process()

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

        if original_tail in self._tails_dict.keys():
            r_tail = self._tails_dict[original_tail]
        # The fallback is specifying the alternative based on the conventions in R
        elif original_tail in self._tails_dict.values():
            r_tail = original_tail
        else:
            raise ValueError(f"{self.tail} is not a valid tail specification. "
                             f"Specify using one of the following - {list(self._tails_dict.keys())}"
                             f" or {list(self._tails_dict.values())}")

        self._r_tail = r_tail

    def _analyze(self):
        self._results = results.TTestResults(
            pyr.rpackages.stats.t_test(
                **{'x': self.x, 'y': self.y,
                   'paired': self.paired,
                   'alternative': self._r_tail,
                   'var.equal': self.assume_equal_variance,
                   'conf.level': self._r_ci
                   })
        )

    def reset(self, **kwargs):
        self.x = None
        self.y = None

        super().reset(**kwargs)

    def report_text(self, effect_size=False):
        """

        Parameters
        ----------
        effect_size, bool : optional
            Should Cohen's d effect size should be reported as well. Default is False.

        Returns
        -------

        """
        visitor = reports.Reporter()
        return visitor.report_text(self, effect_size=effect_size)


class BayesT2Samples(T2Samples):
    """
    Run a Bayesian two-samples t-test, either dependent or independent.

    Parameters
    ----------

    x, y: keys in data or NumPy array of values, optional
        x and y can be used to specify. If str, both have to be keys to columns in the dataframe (`data`
        argument). If array-like, have to contain only objects that can be coerced into numeric. If not
        specified they are inferred based on the following arguments `formula`, and `between` or `within` (in
        this order).
    paired : bool
        Whether the test is dependent/paired-samples (True) or independent-samples (False).
        If not specified, robusta will try to infer based on other input arguments - `formula`, `indpependent`,
         `between` and `within` (in this order). Default is True.
    tail: str, optional
        Direction of the tested alternative hypothesis. Optional values are 'x!=y' (Two sided test; aliased
        by 'two.sided'), 'x<y' (lower tail; aliased by 'less') 'x>y' (upper tail; aliased by 'greater').
        Whitespace characters in the input are ignored. Default value is 'x != y'.
    ci : int
        Width of confidence interval around the sample mean difference. Float between 0 and 100.
        Default value is 95.
    independent : str, optional
        The name of the column identifying the independent variable in the data. The column could be either
        numeric or object, but can contain up to two unique values. Alias for `within` for paired, `between`
        for unpaired.
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.
    fit : bool, optional
        Whether to run the statistical test upon object creation. Default is True.
    scale_prior : float, optional
        Controls the scale (width) of the prior distribution. Default value is 1.0 which yields a standard Cauchy
        prior. It is also possible to pass 'medium', 'wide' or 'ultrawide' as input arguments instead of a
        float (matching the values of :math:`\\frac{\\sqrt{2}}{2}, 1, \\sqrt{2}`, respectively).
        TODO - limit str input values.
    sample_from_posterior : bool, optional
        If True return samples from the posterior, if False returns Bayes factor. Default is False.
    iterations : int, optional
        Number of samples used to estimate Bayes factor or posterior. Default is 1000.
    mu :  float, optional
        The hypothesized mean of the differences between the samples, default is 0.

    kwargs : mapiing, optional
        Keyword arguments passed down to robusta.groupwise.models.T2Samples.

    Notes
    -----
    R function - ttestBF: https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/ttestBF
    from the BayesFactor[1]_ package

    References
    ----------
    .. [1] Morey, R. D., Rouder, J. N., Jamil, T., & Morey, M. R. D. (2015). Package ‘bayesfactor’.
    """

    def __init__(
            self,
            null_interval=None,  # DEFAULT_GROUPWISE_NULL_INTERVAL,
            prior_scale: str = 'medium',
            sample_from_posterior: bool = False,
            iterations: int = DEFAULT_ITERATIONS,
            mu: float = 0,
            **kwargs):
        self.null_interval = np.array(null_interval)
        self.prior_scale = prior_scale
        self.sample_from_posterior = sample_from_posterior
        self.iterations = iterations
        self.mu = mu

        self._tails_dict = R_BAYES_TEST_TAILS_DICT

        # TODO - see what this
        # if self.paired is False:
        #     if self.mu != 0:
        #         raise ValueError

        super().__init__(**kwargs)

        if self.null_interval != None:
            warnings.warn("null_interval will soon be deprecated, use tail", warnings.DeprecationWarning)

    def _analyze(self):
        b = pyr.rpackages.BayesFactor.ttestBF(
            x=self.x,
            y=self.y,
            paired=self.paired,
            nullInterval=self._r_tail,
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
    Run a Student one-sample t-test.

    Parameters
    ----------
    x: key in data or NumPy array of values, optional
        `x` can be used to specify. If str, `x` have to be key to column in `data`. If array-like, have to contain
        only objects that can be coerced into numeric. If not specified they are inferred based on the following
        arguments `formula`, and `between` or `within` (in this order).
    mu :  float, optional
        Value of the population to compare the sample (`x`) to. Default is None. `y` is an alias.
    y :  float, optional
        `y` is an alias of `mu`, superseded by `mu`.
    tail: str, optional
        Direction of the tested alternative hypothesis. Optional values are 'x!=y' (Two sided test; aliased
        by 'two.sided'), 'x<y' (lower tail; aliased by 'less') 'x>y' (upper tail; aliased by 'greater').
        Whitespace characters in the input are ignored. Default value is 'x != y'.
    ci : int
        Width of confidence interval around the sample mean difference. Float between 0 and 100.
        Default value is 95.
    independent : str, optional
        The name of the column identifying the independent variable in the data. The column could be either
        numeric or object, but can contain up to two unique values. Alias for `within` for paired, `between`
        for unpaired.
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.


    Notes
    -----
    R function stats::t.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/t.test

    If `x` is left `None`, the sample values will be the values under the column in `data` that is specified by
    `independent`.
    """

    def __init__(self,
                 tail: str = 'x!=y',
                 x=None,
                 mu=None,
                 y=None,
                 **kwargs):
        kwargs['paired'] = True
        kwargs['tail'] = tail
        kwargs['_max_levels'] = 1
        kwargs['_min_levels'] = 1
        kwargs['x'] = x
        kwargs['y'] = y
        self.mu = mu
        super().__init__(**kwargs)

    def _select_input_data(self):
        # Skip the method implemented by parent (T2Samples)
        GroupwiseModel._select_input_data(self)

        if self.x is None:
            self.x = self._input_data[getattr(self, 'dependent')].values

        if self.mu is None and self.y is not None:
            self.mu = self.y

    def _analyze(self):
        self._results = results.TTestResults(pyr.rpackages.stats.t_test(
            x=self.x, mu=self.mu, alternative=self._r_tail))

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
    Run a Bayesian One-sample t-test.

    Parameters
    ----------
    x: key in data or NumPy array of values, optional
        `x` can be used to specify. If str, `x` have to be key to column in `data`. If array-like, have to contain
        only objects that can be coerced into numeric. If not specified they are inferred based on the following
        arguments `formula`, and `between` or `within` (in this order).
    mu : float, optional
        The null (H0) predicted value for the mean difference compared with the population's mean (mu), in
        un-standardized effect size (e.g., raw measure units). Default value is 0.
    y :  float, optional
        `y` is an alias of `mu`, superseded by `mu`.
    tail: str, optional
        Direction of the tested alternative hypothesis. Optional values are 'x!=y' (Two sided test; aliased
        by 'two.sided'), 'x<y' (lower tail; aliased by 'less') 'x>y' (upper tail; aliased by 'greater').
        Whitespace characters in the input are ignored. Default value is 'x != y'.
    ci : int
        Width of confidence interval around the sample mean difference. Float between 0 and 100.
        Default value is 95.
    independent : str, optional
        The name of the column identifying the independent variable in the data. The column could be either
        numeric or object, but can contain up to two unique values. Alias for `within` for paired, `between`
        for unpaired.
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.
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

    Notes
    -----
    R function anovaBF: https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/anovaBF
    from the BayesFactor packages [1]_.

    References
    ----------
    .. [1] Morey, R. D., Rouder, J. N., Jamil, T., & Morey, M. R. D. (2015). Package ‘bayesfactor’.
    """

    def __init__(
            self,
            # null_interval=(-np.inf, np.inf),
            null_interval=None,
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
        self._tails_dict = R_BAYES_TEST_TAILS_DICT

        super().__init__(**kwargs)

        if self.null_interval != None:
            warnings.warn("null_interval will soon be deprecated, use tail", warnings.DeprecationWarning)

    def _analyze(self):
        b = pyr.rpackages.BayesFactor.ttestBF(
            x=self.x,
            y=pyr.rinterface.NULL,
            nullInterval=self._r_tail,
            iterations=self.iterations,
            posterior=self.sample_from_posterior,
            rscale=self.prior_scale,
            mu=self.mu
        )
        self._results = results.BayesResults(b,
                                             mode='posterior' if self.sample_from_posterior else 'bf')


class Wilcoxon2Samples(T2Samples):
    """
    Run a Wilcoxon rank sum test - non-parametric independent-samples t-test alternative.

    Parameters
    ----------
    x, y: keys in data or NumPy array of values, optional
        x and y can be used to specify. If str, both have to be keys to columns in the dataframe (`data`
        argument). If array-like, have to contain only objects that can be coerced into numeric. If not
        specified they are inferred based on the following arguments `formula`, and `between` or `within` (in
        this order).
    paired : bool
        Whether the test is dependent/paired-samples (True) or independent-samples (False).
        If not specified, robusta will try to infer based on other input arguments - `formula`, `indpependent`,
         `between` and `within` (in this order). Default is True.
    tail: str, optional
        Direction of the tested alternative hypothesis. Optional values are 'x!=y' (Two sided test; aliased
        by 'two.sided'), 'x<y' (lower tail; aliased by 'less') 'x>y' (upper tail; aliased by 'greater').
        Whitespace characters in the input are ignored. Default value is 'x != y'.
    assume_equal_variance : bool, default: True
        Whether to assume that the two samples have equal variance (Applicable only to independent samples
        t-test). If True runs regular two-sample, if False runs Welch two-sample test. Default is True.
    ci : int
        Width of confidence interval around the sample mean difference. Float between 0 and 100.
        Default value is 95.
    independent : str, optional
        The name of the column identifying the independent variable in the data. The column could be either
        numeric or object, but can contain up to two unique values. Alias for `within` for paired, `between`
        for unpaired.
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling
    fit : bool, optional
        Whether to run the statistical test upon object creation. Default is True.
    p_exact :  bool, optional
        Whether to compute exact p-value or approximate it. Default is False.
    p_correction :  bool, optional
        FILL THIS. Default is False.


    Notes
    -----
    Calling Wilcoxon2Samples with `Wilcoxon1Sample(x=x, y=y, paired=True)` is the same as
     `Wilcoxon1Sample(x=x-y, y=0)`.

    R function - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/wilcox.test
    """

    def __init__(self, p_exact: bool = True,
                 p_correction: bool = True, **kwargs):
        self.p_exact = p_exact
        self.p_correction = p_correction
        super().__init__(**kwargs)

    def _analyze(self):
        self._results = results.WilcoxonResults(
            pyr.rpackages.stats.wilcox_test(
                x=self.x, y=self.y, paired=self.paired,
                alternative=self._r_tail,
                exact=self.p_exact, correct=self.p_correction,
                conf_int=self.ci
            ))


class Wilcoxon1Sample(T1Sample):
    """
    Run a Wilcoxon signed rank test - non-parametric one-sample t-test alternative.

    Parameters
    ----------
    x: key in data or NumPy array of values, optional
        `x` can be used to specify. If str, `x` have to be key to column in `data`. If array-like, have to contain
        only objects that can be coerced into numeric. If not specified they are inferred based on the following
        arguments `formula`, and `between` or `within` (in this order).
    mu :  float, optional
        Value of the population to compare the sample (`x`) to. Default is None. `y` is an alias.
    y :  float, optional
        `y` is an alias of `mu`, superseded by `mu`.
    tail: str, optional
        Direction of the tested alternative hypothesis. Optional values are 'x!=y' (Two sided test; aliased
        by 'two.sided'), 'x<y' (lower tail; aliased by 'less') 'x>y' (upper tail; aliased by 'greater').
        Whitespace characters in the input are ignored. Default value is 'x != y'.
    ci : int
        Width of confidence interval around the sample mean difference. Float between 0 and 100.
        Default value is 95.
    independent : str, optional
        The name of the column identifying the independent variable in the data. The column could be either
        numeric or object, but can contain up to two unique values. Alias for `within` for paired, `between`
        for unpaired.
    data :  pd.DataFrame
        Containing the subject, dependent and independent variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of
        (dependent ~ between + within | subject). If used, the parsed formula will overrides the following
        arguments `dependent`, `between`, `within` and `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.
    p_exact :  bool, optional
        Whether to compute exact p-value or approximate it. Default is False.
    p_exact :  bool, optional
        FILL THIS. Default is False.

    Notes
    -----
    R function - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/wilcox.test
    """

    def __init__(self,
                 p_exact: bool = False,
                 p_correction: bool = True,
                 **kwargs):
        self.p_exact = p_exact
        self.p_correction = p_correction

        super().__init__(paired=True, **kwargs)

    def _analyze(self):
        self._results = results.WilcoxonResults(
            pyr.rpackages.stats.wilcox_test(
                x=self.x, mu=self.mu, alternative=self._r_tail,
                exact=self.p_exact, correct=self.p_correction,
                conf_int=self.ci
            ))


class Anova(GroupwiseModel):
    """
    Run a between, within or mixed frequentist ANOVA.

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the `subject`, `dependent`, `between' and `within` variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of (dependent ~ between + within | subject).
        If used, the parsed formula will overrides the following arguments `dependent`, `between`, `within` and
        `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.
    effect_size: str, optional
        Optional values are 'ges', (:math:`generalized-{\\eta}^2`) 'pes' (:math:`partial-{\\eta}^2`) or 'none'.
        Default value is 'pes'.
    correction: str, optional
        Sphericity correction method. Possible values are "GG" (Greenhouse-Geisser), "HF" (Hyunh-Feldt) or
        "none". Default is 'none'.

    Notes
    -----
    R function aov_4: https://www.rdocumentation.org/packages/afex/versions/0.27-2/topics/aov_car from the
    afex package [1]_

    References
    ----------
    .. [1] Singmann, H., Bolker, B., Westfall, J., Aust, F., & Ben-Shachar, M. S. (2015). afex: Analysis of factorial experiments. R package version 0.13–145.
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

        raise RuntimeError("The get_margins function is currently under development.")

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

    def report_text(self, as_list=False):
        # TODO - remake this into a visitor pattern
        visitor = reports.Reporter()
        return visitor.report_text(self, as_list=as_list)


class BayesAnova(Anova):
    # TODO - Formula specification will be using the lme4 syntax and variables
    #  will be parsed from it
    """
    Run a between, within or mixed Bayesian ANOVA.

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the `subject`, `dependent`, `between' and `within` variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of (dependent ~ between + within | subject).
        If used, the parsed formula will overrides the following arguments `dependent`, `between`, `within` and
        `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.
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
        Number of iterations used to estimate Bayes factor. Default value is 10000.
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
    multi_core : bool, optional
        Whether to use multiple cores for estimation. Not available on Windows. Default value is False.
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

    Notes
    -----
    R function anovaBF: https://www.rdocumentation.org/packages/BayesFactor/versions/0.9.12-4.2/topics/anovaBF
    from the BayesFactor packages [1]_.

    References
    ----------
    .. [1] Morey, R. D., Rouder, J. N., Jamil, T., & Morey, M. R. D. (2015). Package ‘bayesfactor’.
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


# TODO - merge KruskalWallisTest and Friedman test into a non-parametric Anova class

class KruskalWallisTest(Anova):
    """Runs a Kruskal-Wallis test, similar to a non-parametric between subject anova for one variable.

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the `subject`, `dependent` and `between' and variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of (dependent ~ between + 1 | subject).
        If used, the parsed formula will overrides the following arguments `dependent`, `between`, and
        `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.

    Raises
    ------
    ValueError
        If a within-subject variable has been specified using `within` argument or formula.
    ValueError
        If more than one between-subjects variable was specified using `between` argument or formula.

    Notes
    -----
    R function - kruskal.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/kruskal.test

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
    """Runs a Friedman test, similar to a non-parametric between subject anova for one variable.

    Parameters
    ----------

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the `subject`, `dependent` and `within` variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of (dependent ~ within | subject).
        If used, the parsed formula will overrides the following arguments `dependent`, `within` and
        `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.

    Raises
    ------
    ValueError
        If a within-subject variable has been specified using `within` argument or formula.
    ValueError
        If more than one between-subjects variable was specified using `between` argument or formula.

    Notes
    -----
    R function - friedman.test: https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/friedman.test
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
    """Run Aligned Ranks Transform - a non-parametric n-way ANOVA for a between, within or mixed design.

    Parameters
    ----------

    Parameters
    ----------
    data :  pd.DataFrame
        Containing the `subject`, `dependent`, `between' and `within` variables as columns.
    formula : str, optional
        An R-style formula describing the statistical model. In the form of (dependent ~ between + within | subject).
        If used, the parsed formula will overrides the following arguments `dependent`, `between`, `within` and
        `subject`.
    dependent : key in data, optional
        The name of the column identifying the dependent variable (i.e., response variable) in the data. The
        column data type should be numeric or a string that can be coerced to numeric.
        Overriden by `formula` if specified. Required if `formula` is not specified.
    between : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable (i.e., predictor variable) in the data.
        Identifies variables that are manipulated between different `subject` units (i.e., exogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `within` is
        is specified.
    within : key(s) in data (str or array-like), optional
        The name of the column identifying the independent variable in the data (i.e., predictor variable). The
        Identifies variables that are manipulated within different `subject` units (i.e., endogenous variable).
        Overriden by `formula` if specified. Not required if `formula` is not specified, given `between` is
        is specified.
    subject : str or key in data, optional
        The name of the column identifying the sampling unit in the data (i.e., subject).
        Overriden by `formula` if specified. Required if `formula` is not specified.
    agg_func : str (name of pandas aggregation function) or callable, optional
        Specified how to aggregate observations within sampling.

    Notes
    -----
    R function - https://www.rdocumentation.org/packages/ART/versions/1.0/topics/aligned.rank.transform from the
    ARTool package [1]_

    References
    ----------
    .. [1] Kay M and Wobbrock J (2021). ARTool: Aligned Rank Transform for Nonparametric Factorial ANOVAs. R package version 0.11.0, https://github.com/mjskay/ARTool. DOI: 10.5281/zenodo.594511.
    """

    def _analyze(self):
        self._results = results.AlignedRanksTestResults(
            pyr.rpackages.ARTool.art(
                data=self._r_input_data,
                formula=self._r_formula
            )
        )

