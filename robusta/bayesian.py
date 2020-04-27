import pandas as pd
import numpy as np
import robusta as rst
from pandas_flavor import register_dataframe_accessor

class _BayesTwoSampleTTest(rst.IndependentSamplesTTest):

    def __init__(
            self,
            null_interval = (-np.inf, np.inf),
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
        kwargs['paired'] = False
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

class IndBayesTTest(_BayesTwoSampleTTest):

    def __init__(self, **kwargs):
        kwargs['paired'] = False
        super().__init__(**kwargs)


class DepBayesTTest(_BayesTwoSampleTTest):

    def __init__(self, **kwargs):
        kwargs['paired'] = True
        super().__init__(**kwargs)


class OneSampleBayesTTest(rst.OneSampleTTest):
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

class _BayesAnova(rst.ttest_and_anova._Anova):
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

@register_dataframe_accessor("anova_bayes_ws")
class AnovaBayesWS(_BayesAnova):
    def __init__(self, **kwargs):
        if 'between' in kwargs:
            raise ValueError(
                "A Within-Subject Anova was selected. Either remove `between` or"
                " use BetweenSubjectsAnova / MixedAnova")
        super().__init__(**kwargs)

@register_dataframe_accessor("anova_bayes_bs")
class AnovaBayesBS(_BayesAnova):
    def __init__(self, **kwargs):
        if 'within' in kwargs:
            raise ValueError(
                "A Within-Subject Anova was selected. Either remove `between` or"
                " use BetweenSubjectsAnova / MixedAnova")
        super().__init__(**kwargs)

@register_dataframe_accessor("anova_bayes_mixed")
class AnovaBayesMixed(_BayesAnova):
    def __init__(self, **kwargs):
        if not ('between' in kwargs and 'within' in kwargs):
            raise ValueError(
                "A Mixed Anova was selected. Specify both `between`"
                "and `within`")
        super().__init__(**kwargs)

