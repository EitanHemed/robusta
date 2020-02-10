import numpy as np
import robusta as rst


class BayesTTest(rst.TTest):
    def __init__(
            self,
            null_interval: int = (-np.inf, np.inf),
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
        try:
            return rst.pyr.rpackages.base.data_frame(rst.pyr.rpackages.bayesfactor.ttestBF(
                x=self.x, y=self.y, paired=self.paired, nullInterval=self.null_interval,
                iterations=self.iterations, posterior=self.sample_from_posterior,
                rscalse=self.prior_r_scale, mu=self.mu))
        except rst.pyr.rinterface.RRuntimeError as e:
            if "data are essentially constant" in e:
                return np.nan


class BayesAnova(rst.Anova):
    # TODO - if formula - ignore subject, between, within, etc. 

    def __init__(
            self,
            data,
            dependent='',
            between='',
            within='',
            subject='',
            formula=None,
            which_models="withmain", # TODO Find out all the possible options
            iterations=10000,
            r_scale_fixed="medium", # TODO Find out all the possible options
            r_scale_random="nuisance",
            r_scale_effects=rst.pyr.rinterface.NULL,
            multi_core=False,
            method="auto",
            no_sample=False,
            **kwargs):
        self.formula = formula
        self.subject = subject
        self.which_models = which_models
        self.iterations = iterations
        self.r_scale_fixed = r_scale_fixed
        self.r_scale_random = r_scale_random
        self.r_scale_effects = r_scale_effects
        self.multi_core = multi_core
        self.method = method
        self.no_sample = no_sample
        # This is a scaffold. It will be removed when we would have formula compatibility for all tests.
        super().__init__(special_inits=self._finalize_inits, **kwargs)

    def _finalize_inits(self):
        if self.formula is not None:
            self.formula = rst.utils.build_anova_formula(
                dependent=self.dependent,
                between=self.between,
                within=self.within,
                subejct=self.subject
            )

    def _run_analysis(self):
        return rst.pyr.rpackages.bayesfactor.anovaBF(
            self.formula,
            self.data,
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