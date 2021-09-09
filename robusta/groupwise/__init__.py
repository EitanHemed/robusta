"""
`groupwise` contains classes used to run statistical tests in which a central tendency measure (e.g., mean) of smples
or populations is compared:

- Frequentist, parametric tests:
    - Anova: Between/Within/Mixed n-way ANOVAs
    - T2Samples, T1Sample: Independent\Dependent samples t-test and One-Sample t-tests

- Frequentist, non-parametric tests:
    - Wilcoxon2Samples, Wilcoxon1Sample: Non-parametric equivalents of Independent samples or One-Sample t-tests
    - KruskalWallisTest, FriedmanTest: Non-parametric equivalents of 1-way between and within subject ANOVAs
    - AlignedRanksTest: Non-parametric equivalents of n-way between, within or mixed ANOVA.

- Bayesian tests:
    - BayesAnova: Between/Within/Mixed n-way Bayesian ANOVAs
    - BayesT2Samples, BayesT1Sample: Bayesian Independent\Dependent samples and One-Sample t-tests
"""

from .models import *
