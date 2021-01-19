# ROBUSTA 

## R-Output-Based-Statistical-Analysis package in python

Author: [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)

Robusta is a statistics package in Python 3, aimed at providing
accsess to many frequent statistical tests and tools. The statistical
analyses are performed throgh a high-level interface with the R language,
using [RPY2](https://github.com/rpy2/rpy2).  

Robusta includes a variety of frequent statistical tests and tools used in the 
academia and industry (most come both with 
    frequentist and Bayesian implementation):
- Correlation coefficients - Pearson, Spearman, Kendell, $`\chi^2`$, 
    part and partial correlation
- T-tests - one sample, paired/unpaired samples, Welch, Wilcoxon  
- Anova - between, within, mixed, Kruskall-Wallis, Friedman
- Linear and logistic regression 
- Mixed-Effects Models (Coming soon)


Along the statistical tools, robusta offers several tools which are
useful for quickly producing publication-ready reports:
- Plain text 
- LaTex formatted   
- ?

The plotting functions of robusta include inferential and descriptive plots,
including:
- Bayesian sequential analysis plots (inspired by 
[JASP](https://jasp-stats.org/))
- Bayesian posterior distribution plots 
- Marginal effect plots
- Correlogram



