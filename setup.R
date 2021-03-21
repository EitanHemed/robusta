# Title     : TODO
# Objective : TODO
# Created by: Eitan Hemed
# Created on: 3/7/2021

install.packages('renv')
library(renv)
init('./renv_robusta', bare=TRUE) # Name project

reqs = c('BayesFactor', 'ARTool', 'Matrix', 'afex',
         'backports', 'base', 'broom', 'datarium', 'datasets',
         'dplyr', 'effsize', 'emmeans', 'generics', 'lme4',
         'ppcor', 'psych', 'stats', 'tibble', 'tidyr', 'utils')

install(packages=reqs) #, project='./renv_robusta')