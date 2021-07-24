from .models import *

__all__ = [
    'linear_regression',
    'bayes_linear_regression',
    'logistic_regression',
    'bayes_logistic_regression'
    ]

def linear_regression(**kwargs):
    return LinearRegressionModel(**kwargs)

def bayes_linear_regression(**kwargs):
    return BayesLinearRegressionModel(**kwargs)

def logistic_regression(**kwargs):
    return LogisticRegressionModel(**kwargs)

def bayes_logistic_regression(**kwargs):
    return BayesLogisticRegressionModel(**kwargs)

def mixed_model(**kwargs):
    return MixedModelModel(**kwargs)