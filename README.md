# ROBUSTA 
## R-Output-Based-Statistical-Analysis
### Author: [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)

robusta is a statistics package in Python3 providing an interface to 
many common statistical analyses, performed using through [R](https://www.r-project.org/)
using [RPY2](https://github.com/rpy2/rpy2).  

**PLEASE NOTE** robusta is under active development and is supplied as-is with no guarantees.


## Installation

Install with pip using `pip install https://github.com/EitanHemed/robusta/archive/master.zip`

## Usage

### Importing the library and loading data
Let's import rosbusta. This could take up to 10 seconds as many R libraries are imported under the hood. If you begin with an empty R environment the first you import robusta should take 1-2 minutes.

    ``` python
    import robusta as rst
    ```
    
First off, we need data. Using robusta we can import R built-in and some imported datasets. You can get a full list of the datasets, similarly to calling to `data()` with no input arguments in R.

    ``` python
    rst.get_available_datasets().head()
    ```
    
We can import a dataset using `rst.load_dataset`

    ``` python
    sleep = rst.load_dataset('sleep')
sleep.head()
    ```
    
### Running statistical analyses

Analyses are performed through using designated model objects that also store the . The model objects are returned through calls to the function API. In this example we create a model (`m`) object by calling `t2samples`. `m` will be used to fit the statistical model, returning the `results` object.

Here is a paired-samples t-test using the Students' sleep dataset previously loaded:

    ``` python
    # Create the model
m = rst.api.t2samples(
    data=rst.load_dataset('sleep'), independent='group', 
    dependent='extra', subject='ID', paired=True, tail='less')
# Fit the data
results = m.fit()
# Dataframe format of the results
results.get_df()
    ```
    
We can reset the models in order to update the model parameters and re-fit it. In this example, we run the same model an an independent samples t-test:

    ``` python
    m.reset(paired=False, assume_equal_variance=True)
m.fit().get_df()
    ```
    
### Supported statistical analyses

#### Frequentist t-tests
As shown above, see also `rst.t1sample`. Relatedly, see non-parametric variations of t-tests such as `wilcoxon_1sample` and `wilcoxon_2samples`.

#### Bayesian t-tests
`bayes_t2samples` and `bayes_t1sample` allow you to calculate Bayes factors or sample from the posterior distribution:

    ``` python
    m = rst.api.bayes_t2samples(
        data=rst.load_dataset('mtcars'), subject='dataset_rownames',
        dependent='mpg', independent='am', prior_scale=0.5,
        paired=False)
print(m.fit().get_df())

# Test different null intervals and prior values:
m.reset(prior_scale=0.1, 
        null_interval=[0, 0.5]); print(m.fit().get_df())
    ```
    
#### Analysis of variance
use `anova` to run between, within or mixed-design ANOVA, we load the anxiety dataset for the next demonstrations. 

For non-parametric ANOVAs see `kruskal_wallis_test`, `friedman_test` and `aligned_ranks_test`


    ``` python
    # Load the dataset and modify it from a 'wide' to 'long' format dataframe
anxiety = rst.load_dataset('anxiety').set_index(['id', 'group']
                                           ).filter(regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_2': 'time'})
anxiety.head()

    ```
    
    ``` python
    m = rst.api.anova(
        data=anxiety, subject='id',
        dependent='score', between='group', within='time')
res = m.fit()
res.get_df()
    ```
    
Similarly, we run the model usign only the between subject term (`group`). As the model was already generated we can simpyl drop the within-subject term:

    ``` python
    m.reset(within=None)
m.fit().get_df()
    ```
    
R and many other statistical packages (e.g., [statsmodels](https://www.statsmodels.org/stable/index.html) support a formula interface to fit statistical models. Here it is shown that a model can also be specified by the formula kwargs rather than specifying `dependent`, `between` etc. The formula indicates that the score column is regressed by the time variable, with observations nested within the id column. 

    ``` python
    m.reset(formula='score~time|id')
res = m.fit()
res.get_df()
    ```
    
Analysis of variance also gives us access to estimated marginal means, as a post-estimation function. 

    ``` python
    res.get_margins('time')
    ```
    
We can also run a similar, bayesian ANOVA using `bayes_anova` comparing the specified terms to the null model:

    ``` python
    m = rst.api.bayes_anova(data=anxiety, within='time',
                        dependent='score', subject='id')
m.fit().get_df()
    ```
    
## Work in progress and planned features

robusta includes several other features that are either under development or planned for the future.


<ins>Currently under work<ins>
- Regressions and correlations modules
  
<ins>Planned<ins>
- Sequential analysis plots (inspired by [JASP](https://jasp-stats.org/))

## Requirements


## Documentation

Mostly docstrings at the moment. But you can help by contributing to robusta in helping make one!

## Contributing

All help is welcomed, please contact [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)


