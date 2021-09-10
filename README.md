# ROBUSTA 

### Author: [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)

robusta is a statistics package in Python3 providing an interface to 
many common statistical analyses, performed using through [R](https://www.r-project.org/)
and [RPY2](https://github.com/rpy2/rpy2).  


**PLEASE NOTE** robusta is under active development and is supplied as-is with no guarantees. 
Also, robusta is currently supported on Linux. 


## Installation

Install with pip using `pip install robusta-stats`

## Usage

### Importing the library and loading data
This could take 10-15 seconds as many R libraries are imported under the hood.
If this is the first time you are importing robusta, then the first import could take a while (more on Linux, 
see Installation for more details).


```python
import robusta as rst
```

    Initializing robusta. Please wait.


    100%|██████████| 16/16 [00:15<00:00,  1.06it/s]



```python
# Define a helper function, to pretty-print output of DFs when converting the notebook to .md
# https://gist.github.com/rgerkin/af5b27a0e30531c30f2bf628aa41a553
from tabulate import tabulate
import IPython.display as d

def md_print_df(df):
    md = tabulate(df, headers='keys', tablefmt='pipe')
    md = md.replace('|    |','| %s |' % (df.index.name if df.index.name else ''))
    result = d.Markdown(md)
    return result
```

First off, we need data. Using robusta we can import R built-in and some imported datasets. You can get a full list of the datasets, similarly to calling to `data()` with no input arguments in R.


```python
md_print_df(rst.get_available_datasets().tail())
```




|     | Package   | Item                  | Description                                                                                  |
|----:|:----------|:----------------------|:---------------------------------------------------------------------------------------------|
| 285 | ARTool    | Higgins1990Table5     | Split-plot Experiment Examining Effect of Moisture and Fertilizer on Dry Matter in Peat Pots |
| 286 | ARTool    | Higgins1990Table1.art | Aligned Rank Transformed Version of Higgins1990Table1                                        |
| 287 | ARTool    | Higgins1990Table1     | Synthetic 3x3 Factorial Randomized Experiment                                                |
| 288 | ARTool    | ElkinABC              | Synthetic 2x2x2 Within-Subjects Experiment                                                   |
| 289 | ARTool    | ElkinAB               | Synthetic 2x2 Within-Subjects Experiment                                                     |



We can import a dataset using `rst.load_dataset`


```python
iris = rst.load_dataset('iris')
md_print_df(iris.head())
```




|  |   dataset_rownames |   Sepal.Length |   Sepal.Width |   Petal.Length |   Petal.Width | Species   |
|---:|-------------------:|---------------:|--------------:|---------------:|--------------:|:----------|
|  0 |                  1 |            5.1 |           3.5 |            1.4 |           0.2 | setosa    |
|  1 |                  2 |            4.9 |           3   |            1.4 |           0.2 | setosa    |
|  2 |                  3 |            4.7 |           3.2 |            1.3 |           0.2 | setosa    |
|  3 |                  4 |            4.6 |           3.1 |            1.5 |           0.2 | setosa    |
|  4 |                  5 |            5   |           3.6 |            1.4 |           0.2 | setosa    |



### Running statistical analyses

Analyses are performed through using designated model objects that also store the . The model objects are returned through calls to the function API. In this example we create a model (`m`) object by calling `t2samples`. `m` will be used to fit the statistical model, returning the `results` object.

Here is a paired-samples t-test using the Students' sleep dataset previously loaded:


```python
# Create the model
m = rst.groupwise.T2Samples(
    data=rst.load_dataset('sleep'), independent='group', 
    dependent='extra', subject='ID', paired=True, tail='less')

# Dataframe format of the results
md_print_df(m.report_table())

```




|  |        t |   df |    p-value |   Cohen-d Low |   Cohen-d |   Cohen-d High |
|---:|---------:|-----:|-----------:|--------------:|----------:|---------------:|
|  0 | -4.06213 |    9 | 0.00141645 |      -4.57304 |  -2.70809 |      -0.758771 |




```python
# Textual report of the results - copy and paste into your results section!
m.report_text()
```




    't(9) = -4.06, p = 0.001'



We can reset the models in order to update the model parameters and re-fit it. In this example, we run the same model an an independent samples t-test:


```python
m.reset(paired=False, assume_equal_variance=True)
md_print_df(m.report_table())
```




|  |        t |   df |   p-value |   Cohen-d Low |   Cohen-d |   Cohen-d High |
|---:|---------:|-----:|----------:|--------------:|----------:|---------------:|
|  0 | -1.86081 |   18 | 0.0395934 |      -1.83678 | -0.877196 |       0.106988 |



#### Bayesian t-tests
`bayes_t2samples` and `bayes_t1sample` allow you to calculate Bayes factors or sample from the posterior distribution:


```python
m = rst.groupwise.BayesT2Samples(
        data=rst.load_dataset('mtcars'), subject='dataset_rownames',
        dependent='mpg', independent='am', prior_scale=0.5,
        paired=False)

md_print_df(m.report_table())
```




|  | model       |      bf |       error |
|---:|:------------|--------:|------------:|
|  0 | Alt., r=0.5 | 71.3861 | 3.96597e-09 |




```python
# Test different null intervals and prior values:
m.reset(prior_scale=0.1, null_interval=[0, 0.5])
m.fit()
print(f'{m.report_text()}\n\n')
md_print_df(m.report_table())

```

    Alt., r=0.1 0<d<0.5 [BF1:0 = 0.46, Error = 0.001%]. Alt., r=0.1 !(0<d<0.5) [BF1:0 = 32.76, Error = 0.001%]
    
    





|  | model                  |        bf |       error |
|---:|:-----------------------|----------:|------------:|
|  0 | Alt., r=0.1 0<d<0.5    |  0.463808 | 1.6519e-06  |
|  1 | Alt., r=0.1 !(0<d<0.5) | 32.7598   | 1.11604e-06 |



#### Analysis of variance
use `Anova` to run between, within or mixed-design ANOVA, we load the anxiety dataset for the next demonstrations. 

For non-parametric ANOVAs see `KruskalWallisTest`, `FriedmanTest` and `AlignedRanksTest`



```python
# Load the dataset and modify it from a 'wide' to 'long' format dataframe
anxiety = rst.load_dataset('anxiety').set_index(['id', 'group']
                                           ).filter(regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_2': 'time'})
md_print_df(anxiety.head())

```




|  |   id | group   | time   |   score |
|---:|-----:|:--------|:-------|--------:|
|  0 |    1 | grp1    | t1     |    14.1 |
|  1 |    1 | grp1    | t2     |    14.4 |
|  2 |    1 | grp1    | t3     |    14.1 |
|  3 |    2 | grp1    | t1     |    14.5 |
|  4 |    2 | grp1    | t2     |    14.6 |




```python
m = rst.groupwise.Anova(
        data=anxiety, subject='id',
        dependent='score', between='group', within='time')
md_print_df(m.report_table())
```

    R[write to console]: Contrasts set to contr.sum for the following variables: group
    





|  | Term       |   p-value |   Partial Eta-Squared |      F |   df1 |   df2 |
|---:|:-----------|----------:|----------------------:|-------:|------:|------:|
|  0 | group      |     0.019 |                 0.172 |   4.35 |  2    | 42    |
|  1 | time       |     0.001 |                 0.904 | 394.91 |  1.79 | 75.24 |
|  2 | group:time |     0.001 |                 0.84  | 110.19 |  3.58 | 75.24 |



Similarly, we run the model usign only the between subject term (`group`). As the model was already generated we can simpyl drop the within-subject term:


```python
m.reset(within=None)
md_print_df(m.report_table())
```

    R[write to console]: Contrasts set to contr.sum for the following variables: group
    





|  | Term   |   p-value |   Partial Eta-Squared |    F |   df1 |   df2 |
|---:|:-------|----------:|----------------------:|-----:|------:|------:|
|  0 | group  |     0.019 |                 0.172 | 4.35 |     2 |    42 |



R and many other statistical packages (e.g., [statsmodels](https://www.statsmodels.org/stable/index.html) support a formula interface to fit statistical models. Here it is shown that a model can also be specified by the formula kwargs rather than specifying `dependent`, `between` etc. The formula indicates that the score column is regressed by the time variable, with observations nested within the id column. 


```python
m.reset(formula='score~time|id')
md_print_df(m.report_table())

```




|  | Term   |   p-value |   Partial Eta-Squared |     F |   df1 |   df2 |
|---:|:-------|----------:|----------------------:|------:|------:|------:|
|  0 | time   |     0.001 |                 0.601 | 66.23 |  1.15 | 50.55 |



Analysis of variance also gives us access to estimated marginal means, as a post-estimation function. 


```python
md_print_df(m.get_margins('time'))
```




|  | time   |   emmean |       SE |      df |   lower.CL |   upper.CL |
|---:|:-------|---------:|---------:|--------:|-----------:|-----------:|
|  0 | t1     |  16.9156 | 0.261236 | 55.0262 |        nan |        nan |
|  1 | t2     |  16.1356 | 0.261236 | 55.0262 |        nan |        nan |
|  2 | t3     |  15.1978 | 0.261236 | 55.0262 |        nan |        nan |



We can also run a similar, bayesian ANOVA using `BayesAnova` comparing the specified terms to the null model:


```python
m = rst.groupwise.BayesAnova(data=anxiety, within='time',
                        dependent='score', subject='id')
md_print_df(m.report_table())
```




|  | model   |      bf |       error |
|---:|:--------|--------:|------------:|
|  0 | time    | 496.129 | 7.82496e-05 |



## Work in progress and planned features

robusta includes several other features that are either under development or planned for the future.


<ins>Currently under work<ins>
- Regressions and correlations modules
  
<ins>Planned<ins>
- Sequential analysis plots (inspired by [JASP](https://jasp-stats.org/))

## Requirements
TODO

## Documentation
TODO

## Contributing

All help is welcomed, please contact [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)


