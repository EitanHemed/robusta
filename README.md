# ROBUSTA 
## R-Output-Based-Statistical-Analysis
### Author: [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)

robusta is a statistics package in Python3 providing an interface to 
many common statistical analyses, performed using through [R](https://www.r-project.org/)
and [RPY2](https://github.com/rpy2/rpy2).  


**PLEASE NOTE** robusta is under active development and is supplied as-is with no guarantees.


## Installation

Install with pip using `pip install https://github.com/EitanHemed/robusta/archive/master.zip`

## Usage

### Importing the library and loading data
Let's import rosbusta. This could take up to 10 seconds as many R libraries are imported under the hood. If you begin with an empty R environment the first you import robusta should take 1-2 minutes, as some R dependencies will be installed.


```python
# https://gist.github.com/rgerkin/af5b27a0e30531c30f2bf628aa41a553
from tabulate import tabulate
import IPython.display as d

def md_print_df(df):
    md = tabulate(df, headers='keys', tablefmt='pipe')
    md = md.replace('|    |','| %s |' % (df.index.name if df.index.name else ''))
    result = d.Markdown(md)
    return result
```


```python
import robusta as rst
```

    Initializing robusta. Please wait.


    100%|██████████| 15/15 [00:13<00:00,  1.10it/s]


First off, we need data. Using robusta we can import R built-in and some imported datasets. You can get a full list of the datasets, similarly to calling to `data()` with no input arguments in R.


```python
rst.get_available_datasets().tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Package</th>
      <th>Item</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>285</th>
      <td>ARTool</td>
      <td>Higgins1990Table5</td>
      <td>Split-plot Experiment Examining Effect of Mois...</td>
    </tr>
    <tr>
      <th>286</th>
      <td>ARTool</td>
      <td>Higgins1990Table1.art</td>
      <td>Aligned Rank Transformed Version of Higgins199...</td>
    </tr>
    <tr>
      <th>287</th>
      <td>ARTool</td>
      <td>Higgins1990Table1</td>
      <td>Synthetic 3x3 Factorial Randomized Experiment</td>
    </tr>
    <tr>
      <th>288</th>
      <td>ARTool</td>
      <td>ElkinABC</td>
      <td>Synthetic 2x2x2 Within-Subjects Experiment</td>
    </tr>
    <tr>
      <th>289</th>
      <td>ARTool</td>
      <td>ElkinAB</td>
      <td>Synthetic 2x2 Within-Subjects Experiment</td>
    </tr>
  </tbody>
</table>
</div>



We can import a dataset using `rst.load_dataset`


```python
iris = rst.load_dataset('iris')
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset_rownames</th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



### Running statistical analyses

Analyses are performed through using designated model objects that also store the . The model objects are returned through calls to the function API. In this example we create a model (`m`) object by calling `t2samples`. `m` will be used to fit the statistical model, returning the `results` object.

Here is a paired-samples t-test using the Students' sleep dataset previously loaded:


```python
# Create the model
m = rst.groupwise.T2Samples(
    data=rst.load_dataset('sleep'), independent='group', 
    dependent='extra', subject='ID', paired=True, tail='less')
# Fit the data
m.fit()

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
m.fit()
md_print_df(m.report_table())
```




|  |        t |   df |   p-value |   Cohen-d Low |   Cohen-d |   Cohen-d High |
|---:|---------:|-----:|----------:|--------------:|----------:|---------------:|
|  0 | -1.86081 |   18 | 0.0395934 |      -1.83678 | -0.877196 |       0.106988 |



### Supported statistical analyses

#### Frequentist t-tests
As shown above, see also `rst.t1sample`. Relatedly, see non-parametric variations of t-tests such as `wilcoxon_1sample` and `wilcoxon_2samples`.

#### Bayesian t-tests
`bayes_t2samples` and `bayes_t1sample` allow you to calculate Bayes factors or sample from the posterior distribution:


```python
m = rst.api.bayes_t2samples(
        data=rst.load_dataset('mtcars'), subject='dataset_rownames',
        dependent='mpg', independent='am', prior_scale=0.5,
        paired=False)
m.fit()
print(m.report_table())

# Test different null intervals and prior values:
m.reset(prior_scale=0.1, null_interval=[0, 0.5])
m.fit()
print(m.report_table())

print(m.report_text())


```

             model         bf         error
    0  Alt., r=0.5  71.386051  3.965971e-09
                        model         bf     error
    0     Alt., r=0.1 0<d<0.5   0.463808  0.000002
    1  Alt., r=0.1 !(0<d<0.5)  32.759791  0.000001
    BF1:0 = 0.46, Error < 0.000%


#### Analysis of variance
use `anova` to run between, within or mixed-design ANOVA, we load the anxiety dataset for the next demonstrations. 

For non-parametric ANOVAs see `kruskal_wallis_test`, `friedman_test` and `aligned_ranks_test`



```python
# Load the dataset and modify it from a 'wide' to 'long' format dataframe
anxiety = rst.load_dataset('anxiety').set_index(['id', 'group']
                                           ).filter(regex='^t[1-3]$').stack().reset_index().rename(
    columns={0: 'score',
             'level_2': 'time'})
anxiety.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>group</th>
      <th>time</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>grp1</td>
      <td>t1</td>
      <td>14.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>grp1</td>
      <td>t2</td>
      <td>14.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>grp1</td>
      <td>t3</td>
      <td>14.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>grp1</td>
      <td>t1</td>
      <td>14.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>grp1</td>
      <td>t2</td>
      <td>14.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
m = rst.api.anova(
        data=anxiety, subject='id',
        dependent='score', between='group', within='time')
m.fit()
m.report_table()
```

    R[write to console]: Contrasts set to contr.sum for the following variables: group
    



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-5-af99e6cb941a> in <module>
          3         dependent='score', between='group', within='time')
          4 m.fit()
    ----> 5 m.report_table()
    

    ~/Projects/robusta/robusta/robusta/groupwise/groupwise_models.py in report_table(self)
        273 
        274     def report_table(self):
    --> 275         return self._results.get_df()
        276 
        277     def report_text(self):


    ~/Projects/robusta/robusta/robusta/misc/base.py in get_df(self)
         69 
         70     def get_df(self):
    ---> 71         return self.results_df.copy()
         72 
         73     def _reformat_r_output_df(self):


    AttributeError: 'NoneType' object has no attribute 'copy'


Similarly, we run the model usign only the between subject term (`group`). As the model was already generated we can simpyl drop the within-subject term:


```python
m.reset(within=None)
m.fit().get_df()
```

    R[write to console]: Contrasts set to contr.sum for the following variables: group
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Effect</th>
      <th>df</th>
      <th>MSE</th>
      <th>F</th>
      <th>ges</th>
      <th>p.value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>group</td>
      <td>2, 42</td>
      <td>2.37</td>
      <td>4.35 *</td>
      <td>.172</td>
      <td>.019</td>
    </tr>
  </tbody>
</table>
</div>



R and many other statistical packages (e.g., [statsmodels](https://www.statsmodels.org/stable/index.html) support a formula interface to fit statistical models. Here it is shown that a model can also be specified by the formula kwargs rather than specifying `dependent`, `between` etc. The formula indicates that the score column is regressed by the time variable, with observations nested within the id column. 


```python
m.reset(formula='score~time|id')
res = m.fit()
res.get_df()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Effect</th>
      <th>df</th>
      <th>MSE</th>
      <th>F</th>
      <th>ges</th>
      <th>p.value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>time</td>
      <td>1.15, 50.55</td>
      <td>0.88</td>
      <td>66.23 ***</td>
      <td>.141</td>
      <td>&lt;.001</td>
    </tr>
  </tbody>
</table>
</div>



Analysis of variance also gives us access to estimated marginal means, as a post-estimation function. 


```python
res.get_margins('time')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>emmean</th>
      <th>SE</th>
      <th>df</th>
      <th>lower.CL</th>
      <th>upper.CL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>t1</td>
      <td>16.915556</td>
      <td>0.261236</td>
      <td>55.026178</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>t2</td>
      <td>16.135556</td>
      <td>0.261236</td>
      <td>55.026178</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>t3</td>
      <td>15.197778</td>
      <td>0.261236</td>
      <td>55.026178</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



We can also run a similar, bayesian ANOVA using `bayes_anova` comparing the specified terms to the null model:


```python
m = rst.api.bayes_anova(data=anxiety, within='time',
                        dependent='score', subject='id')
m.fit().get_df()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>bf</th>
      <th>error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>time</td>
      <td>496.128677</td>
      <td>0.000078</td>
    </tr>
  </tbody>
</table>
</div>



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


