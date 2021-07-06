# ROBUSTA 
## R-Output-Based-Statistical-Analysis
### Author: [Eitan Hemed](mailto:Eitan.Hemed@gmail.com)

robusta is a statistics package in Python3 providing an interface to 
many common statistical analyses, performed using through [R](https://www.r-project.org/)
using [RPY2](https://github.com/rpy2/rpy2). robusta relies on many wonderful projects in R (**TODO - full list here**).

**PLEASE NOTE** robusta is under active development and is supplied as-is with no guarantees.


## Installation

Install with pip using `pip install https://github.com/EitanHemed/robusta/archive/master.zip`

## Usage

### Importing the library and loading data
Let's import rosbusta. This could take up to 10 seconds as many R libraries are imported under the hood. If you begin with an empty R environment the first you import robusta should take 1-2 minutes.


```python
import robusta as rst
```

First off, we need data. Using robusta we can import R built-in and some imported datasets. You can get a full list of the datasets, similarly to calling to `data()` with no input arguments in R.


```python
rst.get_available_datasets().head()
```




<div>
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
      <th>0</th>
      <td>datasets</td>
      <td>women</td>
      <td>Average Heights and Weights for American Women</td>
    </tr>
    <tr>
      <th>1</th>
      <td>datasets</td>
      <td>warpbreaks</td>
      <td>The Number of Breaks in Yarn during Weaving</td>
    </tr>
    <tr>
      <th>2</th>
      <td>datasets</td>
      <td>volcano</td>
      <td>Topographic Information on Auckland's Maunga W...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>datasets</td>
      <td>uspop</td>
      <td>Populations Recorded by the US Census</td>
    </tr>
    <tr>
      <th>4</th>
      <td>datasets</td>
      <td>trees</td>
      <td>Diameter, Height and Volume for Black Cherry T...</td>
    </tr>
  </tbody>
</table>
</div>



We can import a dataset using `rst.load_dataset`


```python
sleep = rst.load_dataset('sleep')
sleep.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset_rownames</th>
      <th>extra</th>
      <th>group</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.7</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>-1.6</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>-0.2</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>-1.2</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>-0.1</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



### Running statistical analyses

Analyses are performed through using designated model objects that also store the . The model objects are returned through calls to the function API. In this example we create a model (`m`) object by calling `t2samples`. `m` will be used to fit the statistical model, returning the `results` object.

Here is a paired-samples t-test using the Students' sleep dataset previously loaded:

```python
# Create the model
m = rst.api.t2samples(
    data=rst.load_dataset('sleep'), independent='group',
    dependent='extra', subject='ID', paired=True, tail='less')
# Fit the data
results = m.fit()
# Dataframe format of the results
results._get_r_output_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimate</th>
      <th>statistic</th>
      <th>p.value</th>
      <th>parameter</th>
      <th>conf.low</th>
      <th>conf.high</th>
      <th>method</th>
      <th>alternative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.58</td>
      <td>-4.062128</td>
      <td>0.001416</td>
      <td>9.0</td>
      <td>-inf</td>
      <td>-0.866995</td>
      <td>Paired t-test</td>
      <td>less</td>
    </tr>
  </tbody>
</table>
</div>



We can reset the models in order to update the model parameters and re-fit it. In this example, we run the same model an an independent samples t-test:

```python
m.reset(paired=False, assume_equal_variance=True)
m.fit()._get_r_output_df()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>estimate</th>
      <th>estimate1</th>
      <th>estimate2</th>
      <th>statistic</th>
      <th>p.value</th>
      <th>parameter</th>
      <th>conf.low</th>
      <th>conf.high</th>
      <th>method</th>
      <th>alternative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.58</td>
      <td>0.75</td>
      <td>2.33</td>
      <td>-1.860813</td>
      <td>0.039593</td>
      <td>18.0</td>
      <td>-inf</td>
      <td>-0.107622</td>
      <td>Two Sample t-test</td>
      <td>less</td>
    </tr>
  </tbody>
</table>
</div>



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
print(m.fit()._get_r_output_df())

# Test different null intervals and prior values:
m.reset(prior_scale=0.1,
        null_interval=[0, 0.5]);
print(m.fit()._get_r_output_df())
```

             model         bf         error
    0  Alt., r=0.5  71.386051  3.965971e-09
                        model         bf     error
    0     Alt., r=0.1 0<d<0.5   0.463808  0.000002
    1  Alt., r=0.1 !(0<d<0.5)  32.759791  0.000001


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
res = m.fit()
res._get_r_output_df()
```

    R[write to console]: Contrasts set to contr.sum for the following variables: group
    





<div>
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
      <td>7.12</td>
      <td>4.35 *</td>
      <td>.168</td>
      <td>.019</td>
    </tr>
    <tr>
      <th>1</th>
      <td>time</td>
      <td>1.79, 75.24</td>
      <td>0.09</td>
      <td>394.91 ***</td>
      <td>.179</td>
      <td>&lt;.001</td>
    </tr>
    <tr>
      <th>2</th>
      <td>group:time</td>
      <td>3.58, 75.24</td>
      <td>0.09</td>
      <td>110.19 ***</td>
      <td>.108</td>
      <td>&lt;.001</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, we run the model usign only the between subject term (`group`). As the model was already generated we can simpyl drop the within-subject term:

```python
m.reset(within=None)
m.fit()._get_r_output_df()
```

    R[write to console]: Contrasts set to contr.sum for the following variables: group
    





<div>
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
res._get_r_output_df()
```




<div>
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
m.fit()._get_r_output_df()
```




<div>
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


