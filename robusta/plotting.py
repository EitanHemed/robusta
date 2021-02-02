"""
TODO: Lots of stuff
     The main problem currently is deciding how to pass the statistics from
     tests to the plots in cases where the test has to be re-run (e.g., on the
     sequential sampling you'd have to recreate the test. One solution would be
     to remove all the `self.foo` kwargs from the bar._analyze function call and
     this way you can always update the data, formula etc.
     The downside of this method is that (a) it nullifies the need in handling
     all the input taken from bar.__init__, (b) we need to re-test arguments (or
     let the user know that the validity of the arguments is not being re-tested).


Many inferential plots.

SequentialAnalyzer - A sequential plot.
PairLevelPlotter - An inferential pairwise (Maybe for dosage, sessions, etc.).
"""
import typing
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import patches

import robusta as rst


class PosteriorPlot():

    # TODO - a cool thing to do here would be an option to select whether you
    #  want to plot the distribution of mu, sigma or both (as a bivariate density
    #  plot

    def __init__(self, test_source: typing.Union[rst.BayesT1Sample,
                                                 rst.BayesT2Samples],
                 test_kw: dict = None,
                 fig: plt.Figure = None,
                 ax: plt.Axes = None,
                 fig_kw: dict = None,
                 ci: typing.Union[float, int, np.ndarray] = 95,
                 hist_kws: dict = None,
                 ci_kws: dict = None,
                 ):

        if test_source not in [rst.BayesT1Sample, rst.BayesT2Samples]:
            raise ValueError

        test_source = test_source
        self.test_args = test_kw
        if self.test_args.get('posterior', None) is not False:
            warnings.warn("`posterior` should be set to True or None. Ignoring"
                          "input!")
        self.ci = np.array(ci)

        if hist_kws is None: hist_kws = {}
        if ci_kws is None: ci_kws = {}

        if test_source.sample_from_posterior is False:
            # we need to run it from scratch... find a more elegant solution

            self.credible_interval = self._get_credible_interval()
            fig, ax = self._run_plot()

        else:
            self.credible_interval = self._get_credible_interval()
            fig, ax = self._run_plot()

        if ax is None:
            fig, ax = plt.subplots()
        self.fig, self.ax = fig, ax

    def _get_credible_interval(self):
        if not np.logical_and(self.ci >= 0, self.ci <= 100).all():
            raise ValueError("Specify `ci` between 0 and 100")

        if self.ci.size == 1:
            lower, upper = (100 - self.ci.size) / 2, (100 + self.ci.size) / 2
        elif self.ci.size == 2:
            lower, upper = self.ci.size
        else:
            # TODO - Better error message
            raise ValueError("Incorrect shape")

        return np.percentile(self.test_source.posterior_values,
                             [lower, upper])

    def _plot(self):

        self._plot_posterior_dist()

        self._plot_credible_interval()

    def _plot_posterior_dist(self):

        sns.histplot(a=self.posterior_values, ax=self.ax, **self.hist_kws)

        peak = self._grab_peak_from_distplot(self.ax)

        self.ax.plot(
            self.credible_interval,
            [peak, peak], **self.ci_kws)

    def _grab_peak_from_distplot(self, ax):
        # grab the peak of the KDE by extracting the last drawn line
        child = ax.get_lines()[-1]
        _, y = child.get_data()
        return y.max()


class SequentialAnalysis():

    def __init__(self,
                 bayes_ttest=None,
                 null_interval=None,
                 mark_inconclusiveness=False,
                 bayes_ttest_results=None,
                 inconclusiveness_low=None,
                 inconclusiveness_high=None,
                 tick_interval=None,
                 invert_axis=False,
                 ax=None):

        self.bayes_ttest = self.bayes_ttest

        self.tick_interval = tick_interval
        self.null_interval = null_interval
        self.bayes_ttest_results = bayes_ttest_results
        self.mark_inconclusiveness = mark_inconclusiveness
        self.inconclusiveness_low = inconclusiveness_low
        self.inconclusiveness_high = inconclusiveness_high
        self.ax = ax

        if self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.sq_bf_vals = self._run_sequential_analysis()

        self.plot()

    def sq_bf(self):
        n = range(2, len(self._input_data))
        p = Pool()
        return pd.DataFrame(
            dict(zip(n, p.map(self._run_sq_bf(n, self.data), n))))

    def _run_sq_bf(self, n, data):
        # TODO possibly there is a faster way. Find out what's the canonical.
        inst = self.bayes_ttest
        trimmed_input_data = inst.get_test_data().iloc[0: n, :]

    def plot(self):
        if self.mark_inconclusiveness:
            self._plot_inconclusiveness()

        #  sns.lineplot(data=self.results[dependent]['t_tests'][k]['sequential_BF'].dropna(),
        #             x='N', y='BF', label=False, ax=ax, alpha=0.5, linewidth=4)

        #  sns.scatterplot(data=self.results[dependent]['t_tests'][k]['sequential_BF'].dropna(),
        #                x='N', y='BF', label=False, ax=ax, color='black', marker='>', alpha=1,
        #                edgecolors=['black'], linewidths=8)

        self.ax.set_x_ticks((range(self.tick_interval, len(self.data))))

    def _plot_inconclusiveness(self):
        rect = patches.Rectangle(
            (0, 1 / 3), len(self.data) + 1, (3 - 1 / 3), linewidth=1,
            edgecolor='none', facecolor='lightgrey',
            alpha=0.35)
        self.ax.add_patch(rect)
