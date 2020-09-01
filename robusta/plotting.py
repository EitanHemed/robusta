"""
TODO: Lots of stuff

Many inferential plots.

SequentialAnalyzer - A sequential plot.
PairLevelPlotter - An inferential pairwise (Maybe for dosage, sessions, etc.).
"""

from matplotlib import patches
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
import pandas as pd
import numpy as np


def plot_posterior_dist(posterior_vals, ax=None, ci_kws={},
                        **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    lower, upper = _get_credible_interval(posterior_vals)

    ax = sns.distplot(a=posterior_vals, **kwargs)

    peak = _grab_peak_from_distplot(ax)

    ax.plot(
        [np.percentile(posterior_vals, lower),
         np.percentile(posterior_vals, upper)],
        [peak, peak], **ci_kws)

    return ax

def _grab_peak_from_distplot(ax):
    # grab the peak of the KDE by extracting the last drawn line
    child = ax.get_lines()[-1]
    _, y = child.get_data()
    return y.max()


def _get_credible_interval(posterior_vals):
    return [np.percentile(posterior_vals, 2.5),
            np.percentile(posterior_vals, 97.5)]


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
        pass

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
