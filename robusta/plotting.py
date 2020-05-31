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
import bayesian




class SequentialAnalysis:

    def __init__(self,
                 null_interval=None,
                 mark_inconclusiveness=False,
                 bayes_ttest_results=None,
                 inconclusiveness_low=None,
                 inconclusiveness_high=None,
                 tick_interval=None,
                 invert_axis=False,
                 ax=None):

        self.tick_interval = tick_interval
        self.null_interval = null_interval
        self.bayes_ttest_results = bayes_ttest_results
        self.mark_inconclusiveness = mark_inconclusiveness
        self.inconclusiveness_low = inconclusiveness_low
        self.inconclusiveness_high = inconclusiveness_high
        self.ax = ax

        if self.ax is None:
            self.fig, self.ax = plt.subplots()

        self.sq_bf = self._run_sequential_analysis()

        self.plot()

    def sq_bf(self):
        n = range(2, self.data['N'].max())

        p = Pool()
        return pd.DataFrame(dict(zip(n, p.map(self._run_sq_bf(n, self.data), n))))

    def _run_sq_bf(self, n, data):
        return bayesian.TTestBF(data.iloc[:n]).results['BF']
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
        rect = patches.Rectangle((0, 1 / 3), len(self.data) + 1, (3 - 1 / 3), linewidth=1, edgecolor='none', facecolor='lightgrey',
                                 alpha=0.35)
        self.ax.add_patch(rect)
