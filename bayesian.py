import numpy as np
from robusta import utils

# 'sequential_BF':
# pd.Series(dict([
#    (n, self.__try_catch_bf__(x[:n], y[:n], paired, null_interval, iters_num))
#    for n in range(2, len(x) + 1)
# ]), name='BF').reset_index(drop=False).rename(columns={'index': 'N'})

class TTestBF(object):

    def __init__(
            self,
            x=None,
            y=None,
            data=None,
            grouping_var=None,
            paired=False,
            test_type=None,
            iterations=1000,
            posterior=False,
            null_interval=[-np.inf, np.inf]):
        self.max_levels = 2
        self.data = data
        self.null_interval = null_interval
        self.x = x
        self.y = y
        #self._test_type = test_type
        self.paired = paired
        self.iterations = iterations
        self.get_posterior = posterior
        #self._test_data_coherence()
        self.results = self.run

    def run(self):
        try:
            return utils.convert_df(pyr.rpackages.base.data_frame(pyr.rpackages.base.data_frame.ttestBF(
                x=self.x, y=self.y, paired=self.paired, nullInterval=self.null_interval,
                iterations=self.iterations, posterior=self.get_posterior)), to='py')[['bf', 'error']]
        except pyr.rinterface.RRuntimeError as e:
            if "data are essentially constant" in e:
                return np.nan

    def test_type(self):
        if self._test_type is not None:
            pass
        if self.x is not None and self.y is None:
            return 'OneSample'
        else:
            if self.x is None:
                self._data = self._construct_data()

    def _test_grouping_var(self):
        if self.data is None:
            print('No data entered, cannot perform grouping')
        elif self.grouping_var not in self.data.columns:
            print("Grouping variable {} not in data".format(self.grouping_var))
        elif len(self.data[self.grouping_var].unique()) < 2:
            pass

    def null_interval(self):
        if self.tail == 'lesser':
            return [0, np.inf]
        elif self.tail == 'greater':
            return [-np.inf, 0]
        elif self.tail == 'two':
            return [-np.inf, np.inf]

    def bayes_factor(self):
        return self.results.round(2).iloc[0].to_dict()['bf']

    def bf_error(self):
        return self.results.round(2).iloc[0].to_dict()['error']

    def construct_data(self):
        if self._test_type == 'OneSample':
            self.y = pyr.rpackages.NULL

    def _test_data_coherence(self):
        if self.data is not None:
            pass
        if self.test_type == "PairedSample":
            self.y = pyr.rpackages.rinterface.NULL
