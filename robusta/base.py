import abc
import typing
from dataclasses import dataclass
import abc
import pandas as pd
import custom_inherit

import robusta as rst


class AbstractClass(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self._pre_process()
        self._process()

    def _pre_process(self):
        self._set_controllers()
        self._select_input_data()
        self._test_input_data()
        self._transform_input_data()

    def _process(self):
        self._analyze()
        self._tidy_results()

    def _set_controllers(self):
        pass

    @abc.abstractmethod
    def _select_input_data(self):
        pass

    @abc.abstractmethod
    def _transform_input_data(self):
        pass

    @abc.abstractmethod
    def _analyze(self):
        pass

    def _tidy_results(self):
        self._results = pd.DataFrame()

    def get_results(self):
        return self._results.apply(pd.to_numeric, errors='ignore').copy()

    def get_test_data(self):
        return self._input_data

    # TODO - do we need this?
    def accept(self, visitor):
        visitor.visit(self)