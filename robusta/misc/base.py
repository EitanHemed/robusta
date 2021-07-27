import abc

from .. import pyr
from ..misc import utils


class BaseModel(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        pass
        # self.reset(**kwargs)

    def reset(self, **kwargs):
        """
        This function can be used by the user to re-set all parameters of the
        model.

        @param kwargs:
        @return: None
        """
        pass

    @abc.abstractmethod
    def _pre_process(self):
        self._set_model_controllers()
        self._select_input_data()
        self._validate_input_data()
        self._transform_input_data()

    @abc.abstractmethod
    def _set_model_controllers(self):
        pass

    @abc.abstractmethod
    def _select_input_data(self):
        pass

    @abc.abstractmethod
    def _transform_input_data(self):
        pass

    @abc.abstractmethod
    def _validate_input_data(self):
        pass

    @abc.abstractmethod
    def _analyze(self):
        pass

    @abc.abstractmethod
    def fit(self):
        """
        This method runs the model defined by the input.
        @return:
        """
        self._pre_process()
        # returns the results objects that is created with the (r) results object
        return self._analyze()


# A problem with the model class and fit class would be that you'd have to
# explicitly pass some information to the results objects (e.g., formula,
# group vs. repeated variables, etc.).
class BaseResults:
    columns_rename = {}
    returned_columns = []

    def __init__(self, r_results, **kwargs):
        self.r_results = r_results

        self.results_df = self._reformat_r_output_df()

    def get_df(self):
        return self.results_df.copy()

    def _reformat_r_output_df(self):
        return None

    def _tidy_results(self):
        return pyr.rpackages.generics.tidy(self.r_results)

    def _get_r_output_df(self):
        return utils.convert_df(self._tidy_results())

    def _reformat_r_output_df(self):
        df = self._get_r_output_df().copy()
        df.rename(columns=self.columns_rename, inplace=True)
        return df[self.returned_columns]
