import abc


class BaseModel(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        self.reset(**kwargs)

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
        self._set_controllers()
        self._select_input_data()
        self._validate_input_data()
        self._transform_input_data()

    @abc.abstractmethod
    def _set_controllers(self):
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
class BaseResults(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        pass

    def _tidy_results(self):
        pass

    def get_results_df(self):
        pass

    # TODO - do we need this?
    def accept(self, visitor):
        visitor.visit(self)

    def re_analyze(self):
        pass
