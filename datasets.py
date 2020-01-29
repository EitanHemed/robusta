import robusta  # So we can get the PyR singleton
from robusta import pyr, utils
from rpy2.robjects.packages import importr, data


def load(dataset_name):
    """
    @type dataset_name: str
    @rtype pd.core.frame.DataFrame
    """
    return utils.convert_df(pyr.rpackages.data(pyr.rpackages.datasets).fetch(dataset_name)[dataset_name])


def get_available_datasets():
    raise NotImplementedError
