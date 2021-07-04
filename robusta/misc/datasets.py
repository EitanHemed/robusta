import numpy as np
import pandas as pd
import rpy2

from .. import pyr
from ..misc import utils

import rpy2.robjects as ro

def load_dataset(dataset_name, package_name=None):
    """Returns either a dataset (if `dataset_name` is specified)
        or information on available datasets (if `dataset_name` is `None`).
        Works similarly to R's 'utils::data'

        Parameters
        ----------
        dataset_name: [None, str]
            Name of requested data set. Default is None (returns data frame
            of all available data sets).
        package_name: [None, str]
            Whether to look for the data set from a specific R package or from
            all available packages.


        Returns
        -------
        pd.DataFrame
            Either a data frame of the requested data set or data frame of all
            available data sets.
        """
    return _load(dataset_name, package_name)


def get_available_datasets():
    # Get a (x, 4) np.array of the
    info = pyr.rpackages.utils.data()
    names = ['Package', 'LibPath', 'Item', 'Description']
    data = np.fliplr(np.array(info[2]).reshape(4, len(info[2]) // 4))
    return pd.DataFrame(data=data.T, columns=names).drop(columns=['LibPath'])


def _load(dataset_name: str, package_name: str = None):
    """
    Load an R-dataset and retrieve it as a pandas dataframe. Row-names
    (similar to pandas
    object index) are included on purpose as they may be an important identifier
    of the sample (e.g., car model in the mtcars dataset).
    @type dataset_name: str
    @rtype pd.core.frame.DataFrame
    """
    if package_name is None:
        available = get_available_datasets()
        available = available.loc[available['Item'] == dataset_name,
                                  'Package']
        if available.shape[0] != 1:
            # TODO Not really a viable case but we should test for this.
            raise RuntimeError('More than one data set was found.')
        else:
            package_name = available.item()

    # TODO - REFACTOR THIS
    with ro.conversion.localconverter(ro.default_converter + ro.pandas2ri.converter):
        return utils.convert_df(utils.convert_df(pyr.rpackages.data(
        getattr(pyr.rpackages, package_name)).fetch(
        dataset_name)[dataset_name], 'dataset_rownames'), 'dataset_rownames')


