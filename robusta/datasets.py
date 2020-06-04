import numpy as np
import pandas as pd
import rpy2

import robusta as rst


def data(dataset_name=None, package_name=None):
    """Returns either a dataset (if `dataset_name` is specieified)
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
    if dataset_name is None:
        return _get_available_datasets()
    else:
        return _load_dataset(dataset_name, package_name)


def _get_available_datasets():
    # Get a (x, 4) np.array of the
    info = rst.pyr.rpackages.utils.data()
    names = ['Package', 'LibPath', 'Item', 'Description']
    data = np.array(info[2]).reshape(4, len(info[2]) // 4)
    return pd.DataFrame(data=data.T, columns=names)


def _load_dataset(dataset_name: str, package_name: str = None):
    """
    Load an R-dataset and retrieve it as a pandas dataframe. Row-names
    (similar to pandas
    object index) are included on purpose as they may be an important identifier
    of the sample (e.g., car model in the mtcars dataset).
    @type dataset_name: str
    @rtype pd.core.frame.DataFrame
    """
    if package_name is None:
        available = _get_available_datasets()
        available = available.loc[available['Item'] == dataset_name,
                                  'Package']
        if available.shape[0] != 1:
            # TODO Not really a viable case but we should test for this.
            raise RuntimeError('More than one data set was found.')
        else:
            package_name = available.item()

    df = rst.pyr.rpackages.data(
        getattr(rst.pyr.rpackages, package_name)).fetch(
        dataset_name)[dataset_name]
    if type(df) == rpy2.robjects.vectors.DataFrame:
        return rst.utils.convert_df(df).join(
            pd.DataFrame(np.array(df.rownames), columns=['row_names']))
    raise NotImplementedError('Currently only supports datases imported to'
                              ' R as `data.frame` or `tibble`')
