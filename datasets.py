import robusta as rst
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr, data


def load(dataset_name: str):
    """
    Load an R-dataset and retrieve it as a pandas dataframe. Row-names (similar to pandas object index) are
    included on purpose as they may be an important identifier of the sample (e.g., car model in the mtcars dataset).
    @type dataset_name: str
    @rtype pd.core.frame.DataFrame
    """
    df = rst.pyr.rpackages.data(rst.pyr.rpackages.datasets).fetch(dataset_name)[dataset_name]
    return rst.utils.convert_df(df).join(pd.DataFrame(np.array(df.rownames), columns=['row_names']))
    #  return rst.utils.convert_df()

def get_available_datasets():
    raise NotImplementedError
