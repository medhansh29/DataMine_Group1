import pandas as pd
from alerce.core import Alerce

def datapoint_filter(ztf_object_id: str, datapoint_count: int, beginning_datapoint: int, ending_datapoint: int, light_curve_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Filters the light curve dataframe based on the datapoint range.

    Args:
        ztf_object_id (str): The ZTF identifier for the transient object (e.g., "ZTF23aabrisv").
        datapoint_count (int): The number of datapoints for the object.
        beginning_datapoint (int): The beginning datapoint range.
        ending_datapoint (int): The ending datapoint range.
        light_curve_df (pd.DataFrame): The light curve dataframe to filter.

    Returns:
        A pandas DataFrame containing the filtered light curve data if the object is found,
        otherwise None.
    """
    # Check if this object has the right number of datapoints
    if datapoint_count >= beginning_datapoint and datapoint_count <= ending_datapoint:
        print(f"{ztf_object_id} has {datapoint_count} datapoints (within range {beginning_datapoint}-{ending_datapoint})")
        light_curve_df['oid'] = ztf_object_id
        return light_curve_df
    else:
        print(f"{ztf_object_id} has {datapoint_count} datapoints (outside range {beginning_datapoint}-{ending_datapoint})")
        return None