import pandas as pd
from alerce.core import Alerce
import os
import time

def fetch_light_curve(ztf_object_id: str, client: Alerce, 
beginning_datapoint: int, ending_datapoint: int) -> pd.DataFrame | None:
    """
    Fetches the light curve (detections) for a given ZTF object ID from the ALeRCE API.

    Args:
        ztf_object_id (str): The ZTF identifier for the transient object (e.g., "ZTF23aabrisv").
        client (Alerce): An initialized ALeRCE client object.
        beginning_datapoint (int): Minimum number of datapoints required.
        ending_datapoint (int): Maximum number of datapoints allowed.

    Returns:
        A pandas DataFrame containing the light curve data if the object is found,
        otherwise None.
    """
    print(f"-> Fetching data for {ztf_object_id} from ALeRCE...")
    try:
        # Query for all detections for the given object ID
        light_curve_df = client.query_detections(ztf_object_id, format="pandas")

        if light_curve_df is not None and not light_curve_df.empty:
            datapoint_count = len(light_curve_df)
            print(f"--> Found {datapoint_count} datapoints for {ztf_object_id}")
            
            # Check if this object has the right number of datapoints
            if datapoint_count >= beginning_datapoint and datapoint_count <= ending_datapoint:
                print(f"{ztf_object_id} has {datapoint_count} datapoints (within range {beginning_datapoint}-{ending_datapoint})")
                light_curve_df['oid'] = ztf_object_id
                return light_curve_df
            else:
                print(f"{ztf_object_id} has {datapoint_count} datapoints (outside range {beginning_datapoint}-{ending_datapoint})")
                return None
        else:
            print(f"--> No data found for {ztf_object_id} in ALeRCE.")
            return None

    except Exception as e:
        print(f"[ERROR] An error occurred while fetching data from ALeRCE for {ztf_object_id}: {e}")
        return None

def fetch_and_save_ztf_data(n_objects: int, output_filename: str = 'ztf_light_curves.csv', 
beginning_datapoint: int = 10, ending_datapoint: int = 300, page_num: int = 1, accumulated_objects: list = None):
    """
    Automates the process of fetching light curves for a given number of ZTF objects
    and saves the combined data to a CSV file.

    Args:
        n_objects (int): The number of ZTF objects to fetch.
        output_filename (str): The name of the CSV file to save the data.
        beginning_datapoint (int): The beginning datapoint range.
        ending_datapoint (int): The ending datapoint range.
        page_num (int): The page number to query.
        accumulated_objects (list): A list of accumulated light curves from previous attempts.
    """
    if n_objects <= 0:
        print("Please provide a positive number of objects to fetch.")
        return

    alerce_client = Alerce()
    
    print(f"Querying for a list of {n_objects} ZTF object IDs...")
    try:
        # Query for a list of recent object IDs. We don't need any special filters here.
        # This is the step that failed previously.
        objects_df = alerce_client.query_objects(page_size=n_objects, page=page_num, format="pandas")
        
        # Check if the objects DataFrame is empty
        if objects_df.empty:
            print("No objects found with the current query. The ALeRCE API might be temporarily unavailable or there are no new objects to return.")
            return
        
        # Convert the object IDs to a list
        object_ids = objects_df['oid'].tolist()
        print(f"Successfully retrieved {len(object_ids)} object IDs.")

    except Exception as e:
        print(f"[ERROR] Failed to query object IDs: {e}")
        print("Please check the ALeRCE API status or try again later.")
        return

    # Initialize or use accumulated light curves from previous attempts
    if accumulated_objects is None:
        all_light_curves = []
    else:
        all_light_curves = accumulated_objects.copy()
        print(f"Continuing with {len(all_light_curves)} objects from previous attempts...")
    
    # Iterate through the retrieved object IDs and fetch the light curve for each.
    for i, oid in enumerate(object_ids):
        light_curve_data = fetch_light_curve(oid, alerce_client, beginning_datapoint, ending_datapoint)
        if light_curve_data is not None:
            all_light_curves.append(light_curve_data)

    # Check if we got the exact number of objects we wanted
    if len(all_light_curves) < n_objects:
        print(f"Found {len(all_light_curves)} valid objects so far, need {n_objects - len(all_light_curves)} more. Retrying with new data...")
        fetch_and_save_ztf_data(n_objects, output_filename, beginning_datapoint, ending_datapoint, page_num + 1, all_light_curves)
        return

    # Combine all light curves into a single DataFrame.
    combined_df = pd.concat(all_light_curves, ignore_index=True)
    
    # Save the combined DataFrame to a CSV file.
    combined_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully fetched data for {len(all_light_curves)} objects.")
    print(f"Combined data saved to '{output_filename}'. Total rows: {len(combined_df)}")