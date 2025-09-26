import pandas as pd
from alerce.core import Alerce
import os

def fetch_light_curve(ztf_object_id: str, client: Alerce) -> pd.DataFrame | None:
    """
    Fetches the light curve (detections) for a given ZTF object ID from the ALeRCE API.

    Args:
        ztf_object_id: The ZTF identifier for the transient object (e.g., "ZTF23aabrisv").
        client: An initialized ALeRCE client object.

    Returns:
        A pandas DataFrame containing the light curve data if the object is found,
        otherwise None.
    """
    print(f"-> Fetching data for {ztf_object_id} from ALeRCE...")
    try:
        # Query for all detections for the given object ID
        light_curve_df = client.query_detections(ztf_object_id, format="pandas")

        if light_curve_df is not None and not light_curve_df.empty:
            print(f"--> Successfully found and extracted light curve with {len(light_curve_df)} data points.")
            light_curve_df['oid'] = ztf_object_id
            return light_curve_df
        else:
            print(f"--> No data found for {ztf_object_id} in ALeRCE.")
            return None
    except Exception as e:
        print(f"[ERROR] An error occurred while fetching data from ALeRCE for {ztf_object_id}: {e}")
        return None

def fetch_and_save_ztf_data(n_objects: int, output_filename: str = 'ztf_light_curves.csv'):
    """
    Automates the process of fetching light curves for a given number of ZTF objects
    and saves the combined data to a CSV file.

    Args:
        n_objects (int): The number of ZTF objects to fetch.
        output_filename (str): The name of the CSV file to save the data.
    """
    if n_objects <= 0:
        print("Please provide a positive number of objects to fetch.")
        return

    alerce_client = Alerce()
    
    print(f"Querying for a list of {n_objects} ZTF object IDs...")
    try:
        # Query for a list of recent object IDs. We don't need any special filters here.
        # This is the step that failed previously.
        objects_df = alerce_client.query_objects(page_size=n_objects, page=1, format="pandas")
        
        if objects_df.empty:
            print("No objects found with the current query. The ALeRCE API might be temporarily unavailable or there are no new objects to return.")
            return
        
        object_ids = objects_df['oid'].tolist()
        print(f"Successfully retrieved {len(object_ids)} object IDs.")

    except Exception as e:
        print(f"[ERROR] Failed to query object IDs: {e}")
        print("Please check the ALeRCE API status or try again later.")
        return

    all_light_curves = []
    
    # Iterate through the retrieved object IDs and fetch the light curve for each.
    for i, oid in enumerate(object_ids):
        light_curve_data = fetch_light_curve(oid, alerce_client)
        if light_curve_data is not None:
            all_light_curves.append(light_curve_data)

    if not all_light_curves:
        print("No light curve data was successfully fetched. Exiting.")
        return

    # Combine all light curves into a single DataFrame.
    combined_df = pd.concat(all_light_curves, ignore_index=True)
    
    # Save the combined DataFrame to a CSV file.
    combined_df.to_csv(output_filename, index=False)
    print(f"\nSuccessfully fetched data for {len(all_light_curves)} objects.")
    print(f"Combined data saved to '{output_filename}'. Total rows: {len(combined_df)}")

if __name__ == '__main__':
    try:
        num_objects_str = input("Enter the number of ZTF objects to fetch: ")
        num_objects = int(num_objects_str)
        fetch_and_save_ztf_data(num_objects)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")