import pandas as pd
from alerce.core import Alerce
import os
import time

def fetch_light_curve(ztf_object_id: str, client: Alerce, 
beginning_datapoint: int, ending_datapoint: int) -> dict | None:
    """
    Fetches the light curve (detections) for a given ZTF object ID from the ALeRCE API.

    Args:
        ztf_object_id (str): The ZTF identifier for the transient object (e.g., "ZTF23aabrisv").
        client (Alerce): An initialized ALeRCE client object.
        beginning_datapoint (int): Minimum number of datapoints required.
        ending_datapoint (int): Maximum number of datapoints allowed.

    Returns:
        A dictionary containing oid and num_detections if the object is found and within range,
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
                return {"oid": ztf_object_id, "num_detections": datapoint_count}
            else:
                print(f"{ztf_object_id} has {datapoint_count} datapoints (outside range {beginning_datapoint}-{ending_datapoint})")
                return None
            
        else:
            print(f"--> No data found for {ztf_object_id} in ALeRCE.")
            return None

    except Exception as e:
        print(f"[ERROR] An error occurred while fetching data from ALeRCE for {ztf_object_id}: {e}")
        return None

def fetch_and_save_ztf_data(n_objects: int, output_filename: str = 'ztf_objects_summary.csv', 
beginning_datapoint: int = 10, ending_datapoint: int = 300):
    """
    Automates the process of fetching light curves for a given number of ZTF objects
    and saves the combined data to a CSV file with only oid and num_detections columns.
    Uses efficient batching to query 500 objects at a time and stops when target is reached.

    Args:
        n_objects (int): The number of ZTF objects to fetch.
        output_filename (str): The name of the CSV file to save the data.
        beginning_datapoint (int): The beginning datapoint range.
        ending_datapoint (int): The ending datapoint range.
    """
    if n_objects <= 0:
        print("Please provide a positive number of objects to fetch.")
        return

    # Read existing OIDs from CSV if it exists
    existing_oids = set()
    if os.path.exists(output_filename):
        try:
            existing_df = pd.read_csv(output_filename)
            if 'oid' in existing_df.columns:
                existing_oids = set(existing_df['oid'].values)
                print(f"Found {len(existing_oids)} existing OIDs in CSV. Will skip duplicates.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV file: {e}")
            print("Proceeding without duplicate check...")

    alerce_client = Alerce()
    all_light_curves = []
    page_num = 1
    batch_size = 500  # Query 500 objects at a time for efficiency
    skipped_count = 0
    
    print(f"Starting efficient query for {n_objects} unique ZTF objects (querying {batch_size} at a time)...")
    
    while len(all_light_curves) < n_objects:
        try:
            # Query for a batch of object IDs
            print(f"Querying batch {page_num} ({batch_size} objects)...")
            objects_df = alerce_client.query_objects(page_size=batch_size, page=page_num, format="pandas")
            
            # Check if the objects DataFrame is empty
            if objects_df.empty:
                print("No more objects found. The ALeRCE API might be temporarily unavailable or we've reached the end of available objects.")
                break
            
            # Convert the object IDs to a list
            object_ids = objects_df['oid'].tolist()
            print(f"Retrieved {len(object_ids)} object IDs from batch {page_num}")

            # Process each object in the batch
            for i, oid in enumerate(object_ids):
                # Stop if we've reached our target
                if len(all_light_curves) >= n_objects:
                    print(f"Target reached! Found {len(all_light_curves)} unique objects.")
                    break
                
                # Skip if OID already exists in CSV
                if oid in existing_oids:
                    skipped_count += 1
                    if skipped_count % 10 == 0:  # Print every 10 skipped
                        print(f"Skipped {skipped_count} duplicate OIDs so far...")
                    continue
                    
                light_curve_data = fetch_light_curve(oid, alerce_client, beginning_datapoint, ending_datapoint)
                if light_curve_data is not None:
                    all_light_curves.append(light_curve_data)
                    existing_oids.add(oid)  # Add to set to avoid duplicates in same run
                    print(f"Progress: {len(all_light_curves)}/{n_objects} unique objects found")

        except Exception as e:
            print(f"[ERROR] Failed to query object IDs on page {page_num}: {e}")
            print("Please check the ALeRCE API status or try again later.")
            break
        
        page_num += 1

    # Create a DataFrame with only oid and num_detections columns
    if all_light_curves:
        # Limit to exactly n_objects if we found more than needed
        if len(all_light_curves) > n_objects:
            all_light_curves = all_light_curves[:n_objects]
            
        combined_df = pd.DataFrame(all_light_curves)
        
        # Check if file exists to determine if we should append or create new
        file_exists = os.path.exists(output_filename)
        
        if skipped_count > 0:
            print(f"\nSkipped {skipped_count} duplicate OID(s) during fetching.")
        
        if file_exists:
            # Append to existing file (duplicates already filtered during fetching)
            combined_df.to_csv(output_filename, mode='a', header=False, index=False)
            print(f"\nSuccessfully added {len(combined_df)} new unique object(s) to '{output_filename}'.")
            print(f"Added OIDs: {combined_df['oid'].tolist()}")
        else:
            # Create new file with header
            combined_df.to_csv(output_filename, index=False)
            print(f"\nSuccessfully fetched data for {len(all_light_curves)} objects.")
            print(f"New file created '{output_filename}'. Total rows: {len(combined_df)}")
    else:
        print(f"\nNo valid unique objects found matching the criteria after skipping {skipped_count} duplicates.")