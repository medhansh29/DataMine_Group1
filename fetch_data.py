import time
import pandas as pd
from alerce.core import Alerce
import os

def collect_light_curves(num_objects_to_fetch: int, objects_per_page: int = 100):
    """
    Collects light curve data for a specified number of ZTF objects
    and saves the combined data to a CSV file.

    This script automatically handles pagination to fetch a large number of objects.

    Args:
        num_objects_to_fetch (int): The total number of objects to fetch light curves for.
        objects_per_page (int): The number of objects to request from the API in a single call.
    """
    print(f"--- Starting Data Collection ---")
    print(f"Goal: Collect light curves for {num_objects_to_fetch} objects.")
    
    client = Alerce()
    all_light_curves = []
    
    # We'll use a loop with pagination to get the required number of objects
    # This is more robust for large numbers than a single API call.
    current_page = 1
    
    while len(all_light_curves) < num_objects_to_fetch:
        print(f"\n--> Fetching page {current_page} of object IDs...")
        
        try:
            # Query for a page of objects.
            # We don't add any filters, so it returns the most recent objects in the database.
            objects = client.query_objects(page_size=objects_per_page, page=current_page)
            
            if objects.empty:
                print("--> Received an empty page. Stopping collection.")
                break

            # Process each object ID from the page until we reach our goal
            for ztf_id in objects['oid']:
                if len(all_light_curves) >= num_objects_to_fetch:
                    break

                print(f"    Processing {ztf_id}...")
                
                # Fetch the light curve data (detections and non-detections)
                lc_df = client.query_lightcurve(ztf_id, format="pandas")
                
                if lc_df is not None and not lc_df.empty:
                    lc_df['oid'] = ztf_id
                    all_light_curves.append(lc_df)
                else:
                    print(f"    Warning: No light curve data found for {ztf_id}.")

        except Exception as e:
            print(f"[ERROR] An error occurred while fetching page {current_page}: {e}")
            print("         Waiting 10 seconds before retrying...")
            time.sleep(10)
            continue
            
        current_page += 1
            
    # --- Combine and save the dataset to a file ---
    if not all_light_curves:
        print("\nNo light curve data was collected. No CSV file will be saved.")
        return

    dataset_df = pd.concat(all_light_curves, ignore_index=True)
    output_path = "ztf_light_curves.csv"
    dataset_df.to_csv(output_path, index=False)
    
    print("\n--- Data Collection Complete ---")
    print(f"Successfully collected light curve data for {len(all_light_curves)} objects.")
    print(f"Combined data saved to: {output_path}")

if __name__ == "__main__":
    try:
        num_objects_str = input("Enter the number of ZTF objects to fetch: ")
        num_objects = int(num_objects_str)
        
        # It's recommended to use a page size of 100 for this API
        collect_light_curves(num_objects_to_fetch=num_objects)
    except ValueError:
        print("Invalid input. Please enter a valid integer.")