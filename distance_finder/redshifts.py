import pandas as pd
import time
import os # NEW: Import os for path handling and directory creation
from alerce.core import Alerce
from alerce.exceptions import ObjectNotFoundError

# --- Configuration ---
REQUIRED_REDSHIFTS = 5 
PAGE_SIZE = 1000 # Max page size for bulk queries
CROSSMATCH_RADIUS = 30.0 # Increased radius (in arcsec) for better host galaxy matching

# File structure configuration
OUTPUT_DIRECTORY = "distance_finder" # NEW: Define the nested folder name (e.g., 'data_exports', 'results', 'output')

# To switch modes:
# 1. BULK FETCH MODE: Use an empty list to search pages: SINGLE_OIDS_TO_TEST = []
# 2. SINGLE OBJECT MODE: Populate the list with OIDs: SINGLE_OIDS_TO_TEST = ["ZTF25abvnrxq"]
SINGLE_OIDS_TO_TEST = ["ZTF17aaaadfi","ZTF17aaaadse","ZTF17aaaafgq", "ZTF17aaaafhd"] 

# --- Cosmological Constants ---
# Speed of light in km/s
C_KMS = 299792.458
# Hubble Constant in km/s/Mpc (Modern accepted value)
H0 = 70.0

def calculate_distance(z):
    """
    Calculates the approximate luminosity distance (in Mpc) using a simplified 
    Hubble's Law approximation (D = c*z / H0). 
    
    NOTE: This is only accurate for z << 1 (low redshifts).
    
    CRITICAL CHANGE: Uses the absolute value of z to return a positive distance
    for blueshifted (z < 0) objects, treating the magnitude of velocity as a proxy for distance.
    """
    # Use absolute value for distance calculation, as distance must be positive
    z_abs = abs(z)
    
    # Guard against zero redshift
    if z_abs == 0.0:
        return 0.0
        
    # Calculate distance in Mpc: D = (c / H0) * |z|
    distance_mpc = (C_KMS / H0) * z_abs
    return distance_mpc

def get_redshift_via_crossmatch(client, ra, dec, oid):
    """
    Performs the catsHTM cross-match to find a redshift for the given coordinates.
    """
    try:
        # catshtm_redshift is a helper that returns the single most reliable redshift 
        # from external catalogs (like NED or SDSS) within the search radius.
        redshift_result = client.catshtm_redshift(
            ra=ra, 
            dec=dec, 
            radius=CROSSMATCH_RADIUS, 
            format='pandas', # Use pandas format
            verbose=False
        )
        
        # The result might be a single float or a pandas Series/DataFrame depending on the ALeRCE version/match.
        # We try to extract the first valid number found.
        if isinstance(redshift_result, pd.DataFrame) and not redshift_result.empty:
             # If it's a DataFrame, try to grab the first numerical value
             return redshift_result.iloc[0, 0]
        elif isinstance(redshift_result, (float, int)):
             # If it's a single float/int (as observed in the test run)
             return redshift_result
        else:
             # Handle empty results or unexpected types
             return None

    except Exception as e:
        # Log the error but continue execution for other objects
        if SINGLE_OIDS_TO_TEST:
             # If in single mode, re-raise the exception to show the full traceback
             raise e
        print(f"    [WARN] Cross-match failed for {oid}. Error: {e}")
        return None

def fetch_redshifts_in_bulk(required_count):
    """
    Iteratively queries object pages until 'required_count' objects with
    a redshift are found, or the API runs out of data.
    """
    client = Alerce()
    
    # Stores results for console output (OID -> Status/Value)
    redshift_data = {}
    # Stores results for CSV output (List of dicts)
    final_results_list = []
    
    redshift_count = 0
    page_num = 1
    
    use_single_mode = bool(SINGLE_OIDS_TO_TEST)
    
    if use_single_mode:
        print(f"Running in SINGLE OBJECT MODE. Target: {len(SINGLE_OIDS_TO_TEST)} OIDs.")
        print(f"--- MODE ACTIVE: Processing fixed list of objects. ---")
        oids_to_process = [{'oid': oid, 'meanra': None, 'meandec': None} for oid in SINGLE_OIDS_TO_TEST]

        # In single mode, we must fetch the full object data to get coordinates 
        updated_oids_to_process = []
        for item in oids_to_process:
            oid = item['oid']
            try:
                # Need to use query_object to fetch coordinates for a single OID
                obj_data = client.query_object(oid=oid, format='pandas')
                if not obj_data.empty:
                    item['meanra'] = obj_data.iloc[0]['meanra']
                    item['meandec'] = obj_data.iloc[0]['meandec']
                    updated_oids_to_process.append(item)
                else:
                    print(f"[WARN] Could not find full data for test OID: {oid}")
            except ObjectNotFoundError:
                print(f"[WARN] Test OID not found in ALeRCE: {oid}")
            except Exception as e:
                print(f"[ERROR] Failed to fetch object data for {oid}: {e}")

        oids_to_process = updated_oids_to_process
    
    print(f"Starting ALeRCE Redshift Fetcher (Target: {required_count} objects)...")
    
    while redshift_count < required_count:
        
        # 1. Get OIDs and Coordinates (BULK QUERY - Only runs if not in single mode)
        if not use_single_mode:
            print(f"Step 1/2: Querying Page {page_num} of candidates...")
            try:
                bulk_data = client.query_objects(
                    page=page_num, 
                    page_size=PAGE_SIZE,
                    format='pandas'
                )
            except Exception as e:
                 print(f"[ERROR] API query failed on page {page_num}: {e}")
                 break

            if bulk_data.empty:
                print("    [INFO] No more objects returned by the API. Stopping search.")
                break
            
            # Extract OIDs and coordinates from the bulk data for processing
            oids_to_process = bulk_data[['oid', 'meanra', 'meandec']].to_dict('records')
        
        # This check is necessary if the single mode list was empty due to OID not found
        if not oids_to_process:
             break
             
        print(f"    Processing {len(oids_to_process)} object(s)...")

        # 2. Process each object for redshift
        for item in oids_to_process:
            oid = item['oid']
            ra = item.get('meanra')
            dec = item.get('meandec')

            if ra is None or dec is None:
                redshift_data[oid] = "N/A (Missing RA/Dec)"
                continue
                
            print(f"    Step 2/2: OID: {oid}. Performing catsHTM cross-match at RA={ra:.2f}, Dec={dec:.2f}...")

            redshift = get_redshift_via_crossmatch(client, ra, dec, oid)
            
            if redshift is not None and redshift != 0.0:
                redshift = float(redshift)
                distance_mpc = calculate_distance(redshift)
                
                # Log for console summary
                redshift_data[oid] = redshift 
                
                # Log for CSV output
                final_results_list.append({
                    'oid': oid,
                    'redshift': redshift,
                    'distance_mpc': distance_mpc
                })

                redshift_count += 1
                print(f"        [FOUND] Redshift for {oid}: {redshift:.6f} (Distance: {distance_mpc:.2f} Mpc)")
            else:
                redshift_data[oid] = "N/A (No catsHTM match)"
            
            # Check if the required count is met
            if redshift_count >= required_count:
                break
        
        # Break the main loop if the count was met inside the inner loop
        if redshift_count >= required_count:
            break
            
        # If running the fixed test list, we only process it once
        if use_single_mode:
            break
            
        page_num += 1
        # Add a small delay between pages to be friendly to the API
        time.sleep(1) 

    # --- Final Summary (Console) ---
    print("\n" + "="*40)
    print(" " * 10 + "FINAL REDSHIFT SUMMARY")
    print("="*40)
    
    found_count = 0
    for oid, result in redshift_data.items():
        # Only print detailed float format if it's a number that contributed to the count
        if isinstance(result, float) or isinstance(result, int):
            print(f"Object ID: {oid:<15} | Redshift: {result:<15.6f}")
            found_count += 1
        else:
            print(f"Object ID: {oid:<15} | Redshift: {result:<15}")
            
    print("="*40)
    print(f"Total redshifts found: {found_count} / {required_count}")
    print("="*40)
    
    # --- CSV Output (Updated to save to nested directory) ---
    if final_results_list:
        df_results = pd.DataFrame(final_results_list)
        output_filename = 'redshifts_summary.csv'
        
        # Create the directory if it doesn't exist
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        
        # Construct the full path
        full_output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)
        
        df_results.to_csv(full_output_path, index=False)
        print(f"\n[SUCCESS] Successfully saved {len(final_results_list)} objects to {full_output_path}")


if __name__ == "__main__":
    fetch_redshifts_in_bulk(REQUIRED_REDSHIFTS)
