import pandas as pd
import time
import os # NEW: Import os for path handling and directory creation
from alerce.core import Alerce
from alerce.exceptions import ObjectNotFoundError

# --- Configuration ---
CROSSMATCH_RADIUS = 30.0 # Increased radius (in arcsec) for better host galaxy matching
CSV_FILE_PATH = "ztf_objects_summary.csv" # Path to the CSV file containing OIDs 

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
        print(f"    [WARN] Cross-match failed for {oid}. Error: {e}")
        return None

def fetch_redshifts_from_csv():
    """
    Reads OIDs from the CSV file and fetches redshift data for each one.
    Updates the CSV file with redshift and distance columns.
    """
    client = Alerce()
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        print(f"[ERROR] CSV file '{CSV_FILE_PATH}' not found!")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully read CSV file with {len(df)} objects.")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return
    
    # Check if required columns exist
    if 'oid' not in df.columns:
        print("[ERROR] CSV file must contain an 'oid' column!")
        return
    
    # Initialize redshift and distance columns if they don't exist
    if 'redshift' not in df.columns:
        df['redshift'] = None
    if 'distance_mpc' not in df.columns:
        df['distance_mpc'] = None
    
    print(f"Processing {len(df)} objects for redshift data...")
    
    # Process each object
    for index, row in df.iterrows():
        oid = row['oid']
        
        # Skip if redshift already exists
        if pd.notna(row['redshift']):
            print(f"Skipping {oid} - redshift already exists.")
            continue
        
        try:
            # Fetch object data to get coordinates
            obj_data = client.query_object(oid=oid, format='pandas')
            if obj_data.empty:
                print(f"[WARN] Could not find data for OID: {oid}")
                df.at[index, 'redshift'] = "N/A (Object not found)"
                df.at[index, 'distance_mpc'] = "N/A"
                continue
            
            ra = obj_data.iloc[0]['meanra']
            dec = obj_data.iloc[0]['meandec']
            
            if pd.isna(ra) or pd.isna(dec):
                print(f"[WARN] Missing coordinates for {oid}")
                df.at[index, 'redshift'] = "N/A (Missing RA/Dec)"
                df.at[index, 'distance_mpc'] = "N/A"
                continue
            
            print(f"Processing {oid} at RA={ra:.2f}, Dec={dec:.2f}...")
            
            # Get redshift via crossmatch
            redshift = get_redshift_via_crossmatch(client, ra, dec, oid)
            
            if redshift is not None and redshift != 0.0:
                redshift = float(redshift)
                distance_mpc = calculate_distance(redshift)
                
                df.at[index, 'redshift'] = redshift
                df.at[index, 'distance_mpc'] = distance_mpc
                
                print(f"  [FOUND] Redshift: {redshift:.6f}, Distance: {distance_mpc:.2f} Mpc")
            else:
                df.at[index, 'redshift'] = "N/A (No catsHTM match)"
                df.at[index, 'distance_mpc'] = "N/A"
                print(f"  [NOT FOUND] No redshift match for {oid}")
            
            # Small delay to be friendly to the API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {oid}: {e}")
            df.at[index, 'redshift'] = "N/A (Error)"
            df.at[index, 'distance_mpc'] = "N/A"
    
    # Save the updated CSV file
    try:
        df.to_csv(CSV_FILE_PATH, index=False)
        print(f"\n[SUCCESS] Updated CSV file '{CSV_FILE_PATH}' with redshift data.")
        
        # Print summary
        found_redshifts = df[df['redshift'].apply(lambda x: isinstance(x, (int, float)) and x != "N/A (No catsHTM match)")].shape[0]
        print(f"Summary: Found redshifts for {found_redshifts} out of {len(df)} objects.")
        
    except Exception as e:
        print(f"[ERROR] Failed to save updated CSV file: {e}")
