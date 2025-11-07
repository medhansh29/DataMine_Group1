import pandas as pd
import time
import os # NEW: Import os for path handling and directory creation
import requests
from alerce.core import Alerce
from alerce.exceptions import ObjectNotFoundError

# --- Configuration ---
CROSSMATCH_RADIUS = 100.0 # Increased radius (in arcsec) for better host galaxy matching
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
        # Check what methods are available
        available_methods = [m for m in dir(client) if 'redshift' in m.lower() or 'crossmatch' in m.lower() or 'catshtm' in m.lower()]
        print(f"    [DEBUG] Available methods: {available_methods}")
        
        # Try crossmatch method first (more reliable)
        if hasattr(client, 'crossmatch'):
            print(f"    [DEBUG] Trying crossmatch method for {oid} at RA={ra:.6f}, Dec={dec:.6f}, radius={CROSSMATCH_RADIUS}")
            try:
                result = client.crossmatch(ra=ra, dec=dec, radius=CROSSMATCH_RADIUS, format='pandas')
                print(f"    [DEBUG] Crossmatch result type: {type(result)}, empty: {result.empty if isinstance(result, pd.DataFrame) else 'N/A'}")
                
                if isinstance(result, pd.DataFrame) and not result.empty:
                    print(f"    [DEBUG] Crossmatch columns: {list(result.columns)}")
                    print(f"    [DEBUG] Crossmatch shape: {result.shape}")
                    print(f"    [DEBUG] First few rows:\n{result.head()}")
                    
                    # Look for redshift column (try various names)
                    redshift_cols = ['redshift', 'z', 'z_redshift', 'zspec', 'z_phot', 'z_best']
                    for col in redshift_cols:
                        if col in result.columns:
                            redshift = result[col].iloc[0]
                            if pd.notna(redshift) and redshift != 0:
                                print(f"    [DEBUG] Found redshift in column '{col}': {redshift}")
                                return float(redshift)
                    
                    # If no redshift column, check all numeric columns
                    numeric_cols = result.select_dtypes(include=[float, int]).columns
                    for col in numeric_cols:
                        val = result[col].iloc[0]
                        if pd.notna(val) and 0 < val < 10:  # Reasonable redshift range
                            print(f"    [DEBUG] Found potential redshift in column '{col}': {val}")
                            return float(val)
            except Exception as e:
                print(f"    [DEBUG] Crossmatch failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    [DEBUG] crossmatch method not found on client")
        
        # Try direct API call to ALeRCE catsHTM endpoint
        try:
            print(f"    [DEBUG] Trying direct API call to catsHTM endpoint for {oid}")
            api_url = "https://api.alerce.online/catshtm/redshift"
            params = {
                'ra': ra,
                'dec': dec,
                'radius': CROSSMATCH_RADIUS
            }
            response = requests.get(api_url, params=params, timeout=10)
            print(f"    [DEBUG] API response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"    [DEBUG] API response data: {data}")
                if isinstance(data, dict) and 'redshift' in data:
                    redshift = data['redshift']
                    if redshift is not None and redshift != 0:
                        return float(redshift)
                elif isinstance(data, list) and len(data) > 0:
                    # Try first result
                    first_result = data[0]
                    if isinstance(first_result, dict) and 'redshift' in first_result:
                        redshift = first_result['redshift']
                        if redshift is not None and redshift != 0:
                            return float(redshift)
                    elif isinstance(first_result, (float, int)):
                        if first_result != 0:
                            return float(first_result)
        except Exception as e:
            print(f"    [DEBUG] Direct API call failed: {e}")
        
        # Try catshtm_redshift method with different catalogs
        if hasattr(client, 'catshtm_redshift'):
            # List of catalogs to try (common catalogs with redshift data)
            catalogs_to_try = [
                "GAIA/DR1",
                "GAIA/DR2",
                "GAIA/DR3",
                "SDSS/DR16",
                "SDSS/DR17",
                "NED",
                "SIMBAD",
                "2MASS",
                "WISE",
                "PanSTARRS",
                "DES",
                "PS1"
            ]
            
            print(f"    [DEBUG] Trying catshtm_redshift for {oid} at RA={ra:.6f}, Dec={dec:.6f}, radius={CROSSMATCH_RADIUS}")
            
            redshift_result = None
            for catalog_name in catalogs_to_try:
                try:
                    print(f"    [DEBUG] Trying catalog: {catalog_name}")
                    result = client.catshtm_redshift(ra=ra, dec=dec, radius=CROSSMATCH_RADIUS, catalog_name=catalog_name)
                    print(f"    [DEBUG] Result from {catalog_name}: type={type(result)}, value={result}")
                    
                    if result is not None and result != 0:
                        redshift_result = result
                        print(f"    [DEBUG] Found redshift from {catalog_name}: {redshift_result}")
                        break
                except Exception as e:
                    print(f"    [DEBUG] Catalog {catalog_name} failed: {e}")
                    continue
            
            # If no catalog worked, try without catalog_name (some versions might not require it)
            if redshift_result is None:
                try:
                    print(f"    [DEBUG] Trying catshtm_redshift without catalog_name")
                    redshift_result = client.catshtm_redshift(ra=ra, dec=dec, radius=CROSSMATCH_RADIUS, verbose=False)
                    print(f"    [DEBUG] catshtm_redshift (no catalog) result type: {type(redshift_result)}, value: {redshift_result}")
                except Exception as e:
                    print(f"    [DEBUG] catshtm_redshift (no catalog) failed: {e}")
                    redshift_result = None
        else:
            redshift_result = None
            print(f"    [WARN] catshtm_redshift method not found")
        
        print(f"    [DEBUG] Final result type: {type(redshift_result)}, value: {redshift_result}")
        
        # The result might be a single float or a pandas Series/DataFrame depending on the ALeRCE version/match.
        # We try to extract the first valid number found.
        if isinstance(redshift_result, pd.DataFrame) and not redshift_result.empty:
             print(f"    [DEBUG] Result is DataFrame with columns: {list(redshift_result.columns)}")
             # If it's a DataFrame, try to grab the first numerical value
             # Check for redshift column first
             if 'redshift' in redshift_result.columns:
                 redshift = redshift_result['redshift'].iloc[0]
                 if pd.notna(redshift) and redshift != 0:
                     return float(redshift)
             elif 'z' in redshift_result.columns:
                 redshift = redshift_result['z'].iloc[0]
                 if pd.notna(redshift) and redshift != 0:
                     return float(redshift)
             else:
                 # Try first column
                 val = redshift_result.iloc[0, 0]
                 if pd.notna(val) and val != 0:
                     return float(val)
        elif isinstance(redshift_result, pd.Series) and not redshift_result.empty:
             print(f"    [DEBUG] Result is Series with index: {list(redshift_result.index)}")
             # Check for redshift in index
             if 'redshift' in redshift_result.index:
                 redshift = redshift_result['redshift']
                 if pd.notna(redshift) and redshift != 0:
                     return float(redshift)
             elif 'z' in redshift_result.index:
                 redshift = redshift_result['z']
                 if pd.notna(redshift) and redshift != 0:
                     return float(redshift)
             else:
                 val = redshift_result.iloc[0]
                 if pd.notna(val) and val != 0:
                     return float(val)
        elif isinstance(redshift_result, (float, int)):
             # If it's a single float/int (as observed in the test run)
             if redshift_result != 0:
                 return float(redshift_result)
             else:
                 return None
        elif redshift_result is None:
             print(f"    [DEBUG] Result is None")
             return None
        else:
             # Handle empty results or unexpected types
             print(f"    [DEBUG] Unexpected result type: {type(redshift_result)}, value: {redshift_result}")
             return None

    except AttributeError as e:
        # Method doesn't exist
        print(f"    [ERROR] Method not found: {e}")
        print(f"    [DEBUG] Available methods with 'redshift' or 'crossmatch': {[m for m in dir(client) if 'redshift' in m.lower() or 'crossmatch' in m.lower()]}")
        return None
    except Exception as e:
        # Log the error but continue execution for other objects
        print(f"    [WARN] Cross-match failed for {oid}. Error: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Count objects that need processing
    needs_processing = df[df['redshift'].isna() | df['distance_mpc'].isna()]
    if len(needs_processing) == 0:
        print(f"All {len(df)} objects already have redshift and distance data. Skipping processing.")
        return
    
    print(f"Processing {len(needs_processing)} objects (skipping {len(df) - len(needs_processing)} with existing data)...")
    
    # Process each object
    for index, row in df.iterrows():
        oid = row['oid']
        
        # Skip if both redshift and distance already exist
        if pd.notna(row['redshift']) and pd.notna(row['distance_mpc']):
            continue
        
        try:
            # Fetch object data to get coordinates
            obj_data = client.query_object(oid=oid, format='pandas')
            if obj_data.empty:
                print(f"[WARN] Could not find data for OID: {oid}")
                df.at[index, 'redshift'] = "N/A (Object not found)"
                df.at[index, 'distance_mpc'] = "N/A"
                continue
            
            # Check if redshift is already in the object data
            print(f"    [DEBUG] Object data columns: {list(obj_data.columns)}")
            if 'redshift' in obj_data.columns:
                redshift_val = obj_data.iloc[0]['redshift']
                if pd.notna(redshift_val) and redshift_val != 0:
                    print(f"    [DEBUG] Found redshift in object data: {redshift_val}")
                    redshift = float(redshift_val)
                    distance_mpc = calculate_distance(redshift)
                    df.at[index, 'redshift'] = redshift
                    df.at[index, 'distance_mpc'] = distance_mpc
                    print(f"  [FOUND] Redshift from object data: {redshift:.6f}, Distance: {distance_mpc:.2f} Mpc")
                    time.sleep(0.5)
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
