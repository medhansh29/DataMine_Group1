import pandas as pd
import numpy as np
import os
import math
from alerce.core import Alerce
from astropy.coordinates import SkyCoord
from astropy import units as u

# --- DELIGHT IMPORT (Requires local installation) ---
DELIGHT_AVAILABLE = False
Delight = None
try:
    from delight.delight import Delight  # common layout
    DELIGHT_AVAILABLE = True
except Exception:
    try:
        from delight import Delight  # flat module export
        DELIGHT_AVAILABLE = True
    except Exception:
        try:
            from astro_delight import Delight  # alternative package name
            DELIGHT_AVAILABLE = True
        except Exception:
            try:
                from astrodelight import Delight  # another alternative
                DELIGHT_AVAILABLE = True
            except Exception:
                print("WARNING: DELIGHT module not found (tried: delight.delight, delight, astro_delight, astrodelight). Cannot run DELIGHT.")
# -------------------------------------------------------------------------

# --- 1. Configuration and Setup ---
alerce = Alerce()
TRANSIENT_OIDS = [
    "ZTF23abaujuy"
    #"ZTF23aaqdjhi"
]
DATA_DIR = './delight_data' 

# --- 2. Data Retrieval: Coordinates (UNCHANGED) ---

def get_transient_coordinates(oids: list) -> pd.DataFrame:
    """Retrieves the transient's RA/DEC coordinates (meanra, meandec) from ALeRCE."""
    print(f"\n-> STEP 1: Retrieving coordinates for {len(oids)} objects...")
    
    data = []
    for oid in oids:
        ra, dec = np.nan, np.nan
        try:
            obj_info_df = alerce.query_objects(oid=oid, format='pandas')
            if not obj_info_df.empty and 'meanra' in obj_info_df.columns and 'meandec' in obj_info_df.columns:
                ra = obj_info_df['meanra'].iloc[0]
                dec = obj_info_df['meandec'].iloc[0]
            else:
                print(f"   - Skipping {oid}: RA/DEC not available from ALeRCE")
        except Exception:
            print(f"   - Skipping {oid}: failed to query ALeRCE")
        data.append({'oid': oid, 'ra': ra, 'dec': dec})
    
    df = pd.DataFrame(data).set_index('oid')
    data_ready = df[(df['ra'].notna()) & (df['dec'].notna())].copy()
    return data_ready

# --- 3. Host Offset and Size Calculation (FIXED) ---

def run_delight_and_get_angular_metrics(data_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """
    Executes DELIGHT, FIXES the FITS file path error, and calculates angular metrics.
    """
    if not DELIGHT_AVAILABLE:
        return data_df

    print("\n-> STEP 2: Running DELIGHT to get Angular Offset (theta)...")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dclient = Delight(data_dir, data_df.index.values, data_df['ra'].values, data_df['dec'].values)

    try:
        # 1. Download FITS files (DELIGHT handles filenames internally)
        print("   - Downloading Pan-STARRS images...")
        dclient.download()
        
        # 1.5. Verify FITS files were downloaded (with retry)
        fits_dir = os.path.join(data_dir, 'fits')
        import time
        max_wait = 30  # Wait up to 30 seconds for files to appear
        wait_interval = 2
        waited = 0
        
        while waited < max_wait:
            if os.path.exists(fits_dir):
                downloaded_files = os.listdir(fits_dir)
                if len(downloaded_files) > 0:
                    print(f"   - Found {len(downloaded_files)} FITS files in {fits_dir}")
                    break
            if waited == 0:
                print(f"   - Waiting for FITS files to download...")
            time.sleep(wait_interval)
            waited += wait_interval
        
        if not os.path.exists(fits_dir):
            print(f"   ⚠️  Warning: FITS directory not found: {fits_dir}")
            print("   DELIGHT download may have failed. Attempting to continue...")
        else:
            downloaded_files = os.listdir(fits_dir)
            if len(downloaded_files) == 0:
                print("   ⚠️  Warning: No FITS files found after download!")
                print("   DELIGHT may still be downloading. Files may appear later.")
            else:
                print(f"   - Verified {len(downloaded_files)} FITS files are available")
        
        # 1.6. Use actual downloaded filenames from DELIGHT or construct them
        try:
            if os.path.exists(fits_dir):
                actual_files = set(os.listdir(fits_dir))
                
                # Try to match actual files with expected filenames
                if hasattr(dclient, 'df') and isinstance(dclient.df, pd.DataFrame):
                    # First, check if DELIGHT already has filenames
                    if 'filename' in dclient.df.columns:
                        print("   - Using DELIGHT-provided filenames")
                    else:
                        # Construct expected filenames and match with actual files
                        filenames = []
                        for ra_val, dec_val in zip(data_df['ra'].values, data_df['dec'].values):
                            # Try different filename formats
                            possible_names = [
                                f"stack_r_ra{ra_val:.6f}_dec{dec_val:.6f}_arcsec120.fits",
                                f"stack_r_ra{ra_val:.5f}_dec{dec_val:.5f}_arcsec120.fits",
                                f"stack_r_ra{ra_val:.4f}_dec{dec_val:.4f}_arcsec120.fits",
                            ]
                            
                            # Find matching file
                            matched = False
                            for possible_name in possible_names:
                                if possible_name in actual_files:
                                    filenames.append(possible_name)
                                    matched = True
                                    break
                            
                            if not matched:
                                # Use the first format as default
                                filenames.append(possible_names[0])
                                print(f"   ⚠️  Warning: Expected file not found for RA={ra_val:.6f}, DEC={dec_val:.6f}")
                                print(f"      Looking for: {possible_names[0]}")
                                print(f"      Available files: {list(actual_files)[:3]}...")
                        
                        dclient.df['filename'] = filenames
                        
                        # Verify files exist before proceeding
                        missing_files = []
                        for idx, fname in enumerate(filenames):
                            full_path = os.path.join(fits_dir, fname)
                            if not os.path.exists(full_path):
                                missing_files.append((data_df.index[idx], fname))
                        
                        if missing_files:
                            print(f"   ⚠️  Warning: {len(missing_files)} FITS file(s) missing:")
                            for oid, fname in missing_files[:5]:  # Show first 5
                                print(f"      - {oid}: {fname}")
                            if len(missing_files) > 5:
                                print(f"      ... and {len(missing_files) - 5} more")
        except Exception as e:
            print(f"   ⚠️  Warning: Could not verify FITS filenames: {e}")
            import traceback
            traceback.print_exc()
            pass

        # 2. Run core prediction pipeline
        print("   - Reading WCS solutions and processing images...")
        dclient.get_pix_coords()
        dclient.compute_multiresolution(nlevels=5, domask=False, doobject=True, doplot=False)
        dclient.load_model()
        dclient.preprocess()
        dclient.predict() 

        # 2.5. Compute host sizes (semi-major axis) for normalized offset if available
        try:
            if hasattr(dclient, 'df') and isinstance(dclient.df, pd.DataFrame):
                print("   - Estimating host sizes (semi-major axis)...")
                for oid in list(dclient.df.index):
                    try:
                        result = dclient.get_hostsize(oid, doplot=False)
                        # Some versions return a dict with size info; persist it into the dataframe
                        if isinstance(result, dict):
                            size_keys = ['a', 'a_arcsec', 'host_a', 'semi_major_axis_arcsec', 'host_size_arcsec', 'r_eff', 'reff_arcsec']
                            for k in size_keys:
                                if k in result and pd.notna(result[k]):
                                    dclient.df.loc[oid, k] = result[k]
                            print(f"   - {oid}: hostsize keys saved -> {[k for k in size_keys if k in result]}")
                        # Some versions might return a pandas Series
                        elif hasattr(result, 'to_dict'):
                            rd = result.to_dict()
                            for k, v in rd.items():
                                if pd.notna(v):
                                    dclient.df.loc[oid, k] = v
                            print(f"   - {oid}: hostsize series saved -> {list(rd.keys())}")
                    except Exception:
                        # Best-effort; if it fails for one object, continue
                        continue
        except Exception:
            pass
        
        # Note: We are only computing host angular offset here (no host size)

    except Exception as e:
        error_msg = str(e)
        print(f"   ❌ DELIGHT execution failed: {error_msg}")
        
        # Check if it's a file not found error
        if "No such file or directory" in error_msg or "[Errno 2]" in error_msg:
            fits_dir = os.path.join(data_dir, 'fits')
            if os.path.exists(fits_dir):
                actual_files = os.listdir(fits_dir)
                print(f"   - FITS directory exists: {fits_dir}")
                print(f"   - Found {len(actual_files)} files in FITS directory")
                if actual_files:
                    print(f"   - Sample files: {actual_files[:3]}")
                    # Try to find the missing file pattern
                    if "stack_r_ra" in error_msg:
                        print(f"   - DELIGHT is looking for a file matching pattern: stack_r_ra*_dec*_arcsec120.fits")
                        matching_files = [f for f in actual_files if "stack_r_ra" in f and "_dec" in f and "_arcsec120.fits" in f]
                        if matching_files:
                            print(f"   - Found {len(matching_files)} matching files")
                        else:
                            print(f"   - No matching files found. DELIGHT may have downloaded files with different naming.")
            else:
                print(f"   - FITS directory does not exist: {fits_dir}")
                print(f"   - DELIGHT download may have failed. Try running again.")
        
        import traceback
        print(f"   - Full error traceback:")
        traceback.print_exc()
        return data_df

    # 4. Calculate Angular Offset (theta) and Normalized Offset (theta/a)
    delight_results = dclient.df.copy()

    # Normalize index if DELIGHT provides an 'oid' column
    if 'oid' in delight_results.columns and delight_results.index.name != 'oid':
        try:
            delight_results = delight_results.set_index('oid')
        except Exception:
            pass

    # Try to locate predicted RA/DEC columns flexibly
    # Prioritize DELIGHT predictions over SEXTRACTOR
    def _find_col(cands: list[str]) -> str | None:
        for c in cands:
            if c in delight_results.columns:
                return c
        return None

    # Prioritize DELIGHT predictions (ra_delight, dec_delight) over SEXTRACTOR (ra_sex, dec_sex)
    ra_candidates = ['ra_delight', 'ra_pred', 'ra_host', 'host_ra', 'ra_prediction', 'ra_p']
    dec_candidates = ['dec_delight', 'dec_pred', 'dec_host', 'host_dec', 'dec_prediction', 'dec_p']
    ra_col = _find_col(ra_candidates)
    dec_col = _find_col(dec_candidates)

    # Only use SEXTRACTOR as last resort (avoid if DELIGHT is available)
    if ra_col is None or dec_col is None:
        # Check if we have DELIGHT predictions but missed them
        if 'ra_delight' in delight_results.columns and 'dec_delight' in delight_results.columns:
            ra_col = 'ra_delight'
            dec_col = 'dec_delight'
        # Fallback: use SEXTRACTOR only if DELIGHT not available
        elif 'ra_sex' in delight_results.columns and 'dec_sex' in delight_results.columns:
            ra_col = 'ra_sex'
            dec_col = 'dec_sex'
        # Last resort: heuristic search (but avoid _sex and _delight to prevent duplicates)
        else:
            if ra_col is None:
                for c in delight_results.columns:
                    lc = c.lower()
                    if 'ra' in lc and 'err' not in lc and c != 'ra' and '_sex' not in c and '_delight' not in c:
                        ra_col = c
                        break
            if dec_col is None:
                for c in delight_results.columns:
                    lc = c.lower()
                    if 'dec' in lc and 'err' not in lc and c != 'dec' and '_sex' not in c and '_delight' not in c:
                        dec_col = c
                        break

    if ra_col is not None and dec_col is not None:
        # Collect columns to join: RA/DEC predictions plus any host size columns
        cols_to_join = [ra_col, dec_col]
        size_candidates = ['hostsize', 'a_arcsec', 'semi_major_axis_arcsec', 'host_size_arcsec', 'host_a_arcsec', 're_arcsec', 'r_eff_arcsec', 'r50_arcsec']
        host_size_col = None
        for c in size_candidates:
            if c in delight_results.columns:
                host_size_col = c
                cols_to_join.append(c)
                break
        
        # Join with renamed columns
        rename_map = {ra_col: 'ra_pred', dec_col: 'dec_pred'}
        if host_size_col:
            rename_map[host_size_col] = 'host_size_arcsec'
        delight_results_subset = delight_results[cols_to_join].rename(columns=rename_map)
        data_df = data_df.join(delight_results_subset, how='inner')

        # Calculate Angular Separation (theta)
        transient_coords = SkyCoord(ra=data_df['ra'].values * u.deg, dec=data_df['dec'].values * u.deg)
        predicted_host_coords = SkyCoord(ra=data_df['ra_pred'].values * u.deg, dec=data_df['dec_pred'].values * u.deg)
        separation = transient_coords.separation(predicted_host_coords).to(u.arcsec).value
        data_df['angular_offset_arcsec'] = separation

        # Compute normalized offset if host size is available
        if host_size_col is not None and 'host_size_arcsec' in data_df.columns:
            # Compute normalized offset where host size is valid
            with np.errstate(divide='ignore', invalid='ignore'):
                data_df['angular_normalized_offset'] = data_df['angular_offset_arcsec'] / data_df['host_size_arcsec']
                # Clean invalid values
                data_df.loc[~np.isfinite(data_df['angular_normalized_offset']), 'angular_normalized_offset'] = np.nan
            valid_norm = int(data_df['angular_normalized_offset'].notna().sum())
            print(f"   - Normalized offset computed for {valid_norm} objects using '{host_size_col}' column.")
        else:
            # Try pixel-based size conversion
            pix_cols = ['pixscale', 'pix_scale', 'pixel_scale']
            pix_col = next((p for p in pix_cols if p in delight_results.columns), None)
            if 'a' in delight_results.columns and pix_col is not None:
                temp = delight_results[['a', pix_col]].copy()
                temp['host_size_arcsec'] = temp['a'] * temp[pix_col]
                data_df = data_df.join(temp[['host_size_arcsec']], how='left')
                with np.errstate(divide='ignore', invalid='ignore'):
                    data_df['angular_normalized_offset'] = data_df['angular_offset_arcsec'] / data_df['host_size_arcsec']
                    data_df.loc[~np.isfinite(data_df['angular_normalized_offset']), 'angular_normalized_offset'] = np.nan
                valid_norm = int(data_df['angular_normalized_offset'].notna().sum())
                print(f"   - Normalized offset computed from pixels for {valid_norm} objects.")
            else:
                print("   - Host size column not found in DELIGHT results; normalized offset unavailable.")

        # Debug: show what columns are available from DELIGHT for transparency
        try:
            print(f"   - DELIGHT result columns: {list(delight_results.columns)}")
        except Exception:
            pass
        
        print("   - Angular offset successfully calculated.")
    
    else:
        print(f"\n⚠️ DELIGHT output columns missing. Available columns: {list(delight_results.columns)}")
        return data_df 

    # Prepare return columns: always include offsets; include host size and normalized offset if available
    base_cols = ['angular_offset_arcsec']
    if 'host_size_arcsec' in data_df.columns:
        base_cols.append('host_size_arcsec')
    if 'angular_normalized_offset' in data_df.columns:
        base_cols.append('angular_normalized_offset')
    # Keep predicted columns for potential debugging
    pred_cols = [c for c in ['ra_pred', 'dec_pred'] if c in data_df.columns]
    ret_cols = base_cols + pred_cols
    return data_df[data_df['angular_offset_arcsec'].notna()][ret_cols]

# --- 4. CSV Integration Function ---

def process_offsets_from_csv(csv_file: str, data_dir: str = None) -> pd.DataFrame:
    """
    Read OIDs from CSV, compute angular offsets and normalized offsets, and update CSV.
    
    Parameters:
    csv_file: Path to CSV file with 'oid' column
    data_dir: Directory for DELIGHT data storage (defaults to './delight_data' relative to CSV)
    
    Returns:
    DataFrame: Updated dataframe with offset metrics
    """
    print(f"\n--- Starting offset computation for CSV: {csv_file} ---")
    
    if not DELIGHT_AVAILABLE:
        print("ERROR: DELIGHT is not available. Cannot compute offsets.")
        print("\nTo fix this:")
        print("1. Activate the Python 3.10 virtual environment:")
        print("   cd /Users/medhansh29/DataMine/DataMine_Group1")
        print("   source .venv310/bin/activate")
        print("2. Then run your script again")
        return pd.DataFrame()
    
    # Set default data_dir relative to CSV file location if not provided
    if data_dir is None:
        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        data_dir = os.path.join(csv_dir, 'delight_data')
    
    print(f"Using DELIGHT data directory: {data_dir}")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully read CSV file: {csv_file} ({len(df)} rows)")
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    if 'oid' not in df.columns:
        print("ERROR: CSV file must contain an 'oid' column.")
        return pd.DataFrame()
    
    # Get unique OIDs
    unique_oids = df['oid'].dropna().unique().tolist()
    if not unique_oids:
        print("ERROR: No valid OIDs found in CSV.")
        return pd.DataFrame()
    
    # Filter to only process objects missing offset data
    # Check if ANY row for an OID is missing offset data (not just if all rows have it)
    oids_to_process = []
    for oid in unique_oids:
        oid_rows = df[df['oid'] == oid]
        # Check if ALL rows for this OID have angular_offset_arcsec
        if 'angular_offset_arcsec' in df.columns:
            all_have_offset = oid_rows['angular_offset_arcsec'].notna().all()
        else:
            all_have_offset = False
        # Process if ANY row is missing data
        if not all_have_offset:
            oids_to_process.append(oid)
    
    if not oids_to_process:
        print(f"All {len(unique_oids)} unique objects already have angular offset data in all rows. Skipping processing.")
        return df
    
    print(f"\nProcessing {len(oids_to_process)} objects (skipping {len(unique_oids) - len(oids_to_process)} with existing offset data)...")
    
    # Get coordinates
    initial_data = get_transient_coordinates(oids_to_process)
    
    if initial_data.empty:
        print("\n❌ No valid coordinates retrieved. Cannot compute offsets.")
        return df
    
    # Compute offsets
    data_with_metrics = run_delight_and_get_angular_metrics(initial_data.copy(), data_dir)
    
    if 'angular_offset_arcsec' not in data_with_metrics.columns:
        print("\n🚨 Offset calculation failed.")
        return df
    
    # Merge results back into original dataframe
    # Reset index to get 'oid' as a column for merging
    metrics_to_merge = data_with_metrics.reset_index()
    
    # Select only the columns we want to merge (avoid duplicates)
    cols_to_merge = ['oid', 'angular_offset_arcsec']
    if 'host_size_arcsec' in metrics_to_merge.columns:
        cols_to_merge.append('host_size_arcsec')
    if 'angular_normalized_offset' in metrics_to_merge.columns:
        cols_to_merge.append('angular_normalized_offset')
    
    # Remove duplicate _x and _y columns from entire dataframe (cleanup)
    cols_to_drop = [c for c in df.columns if (c.startswith('angular_offset') or c.startswith('host_size') or c.startswith('angular_normalized')) and (c.endswith('_x') or c.endswith('_y'))]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"   - Removed duplicate columns: {cols_to_drop}")
    
    # Merge on 'oid' - only update rows for processed objects
    df_updated = df.copy()
    
    # Ensure columns exist in df_updated
    if 'angular_offset_arcsec' not in df_updated.columns:
        df_updated['angular_offset_arcsec'] = None
    if 'host_size_arcsec' not in df_updated.columns:
        df_updated['host_size_arcsec'] = None
    if 'angular_normalized_offset' not in df_updated.columns:
        df_updated['angular_normalized_offset'] = None
    
    for _, row in metrics_to_merge.iterrows():
        oid = row['oid']
        mask = df_updated['oid'] == oid
        if not mask.any():
            print(f"   - Warning: OID {oid} not found in original dataframe")
            continue
        
        if 'angular_offset_arcsec' in row and pd.notna(row['angular_offset_arcsec']):
            df_updated.loc[mask, 'angular_offset_arcsec'] = row['angular_offset_arcsec']
        if 'host_size_arcsec' in row and pd.notna(row['host_size_arcsec']):
            df_updated.loc[mask, 'host_size_arcsec'] = row['host_size_arcsec']
        if 'angular_normalized_offset' in row and pd.notna(row['angular_normalized_offset']):
            df_updated.loc[mask, 'angular_normalized_offset'] = row['angular_normalized_offset']
    
    # Save updated CSV
    try:
        df_updated.to_csv(csv_file, index=False)
        print(f"\n✅ CSV updated with offset metrics: {csv_file}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save updated CSV: {e}")
    
    return df_updated

# --- 5. Main Execution ---

if __name__ == '__main__':
    
    # 1. Get Coordinates from ALeRCE (RA, DEC)
    initial_data = get_transient_coordinates(TRANSIENT_OIDS)
    
    if initial_data.empty:
        print("\n❌ Calculation halted: Failed to retrieve necessary RA/DEC.")
    else:
        # 2. Get Angular Offset (theta) and Size (a) via DELIGHT
        data_with_metrics = run_delight_and_get_angular_metrics(initial_data.copy(), DATA_DIR)
        
        if 'angular_offset_arcsec' not in data_with_metrics.columns:
            print("\n🚨 Final metrics calculation halted.")
        else:
            # --- 5. Output and Interpretation ---
            
            # Include host size and normalized offset in output if available
            out_cols = ['angular_offset_arcsec']
            if 'host_size_arcsec' in data_with_metrics.columns:
                out_cols.append('host_size_arcsec')
            if 'angular_normalized_offset' in data_with_metrics.columns:
                out_cols.append('angular_normalized_offset')
            final_output = data_with_metrics[out_cols]
            
            print("\n--- ✅ Final Angular Metrics (No Redshift Required) ---")
            print("Note: Values are in arcseconds (arcsec) or dimensionless.")
            print(final_output)
            
            # TDE Classification Context: prefer normalized offset if available, else angular only
            print("\n--- TDE Classification Context (Angular) ---")
            if 'angular_normalized_offset' in final_output.columns:
                tde_candidates = final_output[
                    (final_output['angular_offset_arcsec'] <= 1.0) & 
                    (final_output['angular_normalized_offset'] <= 1.0)
                ]
                if not tde_candidates.empty:
                    print(f"**{len(tde_candidates)} objects show characteristics of nuclear events (small $\\theta$ and $\\theta/a$):**")
                    print(tde_candidates)
                else:
                    print("No objects meet the strict angular and normalized criteria.")
            else:
                tde_candidates = final_output[final_output['angular_offset_arcsec'] <= 1.0]
                if not tde_candidates.empty:
                    print(f"**{len(tde_candidates)} objects show small angular offsets (\\theta <= 1.0 arcsec):**")
                    print(tde_candidates)
                else:
                    print("No objects meet the small angular offset criterion (theta <= 1.0 arcsec).")