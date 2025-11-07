import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from alerce.core import Alerce

def filter_curve(csv_file):
    """
    Process a CSV file with object IDs, fetch light curve data from Alerce,
    calculate peak luminosity from flux and distance, and calculate r^2 for t^(-5/3) decay fit.
    
    Parameters:
    csv_file: Path to CSV file with 'oid' and 'distance' columns
    
    Returns:
    DataFrame: Original CSV with added 'peak_luminosity' and 'r_squared' columns
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize Alerce client
    alerce_client = Alerce()
    
    # Lists to store results
    peak_luminosities = []
    r_squared_values = []
    
    # Get unique object IDs and their distances
    # Support either 'distance' or 'distance_mpc' column
    distance_col = 'distance' if 'distance' in df.columns else ('distance_mpc' if 'distance_mpc' in df.columns else None)
    if distance_col is None:
        print("  Warning: No distance column found ('distance' or 'distance_mpc'). Peak luminosity will be N/A.")
    # Create a mapping from oid to distance (assuming same distance for all rows with same oid)
    distance_dict = df.groupby('oid')[distance_col].first().to_dict() if distance_col else {}
    
    unique_object_ids = df['oid'].unique()
    
    # Initialize columns if they don't exist
    if 'peak_luminosity' not in df.columns:
        df['peak_luminosity'] = None
    if 'r_squared' not in df.columns:
        df['r_squared'] = None
    
    # Filter to only process objects missing data
    # Check if ALL rows for an OID have data (not just any row)
    objects_to_process = []
    for oid in unique_object_ids:
        oid_rows = df[df['oid'] == oid]
        # Check if ALL rows for this OID have both peak_luminosity and r_squared
        all_have_peak = oid_rows['peak_luminosity'].notna().all()
        all_have_r2 = oid_rows['r_squared'].notna().all()
        # Process if ANY row is missing data
        if not (all_have_peak and all_have_r2):
            objects_to_process.append(oid)
    
    if not objects_to_process:
        print(f"All {len(unique_object_ids)} unique objects already have peak_luminosity and r_squared data in all rows. Skipping processing.")
        return df
    
    print(f"Processing {len(objects_to_process)} objects (skipping {len(unique_object_ids) - len(objects_to_process)} with existing data in all rows)...")
    
    # Process each unique object
    for idx, object_id in enumerate(objects_to_process):
        print(f"\n[{idx+1}/{len(objects_to_process)}] Processing {object_id}...")
        
        try:
            # Get distance for this object
            distance = distance_dict.get(object_id) if distance_col else None
            has_distance = distance is not None and not np.isnan(distance)
            
            if not has_distance:
                print(f"  No valid distance found for {object_id}, will use 'N/A' for peak luminosity")
            
            # Fetch light curve data from Alerce
            light_curve_df = alerce_client.query_detections(object_id, format="pandas")
            
            if light_curve_df is None or light_curve_df.empty:
                print(f"  No data found for {object_id}")
                peak_luminosities.append("N/A" if not has_distance else None)
                r_squared_values.append(None)
                continue
            
            # Convert magnitude to flux: flux = 10^(-0.4 * mag)
            light_curve_df['flux'] = 10**(-0.4 * light_curve_df['magpsf'])
            
            # Sort data by time
            data_sorted = light_curve_df.sort_values('mjd').reset_index(drop=True)
            
            # Find peak luminosity (maximum flux)
            peak_idx = data_sorted['flux'].idxmax()
            max_flux = data_sorted.loc[peak_idx, 'flux']
            peak_time = data_sorted.loc[peak_idx, 'mjd']
            
            # Calculate peak luminosity: L = 4π * d² * F (units per input distance)
            # where L is luminosity, d is distance, F is flux
            if has_distance:
                peak_luminosity = 4 * np.pi * (distance ** 2) * max_flux
                print(f"  Maximum flux: {max_flux:.6e} at MJD {peak_time:.2f}")
                print(f"  Peak luminosity: {peak_luminosity:.6e} (distance: {distance:.2e})")
            else:
                peak_luminosity = "N/A"
                print(f"  Maximum flux: {max_flux:.6e} at MJD {peak_time:.2f}")
                print(f"  Peak luminosity: N/A (no distance available)")
            
            # Get data after the peak
            decay_data = data_sorted.iloc[peak_idx+1:].copy()
            
            if len(decay_data) < 3:
                print(f"  Insufficient decay data ({len(decay_data)} points)")
                peak_luminosities.append(peak_luminosity)
                r_squared_values.append(None)
                continue
            
            # Convert time to days since peak
            decay_data['days_since_peak'] = decay_data['mjd'] - peak_time
            
            # Remove any data points with days_since_peak <= 0
            decay_data = decay_data[decay_data['days_since_peak'] > 0]
            
            if len(decay_data) < 3:
                print(f"  No valid decay data after peak")
                peak_luminosities.append(peak_luminosity)
                r_squared_values.append(None)
                continue
            
            # Define the t^(-5/3) power law function
            def power_law(t, A, t0):
                """A * (t + t0)^(-5/3)"""
                return A * (t + t0)**(-5/3)
            
            # Fit t^(-5/3) power law using flux
            # Initial guess: A = max_flux, t0 = 1 day
            popt_power, _ = curve_fit(power_law, decay_data['days_since_peak'], 
                                     decay_data['flux'], 
                                     p0=[max_flux, 1.0],
                                     maxfev=2000)
            
            # Calculate R² for power law fit
            y_pred_power = power_law(decay_data['days_since_peak'], *popt_power)
            ss_res_power = np.sum((decay_data['flux'] - y_pred_power) ** 2)
            ss_tot_power = np.sum((decay_data['flux'] - np.mean(decay_data['flux'])) ** 2)
            r2_power = 1 - (ss_res_power / ss_tot_power)
            
            print(f"  R² value: {r2_power:.4f}")
            
            peak_luminosities.append(peak_luminosity)
            r_squared_values.append(r2_power)
            
        except Exception as e:
            print(f"  Error processing {object_id}: {str(e)}")
            # Check if distance exists for this object to determine what to append
            distance = distance_dict.get(object_id)
            has_distance = distance is not None and not np.isnan(distance)
            peak_luminosities.append("N/A" if not has_distance else None)
            r_squared_values.append(None)
    
    # Add results to dataframe based on oid
    result_dict = {}
    for i, object_id in enumerate(objects_to_process):
        result_dict[object_id] = {
            'peak_luminosity': peak_luminosities[i],
            'r_squared': r_squared_values[i]
        }
    
    # Update only rows for processed objects (preserve existing data for others)
    for oid, results in result_dict.items():
        mask = df['oid'] == oid
        if results['peak_luminosity'] is not None:
            df.loc[mask, 'peak_luminosity'] = results['peak_luminosity']
        if results['r_squared'] is not None:
            df.loc[mask, 'r_squared'] = results['r_squared']
    
    df.to_csv(csv_file, index=False)
    
    return df


