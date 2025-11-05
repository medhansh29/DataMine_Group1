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
    # Create a mapping from oid to distance (assuming same distance for all rows with same oid)
    distance_dict = df.groupby('oid')['distance'].first().to_dict()
    
    unique_object_ids = df['oid'].unique()
    
    print(f"Processing {len(unique_object_ids)} unique objects...")
    
    # Process each unique object
    for idx, object_id in enumerate(unique_object_ids):
        print(f"\n[{idx+1}/{len(unique_object_ids)}] Processing {object_id}...")
        
        try:
            # Get distance for this object
            distance = distance_dict.get(object_id)
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
            
            # Calculate peak luminosity: L = 4π * d² * F
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
    for i, object_id in enumerate(unique_object_ids):
        result_dict[object_id] = {
            'peak_luminosity': peak_luminosities[i],
            'r_squared': r_squared_values[i]
        }
    
    # Add columns to original dataframe
    df['peak_luminosity'] = df['oid'].map(lambda x: result_dict.get(x, {}).get('peak_luminosity'))
    df['r_squared'] = df['oid'].map(lambda x: result_dict.get(x, {}).get('r_squared'))
    
    df.to_csv(csv_file, index=False)
    
    return df


