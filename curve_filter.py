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
            
            # CRITICAL: Filter to only r-band (fid=2) for consistent fitting
            # Cannot mix r-band and g-band data as they have different magnitudes/fluxes
            # TDEs are typically analyzed in r-band
            r_band_df = light_curve_df[light_curve_df['fid'] == 2].copy()
            
            if r_band_df.empty:
                print(f"  No r-band (fid=2) data found for {object_id}")
                peak_luminosities.append("N/A" if not has_distance else None)
                r_squared_values.append(None)
                continue
            
            # Robust magnitude selection (prefer corrected magnitude if available)
            if 'magpsf_corr' in r_band_df.columns:
                mag_col = 'magpsf_corr'
            elif 'magpsf' in r_band_df.columns:
                mag_col = 'magpsf'
            else:
                print(f"  No valid magnitude column found for {object_id}")
                peak_luminosities.append("N/A" if not has_distance else None)
                r_squared_values.append(None)
                continue
            
            # Convert magnitude to flux: flux = 10^(-0.4 * mag)
            r_band_df['flux'] = 10**(-0.4 * r_band_df[mag_col])
            
            # Sort r-band data by time
            r_band_sorted = r_band_df.sort_values('mjd').reset_index(drop=True)
            
            # Find peak luminosity (maximum flux in r-band)
            peak_idx = r_band_sorted['flux'].idxmax()
            max_flux = r_band_sorted.loc[peak_idx, 'flux']
            peak_time = r_band_sorted.loc[peak_idx, 'mjd']
            
            # Calculate peak luminosity: L = 4π * d² * F (units per input distance)
            # where L is luminosity, d is distance, F is flux
            if has_distance:
                peak_luminosity = 4 * np.pi * (distance ** 2) * max_flux
                print(f"  Maximum r-band flux: {max_flux:.6e} at MJD {peak_time:.2f}")
                print(f"  Peak luminosity: {peak_luminosity:.6e} (distance: {distance:.2e})")
            else:
                peak_luminosity = "N/A"
                print(f"  Maximum r-band flux: {max_flux:.6e} at MJD {peak_time:.2f}")
                print(f"  Peak luminosity: N/A (no distance available)")
            
            # Get r-band data after the peak (for decay fitting)
            decay_data = r_band_sorted.iloc[peak_idx+1:].copy()
            
            if len(decay_data) < 3:
                print(f"  Insufficient r-band decay data ({len(decay_data)} points)")
                peak_luminosities.append(peak_luminosity)
                r_squared_values.append(None)
                continue
            
            # Convert time to days since peak
            decay_data['days_since_peak'] = decay_data['mjd'] - peak_time
            
            # Remove any data points with days_since_peak <= 0
            decay_data = decay_data[decay_data['days_since_peak'] > 0]
            
            if len(decay_data) < 3:
                print(f"  No valid r-band decay data after peak")
                peak_luminosities.append(peak_luminosity)
                r_squared_values.append(None)
                continue
            
            # Define the t^(-5/3) power law function
            def power_law(t, A, t0):
                """A * (t + t0)^(-5/3)"""
                # Ensure t + t0 is positive to avoid numerical issues
                return A * np.maximum(t + t0, 1e-10)**(-5/3)
            
            # ROBUST FITTING: Iteratively reweighted least squares with outlier rejection
            # This handles scatter, rebrightening, and non-monotonic decay
            try:
                # Step 1: Filter obvious rebrightening first (points that increase significantly after peak)
                # This helps identify the true decay phase
                filtered_data = decay_data.copy()
                
                # Identify rebrightening: points where flux increases by >5% relative to previous point
                if len(filtered_data) > 1:
                    flux_diff = filtered_data['flux'].diff()
                    flux_relative_change = flux_diff / filtered_data['flux'].shift(1)
                    # Flag significant increases (rebrightening) - use 5% threshold
                    rebrightening = (flux_diff > 0) & (flux_relative_change > 0.05)
                    # Keep first point (can't have rebrightening)
                    rebrightening.iloc[0] = False
                    
                    # Remove obvious rebrightening points
                    if rebrightening.sum() > 0:
                        filtered_data = filtered_data[~rebrightening].copy()
                        if len(filtered_data) < 3:
                            # If too many rebrightening points, use original data
                            filtered_data = decay_data.copy()
                
                # Step 2: Initial fit to filtered data
                if len(filtered_data) < 3:
                    filtered_data = decay_data.copy()
                
                popt_power, _ = curve_fit(power_law, filtered_data['days_since_peak'], 
                                         filtered_data['flux'], 
                                         p0=[max_flux, 1.0],
                                         bounds=([0, 0], [np.inf, np.inf]),
                                         maxfev=2000)
                
                # Step 3: Iterative outlier rejection (up to 3 iterations)
                # Use median absolute deviation (MAD) for more robust outlier detection
                n_iterations = 3
                best_r2 = -np.inf
                best_data = filtered_data.copy()
                best_popt = popt_power
                
                for iteration in range(n_iterations):
                    if len(filtered_data) < 3:
                        break
                    
                    # Predict flux for current data
                    y_pred = power_law(filtered_data['days_since_peak'], *popt_power)
                    
                    # Calculate residuals
                    residuals = filtered_data['flux'] - y_pred
                    
                    # Use median absolute deviation (MAD) for robust outlier detection
                    median_residual = np.median(np.abs(residuals))
                    mad = np.median(np.abs(residuals - np.median(residuals)))
                    # Use 3 MAD for outlier threshold (more robust than std)
                    if mad > 0:
                        outlier_threshold = 3.0 * mad
                    else:
                        # Fallback to std if MAD is zero
                        outlier_threshold = 2.5 * np.std(residuals)
                    
                    # Flag outliers: points far from the fit
                    # Prioritize removing points above the fit (rebrightening)
                    residuals_abs = np.abs(residuals)
                    is_outlier = residuals_abs > outlier_threshold
                    
                    # Preferentially remove points above the fit (rebrightening)
                    # but be more lenient with points below (could be noise)
                    above_fit = residuals > 0
                    is_outlier_above = (residuals > outlier_threshold * 0.7) & above_fit
                    is_outlier = is_outlier | is_outlier_above
                    
                    n_outliers = is_outlier.sum()
                    
                    if n_outliers == 0:
                        # No more outliers, stop iterating
                        break
                    
                    # Remove outliers and refit
                    filtered_data = filtered_data[~is_outlier].copy()
                    
                    if len(filtered_data) < 3:
                        # Not enough data after filtering, use previous best
                        break
                    
                    # Refit with filtered data
                    try:
                        popt_power, _ = curve_fit(power_law, 
                                                 filtered_data['days_since_peak'], 
                                                 filtered_data['flux'], 
                                                 p0=popt_power,  # Use previous fit as starting point
                                                 bounds=([0, 0], [np.inf, np.inf]),
                                                 maxfev=2000)
                        
                        # Check if this fit is better
                        y_pred_test = power_law(filtered_data['days_since_peak'], *popt_power)
                        ss_res_test = np.sum((filtered_data['flux'] - y_pred_test) ** 2)
                        ss_tot_test = np.sum((filtered_data['flux'] - np.mean(filtered_data['flux'])) ** 2)
                        
                        if ss_tot_test > 0:
                            r2_test = 1 - (ss_res_test / ss_tot_test)
                            if r2_test > best_r2:
                                best_r2 = r2_test
                                best_data = filtered_data.copy()
                                best_popt = popt_power
                    except (RuntimeError, ValueError):
                        # If refit fails, use previous best
                        break
                
                # Step 4: Use best fit (either from iterations or initial)
                if best_r2 > -np.inf:
                    filtered_data = best_data
                    popt_power = best_popt
                
                # Step 5: Calculate R² using best filtered data
                if len(filtered_data) >= 3:
                    y_pred_power = power_law(filtered_data['days_since_peak'], *popt_power)
                    ss_res_power = np.sum((filtered_data['flux'] - y_pred_power) ** 2)
                    ss_tot_power = np.sum((filtered_data['flux'] - np.mean(filtered_data['flux'])) ** 2)
                    
                    if ss_tot_power > 0:
                        r2_power = 1 - (ss_res_power / ss_tot_power)
                    else:
                        r2_power = np.nan
                    
                    n_filtered = len(decay_data) - len(filtered_data)
                    if n_filtered > 0:
                        print(f"  R² value (r-band only, robust fit): {r2_power:.4f} (filtered {n_filtered} outliers/rebrightening)")
                    else:
                        print(f"  R² value (r-band only, robust fit): {r2_power:.4f}")
                else:
                    # Fallback: use original fit if filtering removed too much data
                    y_pred_power = power_law(decay_data['days_since_peak'], *popt_power)
                    ss_res_power = np.sum((decay_data['flux'] - y_pred_power) ** 2)
                    ss_tot_power = np.sum((decay_data['flux'] - np.mean(decay_data['flux'])) ** 2)
                    
                    if ss_tot_power > 0:
                        r2_power = 1 - (ss_res_power / ss_tot_power)
                    else:
                        r2_power = np.nan
                    
                    print(f"  R² value (r-band only, fallback): {r2_power:.4f} (insufficient data after filtering)")
                
            except (RuntimeError, ValueError) as e:
                print(f"  Error fitting power law: {e}")
                r2_power = None
            
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


