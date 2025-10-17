import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

def filter_curve(data, object_id=None):
    """
    Check if the light curve follows a t^(-5/3) decay law after peak luminosity.
    
    Parameters:
    data: DataFrame with columns ['mjd', 'magpsf', 'oid'] or similar
    object_id: Optional object identifier for logging
    
    Returns:
    dict: Results including peak info, decay slope, and whether it matches t^(-5/3)
    """
    
    # Sort data by time
    data_sorted = data.sort_values('mjd').reset_index(drop=True)
    
    # Find peak luminosity (brightest magnitude = minimum magnitude value)
    peak_idx = data_sorted['magpsf'].idxmin()
    peak_time = data_sorted.loc[peak_idx, 'mjd']
    peak_mag = data_sorted.loc[peak_idx, 'magpsf']
    
    # Get data after the peak
    decay_data = data_sorted.iloc[peak_idx+1:].copy()
    
    if len(decay_data) < 10:
        return {
            'object_id': object_id,
            'peak_time': peak_time,
            'peak_mag': peak_mag,
            'decay_points': len(decay_data),
            'slope': None,
            'expected_slope': -5/3,
            'matches_t53': False,
            'error': 'Insufficient decay data'
        }
    
    # Convert time to days since peak
    decay_data['days_since_peak'] = decay_data['mjd'] - peak_time
    
    # Remove any data points with days_since_peak <= 0 (shouldn't happen but safety check)
    decay_data = decay_data[decay_data['days_since_peak'] > 0]
    
    if len(decay_data) < 3:
        return {
            'object_id': object_id,
            'peak_time': peak_time,
            'peak_mag': peak_mag,
            'decay_points': len(decay_data),
            'slope': None,
            'expected_slope': -5/3,
            'matches_t53': False,
            'error': 'No valid decay data after peak'
        }
    
    # Define the t^(-5/3) power law function
    def power_law(t, A, t0):
        """A * (t + t0)^(-5/3)"""
        return A * (t + t0)**(-5/3)
    
    # Define a linear decay function for comparison
    def linear_decay(t, m, b):
        """m * t + b"""
        return m * t + b
    
    try:
        # Fit t^(-5/3) power law
        # Initial guess: A = peak_mag, t0 = 1 day
        popt_power, _ = curve_fit(power_law, decay_data['days_since_peak'], 
                                 decay_data['magpsf'], 
                                 p0=[peak_mag, 1.0],
                                 maxfev=2000)
        
        # Calculate R² for power law fit
        y_pred_power = power_law(decay_data['days_since_peak'], *popt_power)
        ss_res_power = np.sum((decay_data['magpsf'] - y_pred_power) ** 2)
        ss_tot_power = np.sum((decay_data['magpsf'] - np.mean(decay_data['magpsf'])) ** 2)
        r2_power = 1 - (ss_res_power / ss_tot_power)
        
        # Fit linear decay for comparison
        popt_linear, _ = curve_fit(linear_decay, decay_data['days_since_peak'], 
                                  decay_data['magpsf'])
        
        # Calculate R² for linear fit
        y_pred_linear = linear_decay(decay_data['days_since_peak'], *popt_linear)
        ss_res_linear = np.sum((decay_data['magpsf'] - y_pred_linear) ** 2)
        ss_tot_linear = np.sum((decay_data['magpsf'] - np.mean(decay_data['magpsf'])) ** 2)
        r2_linear = 1 - (ss_res_linear / ss_tot_linear)
        
        # Determine if t^(-5/3) is a good fit
        # Consider it a match if R² > 0.7 and power law fits better than linear
        matches_t53 = (r2_power > 0.7) and (r2_power > r2_linear)
        
        return {
            'object_id': object_id,
            'peak_time': peak_time,
            'peak_mag': peak_mag,
            'decay_points': len(decay_data),
            'power_law_params': popt_power,
            'power_law_r2': r2_power,
            'linear_slope': popt_linear[0],
            'linear_r2': r2_linear,
            'expected_slope': -5/3,
            'matches_t53': matches_t53,
            'decay_data': decay_data
        }
        
    except Exception as e:
        return {
            'object_id': object_id,
            'peak_time': peak_time,
            'peak_mag': peak_mag,
            'decay_points': len(decay_data),
            'slope': None,
            'expected_slope': -5/3,
            'matches_t53': False,
            'error': f'Fitting failed: {str(e)}'
        }

def analyze_light_curves_from_csv(csv_file):
    """
    Analyze all light curves in a CSV file for t^(-5/3) decay.
    
    Parameters:
    csv_file: Path to CSV file with light curve data
    
    Returns:
    DataFrame: Results for each object
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Group by object ID and analyze each
    results = []
    
    for oid in df['oid'].unique():
        if pd.isna(oid) or oid == '':
            continue
            
        object_data = df[df['oid'] == oid].copy()
        
        # Convert magnitude to flux-like quantity for analysis
        # Brighter objects have lower magnitudes, so we'll work with magnitude directly
        result = filter_curve(object_data, oid)
        results.append(result)
    
    return pd.DataFrame(results)
