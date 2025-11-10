import pandas as pd
import numpy as np
from alerce.core import Alerce
from tqdm import tqdm 
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from scipy.optimize import curve_fit
import warnings 

# --- 1. Configuration (UNCHANGED) ---
alerce = Alerce()

# Confirmed Class Names
TARGET_CLASSIFIERS = {
    'SNIa': 'lc_classifier_BHRF_forced_phot_transient', 
    'SNII': 'lc_classifier_BHRF_forced_phot_transient', # NEW
    'QSO': 'lc_classifier_BHRF_forced_phot_stochastic', 
    'AGN': 'lc_classifier_BHRF_forced_phot_stochastic'
}
CLASSIFIER_VERSION = '2.1.0' 
MAX_SAMPLES_PER_CLASS = 500

# Parallelization settings
MAX_WORKERS = 20  # Number of concurrent requests (adjust based on API rate limits)
RATE_LIMIT = 0.05  # Seconds between requests per worker (lower = faster, but may hit rate limits) 

# --- 2. Feature Extraction Utilities (UNCHANGED) ---

def get_oids_by_class(class_name: str, classifier_name: str, max_count: int) -> list:
    """Retrieves OIDs using the dedicated classifier name and version."""
    print(f"-> Querying OIDs for class: {class_name} using {classifier_name}/{CLASSIFIER_VERSION}...")
    
    probability_threshold = 0.5 if class_name == 'SNIa' else 0.8 

    try:
        result_df = alerce.query_objects(
            classifier=classifier_name,
            classifier_version=CLASSIFIER_VERSION, 
            class_name=class_name,
            probability=probability_threshold, 
            page_size=max_count,
            format='pandas'
        )
        
        if result_df.empty or 'oid' not in result_df.columns:
            print(f"   Warning: Query returned empty or invalid DataFrame for {class_name}.")
            return []
            
        oids = result_df['oid'].tolist()
        print(f"   Found {len(oids)} OIDs for {class_name} (P > {probability_threshold}).")
        return oids
    except Exception as e:
        print(f"   Error querying OIDs for {class_name}. Actual Error: {e}")
        return []

def _fetch_single_oid_features(oid: str) -> Optional[pd.Series]:
    """Helper function to fetch features for a single OID (used in parallel execution)."""
    try:
        df_feat = alerce.query_features(oid=oid, format='pandas')
        
        if not df_feat.empty:
            df_feat_clean = df_feat.drop_duplicates(subset=['name'], keep='first')
            feature_series = df_feat_clean.set_index('name')['value']
            feature_series['oid'] = oid
            return feature_series
        return None
    except Exception as e:
        print(f"   [Feature Retrieval Error for {oid}]: {e}")
        return None

async def _fetch_features_async(oids: List[str], max_workers: int = 10, rate_limit: float = 0.1) -> List[pd.Series]:
    """Fetch features for multiple OIDs in parallel using ThreadPoolExecutor."""
    # Create a ThreadPoolExecutor for running sync I/O operations
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        loop = asyncio.get_running_loop()
        
        # Create a semaphore to limit concurrent requests (rate limiting)
        semaphore = asyncio.Semaphore(max_workers)
        
        async def fetch_with_rate_limit(oid: str) -> Optional[pd.Series]:
            async with semaphore:
                # Run the sync function in the thread pool
                result = await loop.run_in_executor(executor, _fetch_single_oid_features, oid)
                # Rate limiting
                await asyncio.sleep(rate_limit)
                return result
        
        # Create all tasks
        tasks = [fetch_with_rate_limit(oid) for oid in oids]
        
        # Execute with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching ALeRCE features"):
            result = await coro
            if result is not None:
                results.append(result)
        
        return results
    finally:
        executor.shutdown(wait=True)

def retrieve_and_pivot_features(oids: list, max_workers: int = 10, rate_limit: float = 0.1) -> pd.DataFrame:
    """Retrieves features, handles duplicates, pivots, and concatenates using async parallelization."""
    print(f"-> Retrieving and pivoting features for {len(oids)} objects...")
    print(f"   Using {max_workers} parallel workers with {rate_limit}s rate limit")
    
    # Run async function
    try:
        all_features = asyncio.run(_fetch_features_async(oids, max_workers, rate_limit))
    except RuntimeError:
        # If there's already an event loop running, use nest_asyncio or create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_features = loop.run_until_complete(_fetch_features_async(oids, max_workers, rate_limit))
        loop.close()

    if not all_features:
        return pd.DataFrame()

    features_df = pd.concat(all_features, axis=1).T.set_index('oid')
    return features_df

# --- 3. Custom TDE Feature Calculation (FIXED) ---

def calculate_custom_tde_features(oid: str) -> dict:
    """
    Calculates TDE-specific features including:
    - max_mag_r: Peak magnitude in r-band
    - peak_mjd: Time of peak magnitude
    - rise_duration: Duration from first detection to peak
    - r_squared_t53: R² for t^(-5/3) decay fit (key TDE signature)
    """
    
    try:
        df_det = alerce.query_detections(oid=oid, format='pandas')
    except Exception as e:
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan, 
            'r_squared_t53': np.nan,
            'det_error': str(e)
        }

    if df_det.empty:
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan,
            'r_squared_t53': np.nan
        }

    # --- ROBUST MAGNITUDE SELECTION ---
    # 1. Check for the corrected magnitude column
    if 'magpsf_corr' in df_det.columns:
        mag_col = 'magpsf_corr'
    # 2. Fall back to the uncorrected magnitude column
    elif 'magpsf' in df_det.columns:
        mag_col = 'magpsf'
    else:
        # If neither is available, we cannot calculate features
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan,
            'r_squared_t53': np.nan,
            'det_error': "No valid magnitude column (magpsf_corr or magpsf) found."
        }
    # -----------------------------------

    # Focus on r-band (fid=2)
    r_band_dets = df_det[df_det['fid'] == 2].copy()

    if r_band_dets.empty:
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan,
            'r_squared_t53': np.nan
        }

    # Sort by MJD
    r_band_dets = r_band_dets.sort_values('mjd').reset_index(drop=True)

    # Use the selected magnitude column for calculation
    peak_mag = r_band_dets[mag_col].min()
    
    if not np.isfinite(peak_mag):
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan,
            'r_squared_t53': np.nan
        }
    
    # Find peak row
    peak_row = r_band_dets[r_band_dets[mag_col] == peak_mag].iloc[0]
    peak_mjd = peak_row['mjd']
    peak_idx = r_band_dets[r_band_dets[mag_col] == peak_mag].index[0]
    first_mjd = r_band_dets['mjd'].min()
    rise_duration = peak_mjd - first_mjd
    
    # --- Calculate R² for t^(-5/3) decay fit ---
    r_squared_t53 = np.nan
    
    # Calculate flux from magnitude (ALeRCE doesn't provide flux directly)
    # flux = 10^(-0.4 * mag) for ZTF (using magpsf or magpsf_corr)
    if mag_col in r_band_dets.columns:
        # Calculate flux from magnitude
        r_band_dets = r_band_dets.copy()
        r_band_dets['flux'] = 10**(-0.4 * r_band_dets[mag_col])
    
    # Check if flux column exists (needed for fitting)
    if 'flux' in r_band_dets.columns:
        try:
            # Get data after the peak
            decay_data = r_band_dets.iloc[peak_idx+1:].copy()
            
            if len(decay_data) >= 3:  # Need at least 3 points for fitting
                # Convert time to days since peak
                decay_data = decay_data.copy()
                decay_data['days_since_peak'] = decay_data['mjd'] - peak_mjd
                
                # Remove any data points with days_since_peak <= 0 or invalid flux
                decay_data = decay_data[
                    (decay_data['days_since_peak'] > 0) & 
                    (decay_data['flux'].notna()) & 
                    (decay_data['flux'] > 0)
                ].copy()
                
                if len(decay_data) >= 3:
                    # Get flux and time arrays
                    t_decay = decay_data['days_since_peak'].values
                    flux_decay = decay_data['flux'].values
                    
                    # Remove any infinite or NaN values
                    valid_mask = np.isfinite(t_decay) & np.isfinite(flux_decay) & (t_decay > 0) & (flux_decay > 0)
                    t_decay = t_decay[valid_mask]
                    flux_decay = flux_decay[valid_mask]
                    
                    if len(t_decay) >= 3:
                        # Define the t^(-5/3) power law function
                        def power_law(t, A, t0):
                            """A * (t + t0)^(-5/3)"""
                            # Ensure t + t0 > 0 to avoid negative base
                            return A * np.power(np.maximum(t + t0, 1e-6), -5/3)
                        
                        # Initial guess: A = max_flux, t0 = 1 day
                        max_flux = np.max(flux_decay)
                        try:
                            # Suppress optimization warnings during fitting
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                # Fit t^(-5/3) power law using flux
                                # Use bounds to ensure positive parameters and avoid numerical issues
                                popt_power, _ = curve_fit(
                                    power_law, 
                                    t_decay, 
                                    flux_decay, 
                                    p0=[max_flux, 1.0],
                                    bounds=([0, 0], [np.inf, np.inf]),
                                    maxfev=2000
                                )
                            
                            # Calculate R² for power law fit
                            y_pred_power = power_law(t_decay, *popt_power)
                            ss_res_power = np.sum((flux_decay - y_pred_power) ** 2)
                            ss_tot_power = np.sum((flux_decay - np.mean(flux_decay)) ** 2)
                            
                            if ss_tot_power > 0:
                                r_squared_t53 = 1 - (ss_res_power / ss_tot_power)
                            else:
                                r_squared_t53 = np.nan
                                
                        except (RuntimeError, ValueError) as fit_error:
                            # Fit failed, leave r_squared_t53 as NaN
                            r_squared_t53 = np.nan
                        except Exception as fit_error:
                            # Any other error during fitting
                            r_squared_t53 = np.nan
        except Exception as e:
            # Error calculating R², leave as NaN
            r_squared_t53 = np.nan
    else:
        # No flux column available (shouldn't happen after our calculation, but safety check)
        r_squared_t53 = np.nan
    
    return {
        'max_mag_r': peak_mag, 
        'peak_mjd': peak_mjd, 
        'rise_duration': rise_duration,
        'r_squared_t53': r_squared_t53
    }


def _calculate_single_custom_features(oid: str) -> Dict:
    """Helper function to calculate custom features for a single OID (used in parallel execution)."""
    features = calculate_custom_tde_features(oid)
    features['oid'] = oid
    return features

async def _calculate_custom_features_async(oids: List[str], max_workers: int = 10, rate_limit: float = 0.1) -> List[Dict]:
    """Calculate custom features for multiple OIDs in parallel using ThreadPoolExecutor."""
    # Create a ThreadPoolExecutor for running sync I/O operations
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        loop = asyncio.get_running_loop()
        
        # Create a semaphore to limit concurrent requests (rate limiting)
        semaphore = asyncio.Semaphore(max_workers)
        
        async def calculate_with_rate_limit(oid: str) -> Dict:
            async with semaphore:
                # Run the sync function in the thread pool
                result = await loop.run_in_executor(executor, _calculate_single_custom_features, oid)
                # Rate limiting
                await asyncio.sleep(rate_limit)
                return result
        
        # Create all tasks
        tasks = [calculate_with_rate_limit(oid) for oid in oids]
        
        # Execute with progress bar
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Calculating custom features"):
            result = await coro
            results.append(result)
        
        return results
    finally:
        executor.shutdown(wait=True)

def add_custom_features_to_df(features_df: pd.DataFrame, max_workers: int = 10, rate_limit: float = 0.1) -> pd.DataFrame:
    """Iterates through the feature DataFrame and adds custom TDE features using async parallelization."""
    
    oids = features_df.index.tolist()
    
    print(f"\n-> Calculating custom TDE features for {len(oids)} objects...")
    print(f"   Using {max_workers} parallel workers with {rate_limit}s rate limit")
    
    # Run async function
    try:
        custom_features_list = asyncio.run(_calculate_custom_features_async(oids, max_workers, rate_limit))
    except RuntimeError:
        # If there's already an event loop running, use nest_asyncio or create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        custom_features_list = loop.run_until_complete(_calculate_custom_features_async(oids, max_workers, rate_limit))
        loop.close()
        
    custom_df = pd.DataFrame(custom_features_list).set_index('oid')
    merged_df = features_df.join(custom_df, how='inner')
    
    return merged_df

# --- 4. Main Execution (UNCHANGED) ---

if __name__ == '__main__':
    
    all_background_oids = []
    
    # A. Retrieve OIDs for all target classes
    for cls, classifier_name in TARGET_CLASSIFIERS.items():
        oids = get_oids_by_class(cls, classifier_name, MAX_SAMPLES_PER_CLASS)
        all_background_oids.extend(oids)

    if not all_background_oids:
        print("\n❌ Failed to retrieve any background OIDs. Exiting.")
    else:
        print(f"\nTotal OIDs retrieved for training: {len(all_background_oids)}")
        
        # B. Retrieve and Pivot General Features (from query_features)
        raw_background_df = retrieve_and_pivot_features(all_background_oids, max_workers=MAX_WORKERS, rate_limit=RATE_LIMIT)
        
        if raw_background_df.empty:
            print("\n❌ Feature retrieval failed. Exiting.")
        else:
            print(f"\nSuccessfully retrieved features for {len(raw_background_df)} background objects.")
            
            # C. Calculate and Merge Custom TDE Features
            final_background_df = add_custom_features_to_df(raw_background_df, max_workers=MAX_WORKERS, rate_limit=RATE_LIMIT)
            
            # D. Final Cleaning and Output
            # IMPORTANT: Preserve r_squared_t53 even if it has many NaNs (it's a key TDE feature)
            # Only drop columns that are >90% NaN AND are not critical TDE features
            critical_features = ['r_squared_t53', 'max_mag_r', 'peak_mjd', 'rise_duration']
            nan_threshold = len(final_background_df) * 0.90
            
            # Get columns to drop (exclude critical features)
            cols_to_drop = []
            for col in final_background_df.columns:
                if col not in critical_features:
                    nan_count = final_background_df[col].isna().sum()
                    if nan_count > nan_threshold:
                        cols_to_drop.append(col)
            
            if cols_to_drop:
                print(f"\n-> Dropping {len(cols_to_drop)} columns with >90% NaN values (excluding critical TDE features)")
                final_background_df = final_background_df.drop(columns=cols_to_drop)
            
            # Fill NaN values: use mean for numeric columns, but preserve NaN for r_squared_t53 if all are NaN
            # (we'll fill it with 0 later if needed for model training)
            for col in final_background_df.columns:
                if col != 'is_tde_anomaly' and col not in critical_features:
                    if final_background_df[col].isna().any():
                        final_background_df[col] = final_background_df[col].fillna(final_background_df[col].mean())
            
            # For r_squared_t53: fill NaN with 0 (indicates no valid fit, which is informative)
            if 'r_squared_t53' in final_background_df.columns:
                r53_nan_count = final_background_df['r_squared_t53'].isna().sum()
                if r53_nan_count > 0:
                    print(f"   Note: {r53_nan_count} objects have NaN for r_squared_t53 (no valid decay fit), filling with 0")
                    final_background_df['r_squared_t53'] = final_background_df['r_squared_t53'].fillna(0.0)
            
            final_background_df['is_tde_anomaly'] = 0 
            
            print("\n--- ✅ Final Background Training Data Summary ---")
            print(f"Total Objects (Training Set Size): {len(final_background_df)}")
            print(f"Total Features (Model Input Size): {len(final_background_df.columns)}")
            
            cols_to_show = ['max_mag_r', 'rise_duration', 'r_squared_t53', 'Amplitude', 'Std', 'Gskew']
            available_cols = [col for col in cols_to_show if col in final_background_df.columns]
            
            print("\nFinal DataFrame Head (Sample Features):")
            print(final_background_df[available_cols].head())
            
            # Show summary of r_squared_t53 if available
            if 'r_squared_t53' in final_background_df.columns:
                r53_stats = final_background_df['r_squared_t53'].describe()
                print("\n--- R² (t^(-5/3) decay fit) Statistics ---")
                print(f"Valid fits: {(final_background_df['r_squared_t53'].notna()).sum()} / {len(final_background_df)}")
                print(f"Mean R²: {final_background_df['r_squared_t53'].mean():.4f}")
                print(f"Median R²: {final_background_df['r_squared_t53'].median():.4f}")
                print(f"High quality fits (R² > 0.7): {(final_background_df['r_squared_t53'] > 0.7).sum()}")
                print("Note: Higher R² indicates better match to TDE t^(-5/3) decay signature")
            
            if 'det_error' in final_background_df.columns:
                 print("\n--- ⚠️ Detection Error Summary ---")
                 error_summary = final_background_df[final_background_df['det_error'].notna()]['det_error'].value_counts()
                 if not error_summary.empty:
                     print(f"Number of objects with detection errors: {len(final_background_df[final_background_df['det_error'].notna()])}")
                     print(error_summary)
                 else:
                     print("No detection errors reported for the processed sample.")

            # Save CSV in the MLModel folder (same directory as this script)
            output_path = os.path.join(os.path.dirname(__file__), 'background_training_data.csv')
            final_background_df.to_csv(output_path, index=False)
            print(f"\n✅ Training data saved to: {output_path}")