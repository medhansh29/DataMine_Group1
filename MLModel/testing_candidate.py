import pandas as pd
import numpy as np
from alerce.core import Alerce
from tqdm import tqdm 
import time 
import joblib 
import os
import sys

# Import FeatureWeightScaler from shared module to enable loading pickled models
# The model was trained with this custom transformer, so we need it available for unpickling
try:
    from feature_transforms import FeatureWeightScaler
except ImportError:
    # Fallback: if the module doesn't exist, try importing from anamoly_detection
    # This handles the case where the model was trained before the shared module was created
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from anamoly_detection import FeatureWeightScaler
    except ImportError:
        # Last resort: define it here (must match the exact definition)
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class FeatureWeightScaler(BaseEstimator, TransformerMixin):
            """Custom transformer that applies feature-specific weights before scaling."""
            def __init__(self, feature_weights=None, base_weight=0.67):
                self.feature_weights = feature_weights if feature_weights is not None else {}
                self.base_weight = base_weight
                self.feature_names_ = None
            
            def fit(self, X, y=None):
                if isinstance(X, pd.DataFrame):
                    self.feature_names_ = X.columns.tolist()
                else:
                    self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
                return self
            
            def transform(self, X):
                if isinstance(X, pd.DataFrame):
                    X_weighted = X.copy()
                    for feature_name in X.columns:
                        if feature_name in self.feature_names_:
                            weight = self.feature_weights.get(feature_name, self.base_weight)
                            X_weighted[feature_name] = X[feature_name] * weight
                    return X_weighted
                else:
                    X_weighted = X.copy()
                    for i, feature_name in enumerate(self.feature_names_):
                        if i < X.shape[1]:
                            weight = self.feature_weights.get(feature_name, self.base_weight)
                            X_weighted[:, i] = X[:, i] * weight
                    return X_weighted
            
            def get_feature_names_out(self, input_features=None):
                if input_features is not None:
                    return np.array(input_features)
                if self.feature_names_ is not None:
                    return np.array(self.feature_names_)
                return None

# Make FeatureWeightScaler available in __main__ for pickle compatibility
# This is needed because the model was pickled when running anamoly_detection.py as __main__
import __main__
if not hasattr(__main__, 'FeatureWeightScaler'):
    __main__.FeatureWeightScaler = FeatureWeightScaler

# --- 1. Configuration ---

# Initialize the ALeRCE Client
alerce = Alerce()

# Option 1: Manually specify OIDs to test
# TDE_CANDIDATE_OIDS = [
#     "ZTF24aatxshz"
# ]

# Option 2: Load OIDs from CSV file with filtering (uncomment to use)
# Set CSV_PATH to your ztf_objects_summary.csv file path
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ztf_objects_summary.csv')

def load_candidates_from_csv(csv_path: str, 
                             min_tde_score: float = 0.7,
                             max_angular_normalized_offset: float = 0.1,
                             limit: int = None) -> list:
    """
    Load TDE candidate OIDs from the main CSV file based on filtering criteria.
    
    Args:
        csv_path: Path to ztf_objects_summary.csv
        min_tde_score: Minimum Total_TDE_Score to include (default: 0.7)
        max_angular_normalized_offset: Maximum angular_normalized_offset (default: 0.1)
                                       This is the PRIMARY filter for TDE candidates.
                                       Normalized offset < 0.1 indicates nuclear events.
        limit: Maximum number of candidates to return (optional)
    
    Returns:
        List of OID strings
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"-> Loading candidates from {csv_path}...")
        print(f"   Total objects in CSV: {len(df)}")
        
        # Apply filters
        filtered = df.copy()
        
        # PRIMARY FILTER: Normalized offset (most important for TDE identification)
        if 'angular_normalized_offset' in df.columns:
            # Filter out NaN normalized offsets
            filtered = filtered[filtered['angular_normalized_offset'].notna()]
            filtered = filtered[filtered['angular_normalized_offset'] <= max_angular_normalized_offset]
            print(f"   After normalized offset filter (<= {max_angular_normalized_offset}): {len(filtered)}")
            print(f"   Note: Normalized offset < 0.1 is the key criterion for nuclear TDE events")
        else:
            print(f"   Warning: 'angular_normalized_offset' column not found in CSV!")
            print(f"   Cannot apply primary TDE filter. Proceeding with other filters only.")
        
        # SECONDARY FILTER: TDE score
        if 'Total_TDE_Score' in df.columns:
            # Filter out NaN scores
            filtered = filtered[filtered['Total_TDE_Score'].notna()]
            filtered = filtered[filtered['Total_TDE_Score'] >= min_tde_score]
            print(f"   After TDE score filter (>= {min_tde_score}): {len(filtered)}")
        
        # Sort by normalized offset (lowest first, then by TDE score)
        if 'angular_normalized_offset' in filtered.columns:
            filtered = filtered.sort_values(
                ['angular_normalized_offset', 'Total_TDE_Score'], 
                ascending=[True, False]
            )
        elif 'Total_TDE_Score' in filtered.columns:
            filtered = filtered.sort_values('Total_TDE_Score', ascending=False)
        
        # Get unique OIDs (preserving order)
        seen = set()
        oids = []
        for oid in filtered['oid']:
            if oid not in seen:
                seen.add(oid)
                oids.append(oid)
        
        if limit is not None:
            oids = oids[:limit]
        
        print(f"   Selected {len(oids)} unique candidates for testing")
        return oids
        
    except FileNotFoundError:
        print(f"   Warning: CSV file not found at {csv_path}. Using manual OID list.")
        return []
    except Exception as e:
        print(f"   Error loading candidates from CSV: {e}")
        return []

# Load candidates from CSV (modify filters as needed)
TDE_CANDIDATE_OIDS = load_candidates_from_csv(
    CSV_PATH,
    min_tde_score=0.7,  # Minimum TDE score
    max_angular_normalized_offset=0.1,  # PRIMARY FILTER: Maximum normalized offset (must be < 0.1 for nuclear TDEs)
    limit=10  # Limit to top 10 candidates (sorted by normalized offset, then TDE score)
)

# Fallback to manual list if CSV loading fails
if not TDE_CANDIDATE_OIDS:
    TDE_CANDIDATE_OIDS = [
        "ZTF24aatxshz"
    ]
    print("-> Using manual OID list as fallback")

# Paths relative to the script directory (MLModel folder)
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_FILENAME = os.path.join(SCRIPT_DIR, 'tde_anomaly_detector_pipeline.pkl')
BACKGROUND_FEATURES_PATH = os.path.join(SCRIPT_DIR, 'background_training_data.csv')

# --- 2. Feature Extraction Utilities (Reuse from Training) ---

def retrieve_and_pivot_features(oids: list) -> pd.DataFrame:
    """
    Retrieves features, handles duplicates, pivots, and concatenates.
    This function MUST match the behavior of training_data.py to ensure feature compatibility.
    """
    print(f"-> Retrieving and pivoting features for {len(oids)} candidates...")
    
    all_features = []
    
    for oid in tqdm(oids, desc="Fetching ALeRCE features"):
        try:
            df_feat = alerce.query_features(oid=oid, format='pandas')
            
            if not df_feat.empty:
                # Drop duplicates (same as training script)
                df_feat_clean = df_feat.drop_duplicates(subset=['name'], keep='first')
                
                # Create a Series with feature names as index and values
                # IMPORTANT: This must match exactly how training_data.py processes features
                feature_series = df_feat_clean.set_index('name')['value'].copy()
                
                # Create a DataFrame row with oid as index
                # This ensures consistent column alignment when concatenating
                feature_dict = feature_series.to_dict()
                feature_dict['oid'] = oid
                all_features.append(feature_dict)

        except Exception as e:
            print(f"   [Feature Retrieval Error for {oid}]: {e}")
            time.sleep(0.1) 

    if not all_features:
        return pd.DataFrame()

    # Create DataFrame from list of dicts (same structure as training script)
    features_df = pd.DataFrame(all_features)
    features_df = features_df.set_index('oid')
    
    return features_df

def calculate_custom_tde_features(oid: str) -> dict:
    """
    Calculates TDE-specific features including:
    - max_mag_r: Peak magnitude in r-band
    - peak_mjd: Time of peak magnitude
    - rise_duration: Duration from first detection to peak
    
    Note: R² for t^(-5/3) decay is loaded from ztf_objects_summary.csv (r_squared column),
    not recalculated here to avoid redundant computation.
    """
    
    try:
        df_det = alerce.query_detections(oid=oid, format='pandas')
    except Exception:
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan
        }

    if df_det.empty:
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan
        }

    # Robust magnitude selection logic (same as training)
    mag_col = 'magpsf_corr' if 'magpsf_corr' in df_det.columns else 'magpsf'

    r_band_dets = df_det[df_det['fid'] == 2].copy()

    if r_band_dets.empty:
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan
        }

    # Sort by MJD
    r_band_dets = r_band_dets.sort_values('mjd').reset_index(drop=True)

    peak_mag = r_band_dets[mag_col].min()
    
    if not np.isfinite(peak_mag):
        return {
            'max_mag_r': np.nan, 
            'peak_mjd': np.nan, 
            'rise_duration': np.nan
        }
    
    # Find peak row
    peak_row = r_band_dets[r_band_dets[mag_col] == peak_mag].iloc[0]
    peak_mjd = peak_row['mjd']
    first_mjd = r_band_dets['mjd'].min()
    rise_duration = peak_mjd - first_mjd
    
    return {
        'max_mag_r': peak_mag, 
        'peak_mjd': peak_mjd, 
        'rise_duration': rise_duration
    }


def add_custom_features_to_df(features_df: pd.DataFrame) -> pd.DataFrame:
    """Iterates through the feature DataFrame and adds custom TDE features."""
    
    oids = features_df.index.tolist()
    custom_features_list = []
    
    print(f"\n-> Calculating custom TDE features for {len(oids)} candidates...")
    for oid in tqdm(oids, desc="Calculating custom features"):
        features = calculate_custom_tde_features(oid)
        features['oid'] = oid
        custom_features_list.append(features)
        time.sleep(0.1) 
        
    custom_df = pd.DataFrame(custom_features_list).set_index('oid')
    merged_df = features_df.join(custom_df, how='inner')
    
    return merged_df

# --- 3. Testing and Scoring Workflow ---

def score_candidates(candidate_oids: list, model_path: str, background_path: str, csv_path: str = None):
    """
    Score TDE candidates using the trained anomaly detection model.
    
    Args:
        candidate_oids: List of OID strings to score
        model_path: Path to the trained model pipeline
        background_path: Path to the background training data CSV
        csv_path: Path to ztf_objects_summary.csv to update with ML scores (optional)
    
    Returns:
        DataFrame with scoring results
    """
    # A. Load R² values from CSV (if available) to merge with features
    # Use the provided csv_path or fall back to CSV_PATH
    r_squared_csv_path = csv_path if csv_path else CSV_PATH
    r_squared_from_csv = {}
    
    if r_squared_csv_path and os.path.exists(r_squared_csv_path):
        try:
            summary_df = pd.read_csv(r_squared_csv_path)
            if 'oid' in summary_df.columns and 'r_squared' in summary_df.columns:
                # Get the first (or mean) r_squared value for each OID
                for oid in candidate_oids:
                    oid_rows = summary_df[summary_df['oid'] == oid]
                    if not oid_rows.empty:
                        r_squared_val = oid_rows['r_squared'].iloc[0]  # Use first value
                        if pd.notna(r_squared_val):
                            r_squared_from_csv[oid] = r_squared_val
                if r_squared_from_csv:
                    print(f"-> Loaded R² values from CSV for {len(r_squared_from_csv)}/{len(candidate_oids)} candidates")
                else:
                    print(f"   Note: No R² values found in CSV for candidates (r_squared column may be empty)")
        except Exception as e:
            print(f"   Warning: Could not load R² from CSV: {e}")
    else:
        print(f"   Note: CSV not found or path not provided, R² values will be NaN")
    
    # B. Retrieve and process candidate features
    raw_candidate_df = retrieve_and_pivot_features(candidate_oids)
    processed_candidate_df = add_custom_features_to_df(raw_candidate_df)
    
    # C. Add R² from CSV (r_squared column maps to r_squared_t53 for model compatibility)
    if r_squared_from_csv:
        r_squared_series = pd.Series(r_squared_from_csv, name='r_squared_t53')
        # Merge R² values into the processed dataframe
        processed_candidate_df = processed_candidate_df.join(r_squared_series, how='left')
        print(f"   Merged R² values from CSV into feature matrix")
    
    if processed_candidate_df.empty:
        print("\n❌ Failed to retrieve features for any candidates. Aborting scoring.")
        return None

    # D. Load Model First to Get Expected Features
    print(f"\n-> Loading anomaly detection pipeline from {model_path}...")
    try:
        anomaly_pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at {model_path}. Please check your path.")
        return

    # E. Get Expected Features from Model (Most Reliable Method)
    print("\n-> Aligning candidate features with training features...")
    
    # Try to get feature names from the scaler (if available in sklearn version)
    scaler = anomaly_pipeline.named_steps['scaler']
    expected_features = None
    
    if hasattr(scaler, 'feature_names_in_'):
        # Newer sklearn versions store feature names
        expected_features = scaler.feature_names_in_.tolist()
        print(f"   Found feature names in model: {len(expected_features)} features")
    else:
        # Fallback: Load from training CSV and apply same preprocessing
        print("   Model doesn't store feature names, loading from training CSV...")
        try:
            df_train_template = pd.read_csv(background_path)
            
            # Drop the label column if it exists
            if 'is_tde_anomaly' in df_train_template.columns:
                df_train_template = df_train_template.drop(columns=['is_tde_anomaly'])
            
            # Apply the same preprocessing as training script:
            # 1. Drop columns with too many NaNs (90% threshold, same as training)
            nan_threshold = len(df_train_template) * 0.90
            df_train_template = df_train_template.dropna(axis=1, thresh=nan_threshold)
            
            # 2. Select only numeric columns (same as training)
            numeric_cols = df_train_template.select_dtypes(include=np.number).columns.tolist()
            df_train_template = df_train_template[numeric_cols]
            
            expected_features = df_train_template.columns.tolist()
            print(f"   Loaded {len(expected_features)} features from training CSV")
            
        except FileNotFoundError:
            print(f"❌ Error: Training template file not found at {background_path}.")
            print("   Cannot determine expected features. Please retrain the model or check the CSV path.")
            return
        except Exception as e:
            print(f"❌ Error loading training template: {e}")
            return
    
    if expected_features is None:
        print("❌ Error: Could not determine expected features from model or CSV.")
        return
    
    print(f"   Model expects {len(expected_features)} features")
    print(f"   Candidate data has {len(processed_candidate_df.columns)} features")
    
    # F. Align candidate features to match exactly what the model expects
    # Select only numeric columns from candidates first
    processed_candidate_df = processed_candidate_df.select_dtypes(include=np.number)
    
    # Create a DataFrame with the exact features the model expects
    # Missing features will be filled with 0, extra features will be dropped
    X_candidates = pd.DataFrame(index=processed_candidate_df.index, columns=expected_features)
    
    # Fill in available features
    missing_features = []
    matched_features = []
    for col in expected_features:
        if col in processed_candidate_df.columns:
            X_candidates[col] = processed_candidate_df[col].values
            matched_features.append(col)
        else:
            # Fill missing features with 0
            X_candidates[col] = 0
            missing_features.append(col)
    
    # Check if we have a significant mismatch (model trained on mock data vs real data)
    match_ratio = len(matched_features) / len(expected_features) if expected_features else 0
    
    # Report feature alignment
    print(f"   Matched {len(matched_features)}/{len(expected_features)} features ({match_ratio*100:.1f}%)")
    
    if missing_features:
        print(f"   Warning: {len(missing_features)} features not found in candidate data, filling with 0")
        if len(missing_features) <= 10:
            print(f"   Missing features: {missing_features}")
        else:
            print(f"   Missing features (first 10): {missing_features[:10]}...")
        
        # Check if this looks like a mock vs real data mismatch
        mock_features = ['Amplitude', 'Gskew', 'Std', 'feature_0', 'feature_1']
        if any(f in missing_features for f in mock_features):
            print("\n   ⚠️  WARNING: Model appears to be trained on mock data!")
            print("   The model expects mock features (Amplitude, Gskew, Std, feature_0, etc.)")
            print("   but candidate data has real ALeRCE features (g-r_mean, Coordinate_x, etc.).")
            print("   SOLUTION: Retrain the model using training_data.py (not anamoly_detection.py)")
            print("   to generate real training data, then train with anamoly_detection.py on that data.")
            print("   Continuing with zeros for missing features, but results may be unreliable.\n")
    
    # Report extra features that will be dropped
    extra_features = [col for col in processed_candidate_df.columns if col not in expected_features]
    if extra_features:
        print(f"   Info: {len(extra_features)} extra features in candidate data will be dropped")
        if len(extra_features) <= 5:
            print(f"   Extra features (sample): {extra_features[:5]}")
    
    # Fill any remaining NaN values with 0 (shouldn't happen, but safety check)
    X_candidates = X_candidates.fillna(0)
    
    # Ensure feature order matches exactly (critical for sklearn)
    X_candidates = X_candidates[expected_features]
    
    # Convert to numpy array if sklearn version requires it (some versions are strict about DataFrame vs array)
    # But first, ensure column order is exactly as expected
    print(f"   Final aligned feature matrix: {X_candidates.shape}")
    print(f"   Feature order matches model: {list(X_candidates.columns) == expected_features}")
    
    # G. Score Candidates
    print("\n-> Scoring candidates...")
    
    # Calculate the anomaly score: lower score means higher anomaly (TDE-like)
    # Convert to numpy array to avoid feature name mismatches in some sklearn versions
    X_candidates_array = X_candidates.values
    scores = anomaly_pipeline.decision_function(X_candidates_array)
    
    # Classify based on the model's internal contamination threshold (1 for inlier, -1 for outlier)
    predictions = anomaly_pipeline.predict(X_candidates_array)
    
    # H. Final Results Compilation
    results_df = X_candidates.copy()
    results_df['anomaly_score'] = scores
    results_df['prediction'] = predictions
    results_df['TDE_Priority'] = results_df['prediction'].apply(lambda x: 'HIGH' if x == -1 else 'LOW')
    
    # Rank by the lowest score (most anomalous)
    results_df = results_df.sort_values(by='anomaly_score', ascending=True)
    
    print("\n--- ✅ Final Anomaly Detection Results ---")
    print(results_df[['anomaly_score', 'TDE_Priority']].head(len(candidate_oids)))
    print("\nCandidates ranked HIGH PRIORITY should be prioritized for follow-up.")
    
    # I. Update ztf_objects_summary.csv with ML scores
    update_path = csv_path if csv_path else CSV_PATH
    if update_path and os.path.exists(update_path):
        print(f"\n-> Updating {update_path} with ML scores...")
        try:
            # Load the summary CSV
            summary_df = pd.read_csv(update_path)
            print(f"   Loaded {len(summary_df)} rows from summary CSV")
            
            # Create a mapping from OID to anomaly_score (ML_score)
            # Use the index (OID) from results_df
            ml_scores_dict = dict(zip(results_df.index, results_df['anomaly_score']))
            
            # Initialize ML_score column if it doesn't exist
            if 'ML_score' not in summary_df.columns:
                summary_df['ML_score'] = np.nan
                print("   Created new 'ML_score' column")
            
            # Update ML_score for matching OIDs
            updated_count = 0
            for oid, ml_score in ml_scores_dict.items():
                # Find all rows with this OID and update them
                mask = summary_df['oid'] == oid
                if mask.any():
                    summary_df.loc[mask, 'ML_score'] = ml_score
                    updated_count += mask.sum()
                    print(f"   Updated OID {oid}: ML_score = {ml_score:.6f} ({mask.sum()} rows)")
                else:
                    print(f"   Warning: OID {oid} not found in summary CSV, skipping")
            
            # Save the updated CSV
            summary_df.to_csv(update_path, index=False)
            print(f"\n✅ Successfully updated {updated_count} rows in {update_path}")
            print(f"   ML_score column now contains anomaly detection scores")
            print(f"   Lower scores indicate higher anomaly (more TDE-like)")
            
        except Exception as e:
            print(f"   Error updating summary CSV: {e}")
            print(f"   Results are still available in memory, but CSV was not updated.")
            import traceback
            traceback.print_exc()
    else:
        if update_path:
            print(f"\n   Warning: Summary CSV not found at {update_path}")
            print(f"   Skipping ML score update. Results are still available in memory.")
        else:
            print(f"\n   No CSV path provided, skipping ML score update.")
    
    return results_df

# --- Main Execution ---
if __name__ == '__main__':
    results = score_candidates(TDE_CANDIDATE_OIDS, MODEL_FILENAME, BACKGROUND_FEATURES_PATH, CSV_PATH)
    if results is not None:
        print(f"\n✅ Scoring complete! Processed {len(results)} candidates.")
    else:
        print("\n❌ Scoring failed. Please check the error messages above.")