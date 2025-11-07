import pandas as pd
# Removed 'import io' as we are reading directly from a file path

# Define constants for scoring
WEIGHTS = {
    # 5% was added from num_detections, increasing the weight of the geometric filter
    'offset_score': 0.55, # angular_normalized_offset (HIGH PRIORITY)
    'r_squared_score': 0.30, # r_squared
    'duration_score': 0.10, # duration_days
    'host_size_score': 0.05, # host_size_arcsec
}
PEAK_LUMINOSITY_THRESHOLD = 1.0e-6 # Based on your suggestion
BONUS_SCORE = 0.10

# --- 1. DATA INPUT ---
def load_and_clean_data(file_path):
    """Loads CSV data from a file, replaces 'N/A' and empty strings with NaN, and converts types."""
    try:
        # Read the file directly using the path
        df = pd.read_csv(file_path, skipinitialspace=True, na_values=['N/A', 'N/A (No catsHTM match)', ''])
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        # Re-raise the exception to stop execution gracefully
        raise
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        raise

    # Convert necessary columns to numeric, coercing errors to NaN
    numeric_cols = ['num_detections', 'redshift', 'peak_luminosity', 'r_squared',
                    'duration_days', 'host_size_arcsec', 'angular_normalized_offset']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

# --- 2. SCORING FUNCTIONS (Normalized to 0.0, 0.5, 1.0) ---

def score_normalized_offset(offset):
    """Scores TDE likelihood based on angular_normalized_offset (55% weight)."""
    if pd.isna(offset) or offset <= 0.0:
        return 0.0
    elif offset <= 0.2:
        return 1.0 # High Score: Closest to host center
    elif offset <= 0.5:
        return 0.5 # Moderate Score
    else:
        return 0.0 # Low Score: Too far from center

def score_r_squared(r_squared):
    """Scores TDE likelihood based on r_squared (30% weight)."""
    # Use absolute value, assuming 1.0 is the best fit quality regardless of sign
    r_sq = abs(r_squared) if not pd.isna(r_squared) else 0.0
    if r_sq >= 0.8:
        return 1.0 # High Score: Very good fit
    elif r_sq >= 0.4:
        return 0.5 # Moderate Score
    else:
        return 0.0 # Low Score: Poor fit

def score_duration(duration):
    """Scores TDE likelihood based on duration_days (10% weight)."""
    if pd.isna(duration):
        return 0.0
    elif 200 <= duration <= 1000:
        return 1.0 # High Score: Typical TDE timescale
    elif 100 <= duration < 200 or 1000 < duration <= 2000:
        return 0.5 # Moderate Score: Acceptable range
    else:
        return 0.0 # Low Score: Too short (SN-like) or too long (AGN-like)

def score_host_size(host_size):
    """Scores TDE likelihood based on host_size_arcsec (5% weight)."""
    if pd.isna(host_size):
        return 0.0
    elif host_size > 1.0:
        return 1.0 # High Score: Well-resolved host
    elif host_size > 0.5:
        return 0.5 # Moderate Score
    else:
        return 0.0 # Low Score: Small or unresolved host

# --- 3. MAIN SCORING LOGIC ---

def calculate_tde_score(df):
    """Calculates the primary, bonus, and final total TDE scores."""

    print("--- Applying Primary Scores and Weights ---")

    # Apply primary scoring functions
    df['offset_score'] = df['angular_normalized_offset'].apply(score_normalized_offset)
    df['r_squared_score'] = df['r_squared'].apply(score_r_squared)
    df['duration_score'] = df['duration_days'].apply(score_duration)
    df['host_size_score'] = df['host_size_arcsec'].apply(score_host_size)

    # Calculate Weighted Primary Score (Max 1.0)
    df['Primary_Score'] = (
        df['offset_score'] * WEIGHTS['offset_score'] +
        df['r_squared_score'] * WEIGHTS['r_squared_score'] +
        df['duration_score'] * WEIGHTS['duration_score'] +
        df['host_size_score'] * WEIGHTS['host_size_score']
    )

    print("--- Applying Bonus Score for Redshift/Luminosity ---")

    # Calculate Bonus Score (Max 0.10)
    # Bonus is granted if redshift is known AND peak luminosity exceeds the threshold
    df['Bonus'] = (
        (~df['redshift'].isnull()) &
        (~df['peak_luminosity'].isnull()) &
        (df['peak_luminosity'] >= PEAK_LUMINOSITY_THRESHOLD)
    ).astype(int) * BONUS_SCORE

    print(f"Candidates with Redshift and Peak Luminosity >= {PEAK_LUMINOSITY_THRESHOLD:.2e} receive a +{BONUS_SCORE:.2f} bonus.")

    # Calculate Total Score (capped at 1.0)
    df['Total_TDE_Score'] = df['Primary_Score'] + df['Bonus']
    df['Total_TDE_Score'] = df['Total_TDE_Score'].clip(upper=1.0)

    return df

# --- 4. EXECUTION FUNCTION ---

def calculate_and_save_tde_scores(file_path):
    """
    Main execution function that loads data, calculates TDE scores, and saves results.
    Can be called from main.py or run standalone.
    """
    try:
        df = load_and_clean_data(file_path)
        df_scored = calculate_tde_score(df)

        # Sort the data by the new score column in descending order
        df_sorted = df_scored.sort_values(by='Total_TDE_Score', ascending=False)
        
        # --- WRITE BACK TO CSV: Overwrite the original file with the scored and sorted data ---
        df_sorted.to_csv(file_path, index=False)
        print(f"\nSUCCESS: The file '{file_path}' has been updated.")
        print("New columns ('Primary_Score', 'Bonus', 'Total_TDE_Score', and intermediate scores) have been added, and the data is sorted.")
        # --- END WRITE BACK ---

        print("\n" + "="*50)
        print("TOP 10 TDE Candidates in the Updated File (Sorted by Total_TDE_Score)")
        print("="*50)

        # Select and print relevant columns for the final output
        final_columns = [
            'oid', 'Total_TDE_Score', 'Primary_Score', 'Bonus',
            'angular_normalized_offset', 'r_squared', 'duration_days', 'redshift',
            'peak_luminosity', 'num_detections'
        ]

        # Print the head (top 10) of the sorted data for immediate visualization
        print(df_sorted[final_columns].head(10).to_string(index=False))
        
        return df_sorted

    except FileNotFoundError:
        # Specific error message for the user if the file doesn't exist
        print(f"\nFATAL ERROR: Please ensure the file '{file_path}' exists in the working directory.")
        return None
    except Exception as e:
        # Generic error message for other exceptions
        print(f"\nAn unexpected error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 5. STANDALONE EXECUTION ---

if __name__ == "__main__":
    # Define the file path
    FILE_PATH = "ztf_objects_summary.csv"
    calculate_and_save_tde_scores(FILE_PATH)