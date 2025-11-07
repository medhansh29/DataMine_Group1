import pandas as pd
from alerce.core import Alerce
import os
import time

def _extract_time_columns(df: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Given an ALeRCE detections DataFrame, try to extract first and last observation
    times in Modified Julian Date (MJD). Handles several common timestamp column names.
    Returns a tuple (first_mjd, last_mjd); None values if not available.
    """
    # Common time columns used by brokers/APIs
    possible_time_cols = [
        "mjd",            # Modified Julian Date
        "jd",             # Julian Date
        "timestamp",      # epoch seconds or string
        "time",           # generic
        "date",
    ]

    for col in possible_time_cols:
        if col in df.columns and not df[col].isna().all():
            series = df[col].dropna()
            if series.empty:
                continue

            # If the column is JD, convert to MJD
            if col == "jd":
                jd_min = float(series.min())
                jd_max = float(series.max())
                return jd_min - 2400000.5, jd_max - 2400000.5

            # If the column looks like MJD already
            if col == "mjd":
                return float(series.min()), float(series.max())

            # If it's epoch seconds (numeric and ~1e9 order), convert to MJD
            if pd.api.types.is_numeric_dtype(series):
                epoch_min = float(series.min())
                epoch_max = float(series.max())
                # Convert UNIX epoch seconds → days → JD → MJD
                days_min = epoch_min / 86400.0
                days_max = epoch_max / 86400.0
                jd_epoch_origin = 2440587.5  # JD for 1970-01-01
                mjd_min = (jd_epoch_origin + days_min) - 2400000.5
                mjd_max = (jd_epoch_origin + days_max) - 2400000.5
                return mjd_min, mjd_max

            # If it's a string datetime, parse with pandas and convert to MJD
            try:
                parsed = pd.to_datetime(series, utc=True, errors="coerce")
                parsed = parsed.dropna()
                if parsed.empty:
                    continue
                # Convert UTC timestamps → JD → MJD
                # pandas Timestamp to ordinal days: use to_julian_date()
                jd_min = float(parsed.min().to_julian_date())
                jd_max = float(parsed.max().to_julian_date())
                return jd_min - 2400000.5, jd_max - 2400000.5
            except Exception:
                continue

    return None, None

def _mjd_to_utc_iso(mjd_value: float | None) -> str | None:
    """
    Convert MJD to ISO UTC string using pandas. Returns None if input is None.
    """
    if mjd_value is None:
        return None
    jd = mjd_value + 2400000.5
    # Use pandas to convert JD days to UTC timestamp
    ts = pd.to_datetime(jd, unit="D", origin="julian", utc=True)
    return ts.isoformat()

def get_detection_date_range(ztf_object_id: str, client: Alerce) -> dict | None:
    """
    Fetch detections for an object and return first/last detection times.

    Returns dict with keys: oid, first_mjd, last_mjd, first_utc, last_utc, num_detections.
    Returns None if no detections.
    """
    try:
        df = client.query_detections(ztf_object_id, format="pandas")
        if df is None or df.empty:
            return None

        first_mjd, last_mjd = _extract_time_columns(df)
        if first_mjd is None or last_mjd is None:
            return {
                "oid": ztf_object_id,
                "first_mjd": None,
                "last_mjd": None,
                "first_utc": None,
                "last_utc": None,
                "num_detections": int(len(df))
            }

        return {
            "oid": ztf_object_id,
            "first_mjd": float(first_mjd),
            "last_mjd": float(last_mjd),
            "first_utc": _mjd_to_utc_iso(first_mjd),
            "last_utc": _mjd_to_utc_iso(last_mjd),
            "num_detections": int(len(df))
        }
    except Exception as e:
        print(f"[ERROR] Failed to fetch detections for {ztf_object_id}: {e}")
        return None

def update_date_ranges_from_csv(csv_path: str = "ztf_objects_summary.csv", delay_seconds: float = 0.25) -> None:
    """
    For each OID in the CSV, fetch detections and write first/last detection dates.

    Adds columns if missing: first_mjd, last_mjd, first_utc, last_utc, duration_days.
    Writes results back to the same CSV.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file '{csv_path}' not found!")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {e}")
        return

    if 'oid' not in df.columns:
        print("[ERROR] CSV must contain an 'oid' column")
        return

    # Ensure columns exist
    for col in ["first_mjd", "last_mjd", "first_utc", "last_utc", "duration_days"]:
        if col not in df.columns:
            df[col] = None

    client = Alerce()

    # Count objects that need processing
    needs_processing = df[
        df['first_mjd'].isna() | 
        df['last_mjd'].isna() | 
        df['first_utc'].isna() | 
        df['last_utc'].isna()
    ]
    if len(needs_processing) == 0:
        print(f"All {len(df)} objects already have detection date data. Skipping processing.")
        return
    
    print(f"Processing {len(needs_processing)} objects (skipping {len(df) - len(needs_processing)} with existing data)...")

    for index, row in df.iterrows():
        oid = row['oid']
        # Skip if we already populated both dates
        if pd.notna(row.get('first_mjd')) and pd.notna(row.get('last_mjd')) and pd.notna(row.get('first_utc')) and pd.notna(row.get('last_utc')):
            continue

        print(f"Fetching detection dates for {oid}...")
        result = get_detection_date_range(oid, client)
        if result is None:
            print(f"  No detections found for {oid}.")
            continue

        df.at[index, 'first_mjd'] = result['first_mjd']
        df.at[index, 'last_mjd'] = result['last_mjd']
        df.at[index, 'first_utc'] = result['first_utc']
        df.at[index, 'last_utc'] = result['last_utc']

        if result['first_mjd'] is not None and result['last_mjd'] is not None:
            df.at[index, 'duration_days'] = float(result['last_mjd'] - result['first_mjd'])

        time.sleep(delay_seconds)  # Be kind to the API

    try:
        df.to_csv(csv_path, index=False)
        print(f"[SUCCESS] Updated '{csv_path}' with detection date ranges.")
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")

def fetch_detection_dates_menu_option(csv_path: str = "ztf_objects_summary.csv") -> None:
    """
    Wrapper function for the menu option to fetch detection dates.
    This is the function that should be called from main.py for option 3.
    
    Checks if CSV exists and handles the date range fetching process.
    """
    print("\n--- Fetching Detection Date Ranges ---")
    if not os.path.exists(csv_path):
        print("Error: ztf_objects_summary.csv file not found!")
        print("Please fetch some ZTF objects first (option 1).")
        return
    
    update_date_ranges_from_csv(csv_path)
    print("Detection date range fetch completed!")
