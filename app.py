from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
from alerce.core import Alerce
from plot_light_curve import plot_light_curve

#Initialize the FastAPI app
app = FastAPI()

# Enable CORS for frontend access (Vercel, localhost, etc.)
# For production, replace with specific origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local React dev server
        "http://localhost:5173",  # Vite dev server
        "https://v0-figma-design-revamp.vercel.app",  # Your Vercel frontend
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel deployments (backup)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the CSV file with proper path resolution
CSV_FILE = "ztf_objects_summary.csv"
# Try to find CSV file - check relative path first, then absolute
csv_path = CSV_FILE
if not os.path.exists(csv_path):
    # Try in current directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, CSV_FILE)

try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded CSV file: {csv_path} ({len(df)} rows)")
except FileNotFoundError:
    print(f"ERROR: CSV file not found at: {csv_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    # Create empty dataframe to prevent startup crash
    df = pd.DataFrame()
    raise Exception(f"CSV file 'ztf_objects_summary.csv' not found. Please ensure it exists in the repository.")
except Exception as e:
    print(f"ERROR loading CSV file: {e}")
    import traceback
    traceback.print_exc()
    df = pd.DataFrame()
    raise

# Initialize Alerce client (reuse for all requests)
alerce_client = Alerce()

# Create mapping from OID to r_squared from CSV (for plot generation)
r_squared_map = {}
if not df.empty and 'r_squared' in df.columns and 'oid' in df.columns:
    for oid in df['oid'].unique():
        oid_data = df[df['oid'] == oid]
        r_sq_values = oid_data['r_squared'].dropna()
        if len(r_sq_values) > 0:
            first_val = r_sq_values.iloc[0]
            if isinstance(first_val, str) and first_val.strip().upper() == 'N/A':
                r_squared_map[oid] = None
            else:
                try:
                    r_squared_map[oid] = float(first_val)
                except (ValueError, TypeError):
                    r_squared_map[oid] = None
        else:
            r_squared_map[oid] = None

@app.get("/csv")
@app.get("/data")  # Alias for frontend compatibility
def get_csv(oid: str = Query(None, description="Optional: Get a specific row by OID")):
    """
    Get CSV data as JSON array with plot image URLs included.
    Returns: List of objects with oid, peak_luminosity, r_squared, Total_TDE_Score, plot_url
    If 'oid' parameter is provided, returns only that specific row.
    """
    import sys
    import traceback
    
    # IMPORTANT: global declaration must be at the top, before any use of df
    global df, csv_path
    
    try:
        print(f"[DEBUG] /csv endpoint called", flush=True)
        sys.stdout.flush()
        
        # Check if dataframe is loaded and not empty - with detailed logging
        if 'df' not in globals():
            print(f"[ERROR] DataFrame variable 'df' does not exist in globals", flush=True)
            raise HTTPException(
                status_code=500,
                detail="CSV file not loaded. Please check server logs for errors."
            )
        
        if df.empty:
            print(f"[ERROR] DataFrame is empty! This should not happen after startup.", flush=True)
            print(f"[ERROR] Attempting to reload CSV file...", flush=True)
            # Try to reload the CSV as a safety measure
            try:
                # Reload CSV from the path
                reload_path = CSV_FILE if os.path.exists(CSV_FILE) else os.path.join(os.path.dirname(os.path.abspath(__file__)), CSV_FILE)
                df = pd.read_csv(reload_path)
                print(f"[INFO] Successfully reloaded CSV file: {reload_path} ({len(df)} rows)", flush=True)
            except Exception as reload_error:
                print(f"[ERROR] Failed to reload CSV: {reload_error}", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail="CSV file is empty and could not be reloaded. Please check server logs."
                )
        
        # Verify df still has data after checks
        if df.empty:
            raise HTTPException(
                status_code=500,
                detail="DataFrame is empty after reload attempt. Please check CSV file."
            )
        
        print(f"[DEBUG] DataFrame has {len(df)} rows (safely loaded)", flush=True)
        sys.stdout.flush()
        
        # Get the base URL for constructing plot URLs - use hardcoded URL
        base_url = "https://datamine-group1.onrender.com"
        print(f"[DEBUG] Using base_url: {base_url}", flush=True)
        
        sys.stdout.flush()
        
        # Select the columns to show
        # Note: Column names match CSV exactly (case-sensitive)
        cols_to_show = ["oid", "peak_luminosity", "r_squared", "Total_TDE_Score"]
        print(f"[DEBUG] Looking for columns: {cols_to_show}", flush=True)
        print(f"[DEBUG] Available columns: {list(df.columns)}", flush=True)
        sys.stdout.flush()
        
        # Check if all columns exist
        missing_cols = [col for col in cols_to_show if col not in df.columns]
        if missing_cols:
            error_msg = f"Missing columns in CSV: {missing_cols}. Available columns: {list(df.columns)}"
            print(f"[ERROR] {error_msg}", flush=True)
            raise HTTPException(status_code=500, detail=error_msg)
        
        print(f"[DEBUG] All columns found, selecting subset...", flush=True)
        sys.stdout.flush()
        
        # Select subset and handle NaN values
        # IMPORTANT: Always use .copy() to avoid modifying the original df
        try:
            # Verify original df still has data before filtering
            if df.empty:
                print(f"[ERROR] Original df became empty during processing!", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail="DataFrame became empty during processing. This should not happen."
                )
            
            # If OID is specified, filter to just that row
            if oid:
                print(f"[DEBUG] Filtering for OID: {oid}", flush=True)
                # Use .copy() to ensure we don't modify the original df
                subset = df[df['oid'] == oid][cols_to_show].copy()
                if subset.empty:
                    raise HTTPException(status_code=404, detail=f"OID '{oid}' not found in CSV")
                print(f"[DEBUG] Found 1 row for OID: {oid}", flush=True)
            else:
                # Use .copy() to ensure we don't modify the original df
                subset = df[cols_to_show].copy()
            print(f"[DEBUG] Subset created with {len(subset)} rows (original df still has {len(df)} rows)", flush=True)
            sys.stdout.flush()
            
            # Double-check original df wasn't modified
            if df.empty:
                print(f"[ERROR] Original df was modified and became empty!", flush=True)
                raise HTTPException(
                    status_code=500,
                    detail="DataFrame was modified during processing. This is a bug."
                )
        except HTTPException:
            raise
        except Exception as subset_error:
            print(f"[ERROR] Error creating subset: {subset_error}", flush=True)
            traceback.print_exc()
            raise
        
        # Convert NaN values to None for JSON serialization
        print(f"[DEBUG] Converting NaN to None...", flush=True)
        sys.stdout.flush()
        try:
            for col in subset.columns:
                subset[col] = subset[col].where(pd.notnull(subset[col]), None)
        except Exception as nan_error:
            print(f"[ERROR] Error converting NaN: {nan_error}", flush=True)
            traceback.print_exc()
            raise
        
        print(f"[DEBUG] Converting to dict...", flush=True)
        sys.stdout.flush()
        
        # Convert the subset to a dictionary and return it
        try:
            result = subset.to_dict(orient="records")
            print(f"[DEBUG] Dict conversion complete, {len(result)} records", flush=True)
            sys.stdout.flush()
        except Exception as dict_error:
            print(f"[ERROR] Error converting to dict: {dict_error}", flush=True)
            traceback.print_exc()
            raise
        
        print(f"[DEBUG] Adding plot_url to each item...", flush=True)
        sys.stdout.flush()
        
        # Add plot_url for each OID and ensure OID is a string
        try:
            for i, item in enumerate(result):
                oid = str(item['oid']) if item['oid'] is not None else ""
                item['oid'] = oid
                # Create URL to the plot endpoint for this OID
                item['plot_url'] = f"{base_url}/plot/{oid}"
                if i < 3:  # Log first 3 for debugging
                    print(f"[DEBUG] Item {i}: oid={oid}", flush=True)
        except Exception as item_error:
            print(f"[ERROR] Error processing items: {item_error}", flush=True)
            traceback.print_exc()
            raise
        
        print(f"[DEBUG] Returning {len(result)} records", flush=True)
        sys.stdout.flush()
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"[ERROR] Error in /csv endpoint: {error_details}", flush=True)
        sys.stdout.flush()
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/")
def root():
    """Health check and API info endpoint"""
    return {
        "status": "ok",
        "message": "FastAPI backend is running",
        "endpoints": {
            "csv": "/csv or /data - Get all data as JSON array",
            "plot": "/plot/{oid} - Get light curve plot for an OID"
        },
        "total_records": len(df) if 'df' in globals() else 0
    }

@app.get("/plot/{oid}")
def get_light_curve_plot(oid: str):
    """
    Generate and return a light curve plot for a specific OID.
    
    Parameters:
    -----------
    oid : str
        Object ID (e.g., "ZTF17aaajhvh")
    
    Returns:
    --------
    PNG image of the light curve plot
    """
    # Check if OID exists in CSV
    if oid not in df['oid'].values:
        raise HTTPException(status_code=404, detail=f"OID '{oid}' not found in CSV")
    
    # Get r_squared from CSV if available
    r_squared_from_csv = r_squared_map.get(oid)
    
    import tempfile
    import os as os_module
    
    # Create a temporary file to save the plot
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Generate the plot and save to temp file
        success = plot_light_curve(
            object_id=oid,
            alerce_client=alerce_client,
            r_band_only=True,
            save_path=temp_path,
            show_plot=False,
            r_squared_from_csv=r_squared_from_csv
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to generate plot for OID '{oid}'")
        
        # Read the temp file into a BytesIO buffer
        img_buffer = io.BytesIO()
        with open(temp_path, 'rb') as f:
            img_buffer.write(f.read())
        
        # Clean up temp file
        os_module.unlink(temp_path)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        if os_module.path.exists(temp_path):
            os_module.unlink(temp_path)
        raise
    except Exception as e:
        # Clean up temp file on error
        if os_module.path.exists(temp_path):
            os_module.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")
    
    # Reset buffer position to beginning
    img_buffer.seek(0)
    
    # Return the image as a streaming response
    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={"Content-Disposition": f"inline; filename={oid}_light_curve.png"}
    )