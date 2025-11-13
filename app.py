from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import matplotlib.pyplot as plt
import io
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

#Load the CSV file
df = pd.read_csv("ztf_objects_summary.csv")

# Initialize Alerce client (reuse for all requests)
alerce_client = Alerce()

# Create mapping from OID to r_squared from CSV (for plot generation)
r_squared_map = {}
if 'r_squared' in df.columns:
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
def get_csv():
    # Select the columns to show
    cols_to_show = ["oid", "peak_luminosity", "r_squared", "total_tde_score"]
    subset = df[cols_to_show]

    # Convert the subset to a dictionary and return it
    return subset.to_dict(orient="records")

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