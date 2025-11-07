# DataMine_Group1: ZTF Object Data Fetcher with TDE Analysis

## Overview
This repository contains a comprehensive toolkit for fetching ZTF (Zwicky Transient Facility) object data, performing redshift analysis, computing angular offsets, analyzing light curves, and scoring Tidal Disruption Event (TDE) candidates. The system efficiently queries the ALeRCE API, uses DELIGHT for host galaxy identification, and provides a complete workflow for TDE candidate identification and ranking.

## File Description

### Core Files:

1. **`main.py`**: Main entry point with interactive menu system
   - Provides menu-driven interface for all operations
   - Handles CSV path resolution
   - Coordinates all workflow steps

2. **`datapoint_filter.py`**: Handles ZTF object fetching with efficient batching
   - Queries 500 objects at a time for efficiency
   - Filters by datapoint range
   - Checks for duplicate OIDs before adding
   - Continues fetching until required number of unique objects found

3. **`redshifts.py`**: Manages redshift data fetching via catsHTM cross-matching
   - Tries multiple catalogs (GAIA, SDSS, NED, SIMBAD, etc.)
   - Falls back to direct API calls if needed
   - Calculates distances using Hubble's Law
   - Skips objects that already have redshift data

4. **`date_filter.py`**: Fetches first and last detection dates for TDE objects
   - Queries ALeRCE API for detection dates
   - Calculates observation duration
   - Skips objects that already have date data

5. **`curve_filter.py`**: Analyzes light curves and computes metrics
   - Fetches light curve data from ALeRCE
   - Calculates peak luminosity from flux and distance
   - Fits t^(-5/3) power law decay model
   - Computes R² goodness-of-fit metric
   - Skips objects that already have curve data

6. **`offset.py`**: Computes angular offsets and normalized offsets using DELIGHT
   - Uses astro-delight library for host galaxy identification
   - Downloads Pan-STARRS images
   - Computes angular separation between transient and host
   - Estimates host galaxy size (semi-major axis)
   - Calculates normalized offset (offset/host_size)
   - Skips objects that already have offset data

7. **`scorer.py`**: Calculates TDE likelihood scores and ranks candidates
   - Applies weighted scoring based on multiple criteria
   - Computes Primary Score (offset, R², duration, host size)
   - Applies bonus score for redshift and luminosity
   - Ranks candidates by Total TDE Score
   - Saves scored and sorted data back to CSV

8. **`ztf_objects_summary.csv`**: Output file containing all object data
   - Contains all computed metrics and scores
   - Updated incrementally as each step runs

### Configuration Files:

- **`requirements.txt`**: Standard Python dependencies (for general use)
- **`requirements-py310.txt`**: Python 3.10 specific dependencies with pinned versions
  - Includes TensorFlow-Metal for Apple Silicon Macs
  - Required for DELIGHT functionality (option 5)

## Features

### 🚀 **Efficient Data Fetching**
- Queries 500 objects at a time (vs. 2 in original version)
- Stops immediately when target number is reached
- Checks for duplicates before adding
- Continues fetching until required number of unique objects found

### 🔄 **Independent Operations**
- All operations can run independently
- Each step checks for existing data and skips populated fields
- Resume interrupted processes at any point
- No need to re-run completed steps

### 📊 **Comprehensive Analysis**
- **Redshift Analysis**: Multi-catalog cross-matching
- **Light Curve Analysis**: Peak luminosity and decay fitting
- **Angular Offset**: Host galaxy identification and offset calculation
- **TDE Scoring**: Weighted scoring system for candidate ranking

### 🎯 **Smart Data Management**
- Automatically skips objects with existing data
- Preserves existing data when re-running steps
- Handles duplicate OIDs correctly
- Updates only missing fields

## Installation

### Prerequisites:
1. **Python 3.10** (required for DELIGHT/TensorFlow functionality)
   - For other features, Python 3.7+ is sufficient
2. **Internet connection** for API access
3. **macOS with Apple Silicon** (for TensorFlow-Metal) OR **Windows/Linux** (for standard TensorFlow)

### Virtual Environment Setup

#### For macOS (Apple Silicon):

1. **Install Python 3.10** (if not already installed):
   ```bash
   # Using Homebrew
   brew install python@3.10
   
   # Or download from python.org
   ```

2. **Navigate to project directory**:
   ```bash
   cd DataMine_Group1
   ```

3. **Create virtual environment with Python 3.10**:
   ```bash
   python3.10 -m venv .venv310
   ```

4. **Activate virtual environment**:
   ```bash
   source .venv310/bin/activate
   ```

5. **Upgrade pip**:
   ```bash
   pip install --upgrade pip
   ```

6. **Install dependencies**:
   ```bash
   pip install -r requirements-py310.txt
   ```

7. **Verify DELIGHT installation** (for option 5):
   ```bash
   python -c "from delight.delight import Delight; print('DELIGHT installed successfully')"
   ```

#### For Windows:

1. **Install Python 3.10** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Navigate to project directory**:
   ```cmd
   cd DataMine_Group1
   ```

3. **Create virtual environment with Python 3.10**:
   ```cmd
   python -m venv .venv310
   ```
   Or if you have multiple Python versions:
   ```cmd
   py -3.10 -m venv .venv310
   ```

4. **Activate virtual environment**:
   ```cmd
   .venv310\Scripts\activate
   ```

5. **Upgrade pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

6. **Install dependencies**:
   ```cmd
   pip install -r requirements-py310.txt
   ```

7. **Note**: On Windows, TensorFlow-Metal is not available. The system will use standard TensorFlow.

#### For Linux:

Follow the same steps as Windows, but use:
```bash
source .venv310/bin/activate
```

### Quick Start (Without DELIGHT)

If you only need basic functionality (options 1-4, 6) without DELIGHT:

```bash
# Create standard virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install basic dependencies
pip install -r requirements.txt
```

## Usage

### Main Interface:
```bash
python main.py
```

This will show an interactive menu:
```
==================================================
           DataMine Physics Tool
==================================================
Choose an option:
1. Fetch new ZTF objects
2. Fetch redshift data for existing objects in CSV
3. Fetch first/last detection dates for objects in CSV
4. Filter curve data for objects in CSV
5. Compute angular offsets and normalized offsets for objects in CSV
6. Calculate TDE scores and rank candidates
7. Exit
==================================================
```

### Workflow Steps:

#### **Option 1: Fetch New ZTF Objects**
- Enter number of objects to fetch
- Specify datapoint range (beginning and ending)
- System efficiently queries ALeRCE API
- Checks for duplicate OIDs before adding
- Continues fetching until required number of unique objects found
- Saves/appends to `ztf_objects_summary.csv`

#### **Option 2: Fetch Redshift Data**
- Processes existing objects in CSV
- Tries multiple catalogs (GAIA, SDSS, NED, SIMBAD, etc.)
- Performs catsHTM cross-matching for redshift data
- Calculates distances using Hubble's Law
- Updates CSV with redshift and distance columns
- Skips objects that already have redshift data

#### **Option 3: Fetch Detection Date Ranges**
- Processes existing objects in CSV
- Queries ALeRCE API for all detections per object
- Extracts first and last detection dates (MJD format)
- Calculates observation duration (days)
- Updates CSV with date columns: `first_mjd`, `last_mjd`, `first_utc`, `last_utc`, `duration_days`
- Skips objects that already have date data

#### **Option 4: Filter Curve Data**
- Processes existing objects in CSV
- Fetches light curve data from ALeRCE
- Calculates peak luminosity from flux and distance
- Fits t^(-5/3) power law decay model
- Computes R² goodness-of-fit metric
- Updates CSV with `peak_luminosity` and `r_squared` columns
- Skips objects that already have curve data

#### **Option 5: Compute Angular Offsets** (Requires Python 3.10 + DELIGHT)
- **IMPORTANT**: Requires activated Python 3.10 virtual environment
- Processes existing objects in CSV
- Uses DELIGHT to download Pan-STARRS images
- Identifies host galaxy positions
- Computes angular separation between transient and host (arcsec)
- Estimates host galaxy size (semi-major axis in arcsec)
- Calculates normalized offset (angular_offset / host_size)
- Updates CSV with `angular_offset_arcsec`, `host_size_arcsec`, and `angular_normalized_offset` columns
- Skips objects that already have offset data
- **Note**: This step can take significant time as it downloads images and runs DELIGHT

#### **Option 6: Calculate TDE Scores**
- Processes existing objects in CSV
- Applies weighted scoring based on:
  - Angular normalized offset (55% weight)
  - R² fit quality (30% weight)
  - Duration (10% weight)
  - Host size (5% weight)
- Applies bonus score for objects with redshift and peak luminosity
- Ranks candidates by Total TDE Score
- Updates CSV with score columns and sorts by score
- Displays top 10 TDE candidates

## Output Format

The `ztf_objects_summary.csv` file contains comprehensive data:

```csv
oid,num_detections,redshift,distance_mpc,first_mjd,last_mjd,first_utc,last_utc,duration_days,peak_luminosity,r_squared,angular_offset_arcsec,host_size_arcsec,angular_normalized_offset,offset_score,r_squared_score,duration_score,host_size_score,Primary_Score,Bonus,Total_TDE_Score
ZTF17aaaaaby,24,0.123456,527.45,58509.199,59815.444,2019-01-26T...,2022-08-24T...,1306.245,2.07e-06,0.845,0.454,3.404,0.133,1.0,1.0,0.5,1.0,0.875,0.1,0.975
```

**Column Descriptions:**

### Basic Data:
- `oid`: ZTF object identifier
- `num_detections`: Total number of detections/observations

### Redshift & Distance:
- `redshift`: Redshift value from cross-matching (if available)
- `distance_mpc`: Distance in megaparsecs calculated from redshift

### Temporal Data:
- `first_mjd`: Modified Julian Date of first detection
- `last_mjd`: Modified Julian Date of last detection
- `first_utc`: UTC timestamp of first detection
- `last_utc`: UTC timestamp of last detection
- `duration_days`: Number of days between first and last detection

### Light Curve Metrics:
- `peak_luminosity`: Peak luminosity calculated from flux and distance
- `r_squared`: R² goodness-of-fit for t^(-5/3) decay model

### Angular Offset Metrics:
- `angular_offset_arcsec`: Angular separation between transient and host (arcsec)
- `host_size_arcsec`: Host galaxy semi-major axis (arcsec)
- `angular_normalized_offset`: Normalized offset (angular_offset / host_size)

### Scoring Metrics:
- `offset_score`: Score for normalized offset (0.0, 0.5, or 1.0)
- `r_squared_score`: Score for R² fit quality (0.0, 0.5, or 1.0)
- `duration_score`: Score for duration (0.0, 0.5, or 1.0)
- `host_size_score`: Score for host size (0.0, 0.5, or 1.0)
- `Primary_Score`: Weighted sum of individual scores (max 1.0)
- `Bonus`: Bonus score for redshift + luminosity (0.0 or 0.1)
- `Total_TDE_Score`: Final TDE likelihood score (max 1.0)

## Changes to Existing Files

### `main.py`
- **Added**: Import for `curve_filter`, `offset`, and `scorer` modules
- **Added**: Menu options 4, 5, and 6
- **Updated**: Menu now has 7 options (Exit moved to 7)
- **Added**: CSV path resolution using `BASE_DIR`

### `datapoint_filter.py`
- **Added**: Duplicate OID checking before adding to CSV
- **Added**: Continues fetching until required number of unique objects found
- **Added**: Debug output showing skipped duplicates
- **Changed**: Now filters out existing OIDs during fetching process

### `redshifts.py`
- **Added**: Multi-catalog support (tries GAIA, SDSS, NED, SIMBAD, etc.)
- **Added**: Direct API call fallback
- **Added**: Extensive debug output
- **Added**: Skips objects that already have redshift data
- **Added**: Better error handling and traceback

### `date_filter.py`
- **Added**: Skips objects that already have date data
- **Added**: Summary message showing how many objects are skipped
- **Added**: UTC timestamp columns (`first_utc`, `last_utc`)

## New Files

### `curve_filter.py`
- **Purpose**: Analyzes light curves and computes peak luminosity and R² fit
- **Features**:
  - Fetches light curve data from ALeRCE
  - Calculates peak luminosity from flux and distance
  - Fits t^(-5/3) power law decay model
  - Computes R² goodness-of-fit
  - Skips objects that already have curve data
- **Output**: Adds `peak_luminosity` and `r_squared` columns to CSV

### `offset.py`
- **Purpose**: Computes angular offsets using DELIGHT
- **Features**:
  - Uses astro-delight library for host galaxy identification
  - Downloads Pan-STARRS images
  - Computes angular separation between transient and host
  - Estimates host galaxy size
  - Calculates normalized offset
  - Handles file path issues and missing files
  - Skips objects that already have offset data
- **Dependencies**: Requires Python 3.10 and TensorFlow-Metal (for Apple Silicon)
- **Output**: Adds `angular_offset_arcsec`, `host_size_arcsec`, and `angular_normalized_offset` columns to CSV

### `scorer.py`
- **Purpose**: Calculates TDE likelihood scores and ranks candidates
- **Features**:
  - Weighted scoring system:
    - Angular normalized offset: 55%
    - R² fit quality: 30%
    - Duration: 10%
    - Host size: 5%
  - Bonus score for redshift + luminosity
  - Ranks candidates by Total TDE Score
  - Saves scored and sorted data back to CSV
- **Output**: Adds score columns and sorts CSV by `Total_TDE_Score`

## Performance Improvements

- **10x faster**: Queries 500 objects at a time vs. 2
- **Smart batching**: Stops when target reached
- **Resumable**: Can continue interrupted processes
- **Memory efficient**: Processes data in chunks
- **Smart skipping**: Only processes objects missing data

## Error Handling

- Graceful API error handling
- Missing coordinate detection
- Network timeout management
- CSV file validation
- DELIGHT file path handling
- Duplicate column prevention
- Comprehensive error messages with tracebacks

## Development Setup

### For Contributors:
1. Fork the repository
2. Create a feature branch
3. Set up Python 3.10 virtual environment (see Installation section)
4. Make your changes
5. Test with small datasets first
6. Submit a pull request

### Testing:
```bash
# Activate virtual environment first
source .venv310/bin/activate  # On Windows: .venv310\Scripts\activate

# Test with small dataset
python main.py
# Choose option 1, fetch 2-3 objects
# Choose option 2, fetch redshift data
# Choose option 3, fetch detection dates
# Choose option 4, filter curve data
# Choose option 5, compute offsets (requires DELIGHT)
# Choose option 6, calculate scores
```

## Troubleshooting

### Common Issues:

1. **"CSV file not found"**: Run option 1 first to create the file

2. **"DELIGHT is not available"** (Option 5):
   - Make sure Python 3.10 virtual environment is activated
   - Verify DELIGHT installation: `python -c "from delight.delight import Delight"`
   - Reinstall if needed: `pip install astro-delight`

3. **"TensorFlow errors"** (Option 5):
   - On macOS: Make sure TensorFlow-Metal is installed
   - Try reinstalling: `pip install --force-reinstall tensorflow-macos tensorflow-metal`

4. **"No redshift found"**:
   - The system tries multiple catalogs automatically
   - Some objects may not have redshift matches in any catalog
   - Check debug output to see which catalogs were tried

5. **"Duplicate OIDs"**:
   - Option 1 now automatically checks for duplicates
   - Existing duplicates in CSV won't cause issues
   - New fetches will skip existing OIDs

6. **"API timeouts"**:
   - The system includes automatic retries and delays
   - Network issues may require retrying the operation

### Performance Tips:

- Start with small numbers (2-5 objects) for testing
- Each step can be run independently
- The system automatically skips objects that already have data
- Option 5 (offsets) can take significant time - be patient
- Option 6 (scoring) is fast and can be run multiple times

### About Modified Julian Date (MJD):
- MJD is a standard astronomical time format
- MJD = Julian Date - 2400000.5
- Easier to work with than full Julian Dates
- Example: MJD 58509.199 corresponds to approximately January 26, 2019
- Duration is automatically calculated in days between first and last detection

## Requirements Files

### `requirements.txt`
- Standard dependencies for basic functionality
- Use for Python 3.7+ without DELIGHT

### `requirements-py310.txt`
- Python 3.10 specific dependencies
- Includes TensorFlow-Metal for Apple Silicon
- Required for DELIGHT functionality (option 5)
- Pinned versions for compatibility

## License

[Add your license information here]

## Contributors

[Add contributor information here]
