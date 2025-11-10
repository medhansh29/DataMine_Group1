# DataMine_Group1: ZTF Object Data Fetcher with TDE Analysis

## Overview
This repository contains a comprehensive toolkit for fetching ZTF (Zwicky Transient Facility) object data, performing redshift analysis, computing angular offsets, analyzing light curves, scoring Tidal Disruption Event (TDE) candidates, and applying machine learning-based anomaly detection. The system efficiently queries the ALeRCE API, uses DELIGHT for host galaxy identification, and provides a complete workflow for TDE candidate identification and ranking, including an Isolation Forest-based anomaly detection model that prioritizes R² (t^(-5/3) decay fit) as the primary TDE signature.

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
   - **Uses only r-band (fid=2) data** for consistent fitting (cannot mix r and g bands)
   - Calculates peak luminosity from flux and distance
   - **Robust fitting**: Iteratively removes outliers and rebrightening points
   - Fits t^(-5/3) power law decay model using Median Absolute Deviation (MAD) for outlier detection
   - Computes R² goodness-of-fit metric
   - Handles noisy/variable light curves with rebrightening
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
   - Includes `ML_score` column from anomaly detection model

### Machine Learning Model Files (MLModel/):

9. **`MLModel/training_data.py`**: Generates training data for anomaly detection
   - Fetches ALeRCE features for background objects (SNIa, SNII, QSO, AGN)
   - Uses parallel processing (asyncio + ThreadPoolExecutor) for efficient API calls
   - Calculates custom TDE features including R² for t^(-5/3) decay fit
   - Generates `background_training_data.csv` with ~20k objects (500 per class)
   - Preserves critical TDE features (r_squared_t53) even if mostly NaN

10. **`MLModel/anamoly_detection.py`**: Trains Isolation Forest anomaly detection model
    - Loads training data from `background_training_data.csv`
    - Applies 60-40 feature weighting (R² gets 60% weight, other features 40%)
    - Uses custom `FeatureWeightScaler` transformer for feature weighting
    - Trains Isolation Forest with 1% contamination rate
    - Saves trained pipeline as `tde_anomaly_detector_pipeline.pkl`

11. **`MLModel/testing_candidate.py`**: Scores TDE candidates using trained model
    - Loads candidates from `ztf_objects_summary.csv` with filtering options
    - Filters by normalized offset (< 0.1) and TDE score (>= 0.7)
    - Retrieves ALeRCE features and calculates custom TDE features
    - Loads R² values from CSV (does not recalculate)
    - Scores candidates using trained anomaly detection model
    - Updates `ztf_objects_summary.csv` with `ML_score` (anomaly score)
    - Lower scores indicate higher anomaly (more TDE-like)

12. **`MLModel/feature_transforms.py`**: Shared custom transformer
    - `FeatureWeightScaler`: Applies feature-specific weights before scaling
    - Used by both training and testing scripts
    - Ensures pickle compatibility for saved models

13. **`MLModel/background_training_data.csv`**: Training dataset
    - Contains ALeRCE features + custom TDE features for background objects
    - Generated by `training_data.py`
    - Used to train the anomaly detection model

14. **`MLModel/tde_anomaly_detector_pipeline.pkl`**: Trained model
    - Saved scikit-learn pipeline with feature weighting, scaling, and Isolation Forest
    - Loaded by `testing_candidate.py` to score candidates

### Configuration Files:

- **`requirements.txt`**: Standard Python dependencies (for general use)
- **`requirements-py310.txt`**: Python 3.10 specific dependencies for **macOS (Apple Silicon)**
  - Includes TensorFlow-Metal for Apple Silicon Macs
  - Required for DELIGHT functionality (option 5)
- **`requirements-py310-windows.txt`**: Python 3.10 specific dependencies for **Windows**
  - Includes standard TensorFlow (Windows-compatible)
  - Required for DELIGHT functionality (option 5) on Windows

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
- **Light Curve Analysis**: Peak luminosity and robust decay fitting (r-band only)
- **Angular Offset**: Host galaxy identification and offset calculation
- **TDE Scoring**: Weighted scoring system for candidate ranking
- **ML Anomaly Detection**: Isolation Forest model with 60-40 R² weighting

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

6. **Install dependencies** (use Windows-specific requirements file):
   ```cmd
   pip install -r requirements-py310-windows.txt
   ```

7. **Verify DELIGHT installation** (for option 5):
   ```cmd
   python -c "from delight.delight import Delight; print('DELIGHT installed successfully')"
   ```
   
   **Note**: On Windows, use `requirements-py310-windows.txt` which includes standard TensorFlow instead of TensorFlow-Metal (macOS only).

#### For Linux:

1. **Install Python 3.10** (if not already installed):
   ```bash
   # Using package manager (Ubuntu/Debian)
   sudo apt-get install python3.10 python3.10-venv
   
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

6. **Install dependencies** (use Windows requirements file - same TensorFlow):
   ```bash
   pip install -r requirements-py310-windows.txt
   ```

7. **Verify DELIGHT installation** (for option 5):
   ```bash
   python -c "from delight.delight import Delight; print('DELIGHT installed successfully')"
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
- **Uses only r-band (fid=2) data** for consistent fitting (cannot mix r and g bands)
- Calculates peak luminosity from flux and distance
- **Robust fitting**: Iteratively removes outliers and rebrightening points
- Fits t^(-5/3) power law decay model using MAD-based outlier detection
- Handles noisy/variable light curves with rebrightening
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

### Machine Learning Workflow:

#### **Step 1: Generate Training Data**
```bash
cd MLModel
python training_data.py
```
- Fetches ALeRCE features for background objects (SNIa, SNII, QSO, AGN)
- Uses parallel processing for efficient API calls
- Calculates custom TDE features including R² for t^(-5/3) decay
- Generates `background_training_data.csv` (~20k objects)
- **Note**: This can take significant time due to API rate limits

#### **Step 2: Train Anomaly Detection Model**
```bash
cd MLModel
python anamoly_detection.py
```
- Loads training data from `background_training_data.csv`
- Applies 60-40 feature weighting (R²: 60%, other features: 40%)
- Trains Isolation Forest anomaly detection model
- Saves trained pipeline as `tde_anomaly_detector_pipeline.pkl`
- **Note**: Model must be retrained if training data changes

#### **Step 3: Score TDE Candidates**
```bash
cd MLModel
python testing_candidate.py
```
- Loads candidates from `ztf_objects_summary.csv` (parent directory)
- Filters by normalized offset (< 0.1) and TDE score (>= 0.7)
- Retrieves ALeRCE features and calculates custom TDE features
- Loads R² values from CSV (does not recalculate)
- Scores candidates using trained anomaly detection model
- Updates `ztf_objects_summary.csv` with `ML_score` column
- **Lower ML_score values indicate higher anomaly (more TDE-like)**

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
- `ML_score`: Anomaly detection score from Isolation Forest model
  - **Lower values = more anomalous (more TDE-like)**
  - Negative scores indicate outliers/anomalies
  - Typically ranges from -0.1 to 0.1 for TDE candidates

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

### `curve_filter.py`
- **Added**: R-band only filtering (fid=2) - cannot mix r and g band data
- **Added**: Robust fitting mechanism:
  - Pre-filters obvious rebrightening (flux increases >5%)
  - Uses Median Absolute Deviation (MAD) for outlier detection
  - Iterative outlier removal and refitting (up to 3 iterations)
  - Preferentially removes points above fit (rebrightening)
  - Tracks best fit across iterations
- **Added**: Better handling of noisy/variable light curves
- **Changed**: Now uses only r-band data for consistent fitting
- **Changed**: Improved error handling for curve fitting failures

## New Files

### `curve_filter.py`
- **Purpose**: Analyzes light curves and computes peak luminosity and R² fit
- **Features**:
  - Fetches light curve data from ALeRCE
  - **Uses only r-band (fid=2) data** for consistent fitting (cannot mix r and g bands)
  - Calculates peak luminosity from flux and distance
  - **Robust fitting mechanism**:
    - Pre-filters obvious rebrightening (flux increases >5%)
    - Uses Median Absolute Deviation (MAD) for outlier detection
    - Iteratively removes outliers and refits (up to 3 iterations)
    - Preferentially removes points above fit (rebrightening)
    - Tracks best fit across iterations
  - Fits t^(-5/3) power law decay model
  - Handles noisy/variable light curves with rebrightening
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

### `MLModel/training_data.py`
- **Purpose**: Generates training data for anomaly detection model
- **Features**:
  - Fetches ALeRCE features for background objects (SNIa, SNII, QSO, AGN)
  - Uses parallel processing (asyncio + ThreadPoolExecutor) for efficient API calls
  - Calculates custom TDE features:
    - `max_mag_r`: Peak magnitude in r-band
    - `peak_mjd`: Time of peak magnitude
    - `rise_duration`: Duration from first detection to peak
    - `r_squared_t53`: R² for t^(-5/3) decay fit (key TDE signature)
  - Preserves critical TDE features even if mostly NaN
  - Generates `background_training_data.csv` with ~20k objects
- **Output**: Saves `background_training_data.csv` in MLModel folder

### `MLModel/anamoly_detection.py`
- **Purpose**: Trains Isolation Forest anomaly detection model
- **Features**:
  - Loads training data from `background_training_data.csv`
  - Applies 60-40 feature weighting:
    - R² (r_squared_t53): 60% weight (PRIMARY TDE signature)
    - Other features: 40% weight (combined)
  - Uses custom `FeatureWeightScaler` transformer
  - Trains Isolation Forest with 1% contamination rate
  - Saves trained pipeline as `tde_anomaly_detector_pipeline.pkl`
- **Output**: Saves trained model pipeline in MLModel folder

### `MLModel/testing_candidate.py`
- **Purpose**: Scores TDE candidates using trained anomaly detection model
- **Features**:
  - Loads candidates from `ztf_objects_summary.csv` (parent directory)
  - Filters by normalized offset (< 0.1) and TDE score (>= 0.7)
  - Retrieves ALeRCE features using parallel processing
  - Calculates custom TDE features (max_mag_r, peak_mjd, rise_duration)
  - **Loads R² values from CSV** (does not recalculate to avoid redundancy)
  - Aligns candidate features with training features
  - Scores candidates using trained Isolation Forest model
  - Updates `ztf_objects_summary.csv` with `ML_score` column
  - **Lower ML_score = more anomalous (more TDE-like)**
- **Output**: Updates `ztf_objects_summary.csv` with `ML_score` column

### `MLModel/feature_transforms.py`
- **Purpose**: Shared custom transformer for feature weighting
- **Features**:
  - `FeatureWeightScaler`: Applies feature-specific weights before scaling
  - Used by both training and testing scripts
  - Ensures pickle compatibility for saved models

## Machine Learning Model Details

### Model Architecture:
- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Feature Weighting**: 60% R² (r_squared_t53), 40% other features
- **Contamination Rate**: 1% (expected fraction of outliers)
- **Training Data**: ~20k background objects (SNIa, SNII, QSO, AGN)
- **Key Features**:
  - ALeRCE light curve features (150+ features)
  - Custom TDE features: max_mag_r, peak_mjd, rise_duration, r_squared_t53
  - R² for t^(-5/3) decay fit is the PRIMARY TDE signature (60% weight)

### Model Workflow:
1. **Training Phase**:
   - Generate training data: `python MLModel/training_data.py`
   - Train model: `python MLModel/anamoly_detection.py`
   - Model saved as: `MLModel/tde_anomaly_detector_pipeline.pkl`

2. **Testing Phase**:
   - Score candidates: `python MLModel/testing_candidate.py`
   - Filters candidates by normalized offset (< 0.1) and TDE score (>= 0.7)
   - Updates `ztf_objects_summary.csv` with `ML_score` column

### Interpreting ML Scores:
- **Negative scores**: Indicate anomalies (more TDE-like)
- **Lower scores = higher anomaly**: More negative = more TDE-like
- **Typical range**: -0.1 to 0.1 for TDE candidates
- **HIGH priority**: Candidates with negative ML_score
- **LOW priority**: Candidates with positive ML_score

### Model Features:
- **60-40 Weighting**: R² features get 1.5x weight, other features get 0.67x weight
- **Robust Feature Alignment**: Automatically aligns candidate features with training features
- **Resume Capability**: Skips candidates that already have ML_score
- **Parallel Processing**: Uses asyncio for efficient API calls

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

### Complete Workflow Example:

#### **Basic Data Collection and Analysis:**
```bash
# Activate virtual environment first
source .venv310/bin/activate  # On Windows: .venv310\Scripts\activate

# Run main workflow
python main.py
# Choose option 1, fetch 2-3 objects
# Choose option 2, fetch redshift data
# Choose option 3, fetch detection dates
# Choose option 4, filter curve data (robust fitting, r-band only)
# Choose option 5, compute offsets (requires DELIGHT)
# Choose option 6, calculate scores
```

#### **Machine Learning Pipeline:**
```bash
# Make sure virtual environment is activated
source .venv310/bin/activate  # On Windows: .venv310\Scripts\activate

# Step 1: Generate training data (can take hours due to API rate limits)
cd MLModel
python training_data.py
# This will generate background_training_data.csv with ~20k objects

# Step 2: Train the anomaly detection model
python anamoly_detection.py
# This will train and save tde_anomaly_detector_pipeline.pkl

# Step 3: Score TDE candidates
python testing_candidate.py
# This will update ztf_objects_summary.csv with ML_score column
# Lower ML_score = more anomalous (more TDE-like)
```

### Testing:
```bash
# Activate virtual environment first
source .venv310/bin/activate  # On Windows: .venv310\Scripts\activate

# Test basic workflow
python main.py
# Choose option 1, fetch 2-3 objects
# Choose option 2, fetch redshift data
# Choose option 3, fetch detection dates
# Choose option 4, filter curve data
# Choose option 5, compute offsets (requires DELIGHT)
# Choose option 6, calculate scores

# Test ML pipeline (after basic workflow)
cd MLModel
python training_data.py  # Generate training data (use small MAX_SAMPLES_PER_CLASS for testing)
python anamoly_detection.py  # Train model
python testing_candidate.py  # Score candidates
```

## Troubleshooting

### Common Issues:

1. **"CSV file not found"**: Run option 1 first to create the file

2. **"DELIGHT is not available"** (Option 5):
   - Make sure Python 3.10 virtual environment is activated
   - Verify DELIGHT installation: `python -c "from delight.delight import Delight"`
   - Reinstall if needed: `pip install astro-delight`

3. **"TensorFlow errors"** (Option 5):
   - **On macOS**: Make sure TensorFlow-Metal is installed
     - Use `requirements-py310.txt` (includes tensorflow-macos and tensorflow-metal)
     - Try reinstalling: `pip install --force-reinstall tensorflow-macos tensorflow-metal`
   - **On Windows/Linux**: Use standard TensorFlow
     - Use `requirements-py310-windows.txt` (includes standard tensorflow)
     - Try reinstalling: `pip install --force-reinstall tensorflow==2.13.0`
   - **Common issue**: Wrong requirements file for your platform
     - macOS: Use `requirements-py310.txt`
     - Windows/Linux: Use `requirements-py310-windows.txt`

4. **"Cannot install tensorflow-macos on Windows"**:
   - This is expected! `tensorflow-macos` is macOS-only
   - Use `requirements-py310-windows.txt` instead of `requirements-py310.txt`
   - Windows uses standard `tensorflow` package

5. **"No redshift found"**:
   - The system tries multiple catalogs automatically
   - Some objects may not have redshift matches in any catalog
   - Check debug output to see which catalogs were tried

6. **"Duplicate OIDs"**:
   - Option 1 now automatically checks for duplicates
   - Existing duplicates in CSV won't cause issues
   - New fetches will skip existing OIDs

7. **"API timeouts"**:
   - The system includes automatic retries and delays
   - Network issues may require retrying the operation

8. **"Feature mismatch in ML model"**:
   - Ensure training data was generated using `training_data.py`
   - Retrain the model if training data changes
   - The model expects features from `background_training_data.csv`
   - Check that `r_squared_t53` column exists in training data

9. **"Low R² values for known TDEs"**:
   - Some TDEs have variable/rebrightening light curves
   - The robust fitting mechanism filters outliers automatically
   - Low R² can still be informative (indicates non-standard decay)
   - Check the light curve visually to understand the variability

10. **"ML_score not updating"**:
    - Ensure `testing_candidate.py` is run from the MLModel directory
    - Check that `ztf_objects_summary.csv` exists in parent directory
    - Verify the model file `tde_anomaly_detector_pipeline.pkl` exists
    - Check that candidates meet filtering criteria (normalized offset < 0.1, TDE score >= 0.7)

### Performance Tips:

- Start with small numbers (2-5 objects) for testing
- Each step can be run independently
- The system automatically skips objects that already have data
- Option 4 (curve filtering) now uses robust fitting for better R² values
- Option 5 (offsets) can take significant time - be patient
- Option 6 (scoring) is fast and can be run multiple times
- ML training data generation can take hours - use parallel processing (already enabled)
- ML model training is fast once training data is ready
- ML candidate scoring is fast and can be run multiple times

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

### `requirements-py310.txt` (macOS Apple Silicon)
- Python 3.10 specific dependencies for **macOS**
- Includes TensorFlow-Metal for Apple Silicon
- Required for DELIGHT functionality (option 5) and ML model training/testing
- Includes scikit-learn, joblib for ML models
- Includes tqdm for progress bars
- Pinned versions for compatibility
- **Use on macOS only**

### `requirements-py310-windows.txt` (Windows/Linux)
- Python 3.10 specific dependencies for **Windows and Linux**
- Includes standard TensorFlow (Windows/Linux-compatible)
- Required for DELIGHT functionality (option 5) and ML model training/testing
- Includes scikit-learn, joblib for ML models
- Includes tqdm for progress bars
- Pinned versions for compatibility
- **Use on Windows and Linux**

## License

[Add your license information here]

## Contributors

[Add contributor information here]
