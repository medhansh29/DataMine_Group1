# DataMine_Group1: ZTF Object Data Fetcher with Redshift Analysis

## Overview
This repository contains tools for fetching ZTF (Zwicky Transient Facility) object data, performing redshift analysis, and tracking detection date ranges for Tidal Disruption Events (TDEs). The system efficiently queries the ALeRCE API to collect object information, cross-match with external catalogs for redshift data, and extract temporal information about each event's observation period.

## File Description:

### Core Files:
1. **`main.py`**: Main entry point with interactive menu system
2. **`datapoint_filter.py`**: Handles ZTF object fetching with efficient batching (500 objects at a time)
3. **`redshifts.py`**: Manages redshift data fetching via catsHTM cross-matching
4. **`date_filter.py`**: Fetches first and last detection dates for TDE objects from ALeRCE
5. **`ztf_objects_summary.csv`**: Output file containing object data with redshift, distance, and detection date information
6. **`requirements.txt`**: Python dependencies

### Legacy Files:
- **`alerce_client.py`**: Original client (now superseded by datapoint_filter.py)
- **`fetch_data.py`**: Legacy data fetching module

## Features:

### 🚀 **Efficient Data Fetching**
- Queries 500 objects at a time (vs. 2 in original version)
- Stops immediately when target number is reached
- Appends to existing CSV instead of overwriting

### 🔄 **Independent Operations**
- Fetch new ZTF objects independently
- Fetch redshift data for existing objects independently
- Fetch detection date ranges independently
- Resume interrupted processes

### 📊 **Comprehensive Output**
- Object IDs (oid)
- Number of detections (num_detections)
- Redshift values (redshift)
- Distance in megaparsecs (distance_mpc)
- First detection date (first_mjd - Modified Julian Date format)
- Last detection date (last_mjd - Modified Julian Date format)
- Duration of observation period (duration_days)

### 📅 **Detection Date Tracking**
- Automatically extracts first and last detection dates from ALeRCE
- Calculates observation duration in days
- Uses Modified Julian Date (MJD) format for astronomical precision
- Tracks the full temporal span of each TDE event

## How to Run:

### Prerequisites:
1. **Python 3.7+** installed
2. **Cursor IDE** (recommended) or any Python IDE
3. **Internet connection** for API access

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/medhansh29/DataMine_Group1.git
   cd DataMine_Group1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage:

#### **Main Interface:**
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
4. Exit
==================================================
```

#### **Option 1: Fetch New ZTF Objects**
- Enter number of objects to fetch
- Specify datapoint range (beginning and ending)
- System efficiently queries ALeRCE API
- Saves/appends to `ztf_objects_summary.csv`

#### **Option 2: Fetch Redshift Data**
- Processes existing objects in CSV
- Performs catsHTM cross-matching for redshift data
- Calculates distances using Hubble's Law
- Updates CSV with redshift and distance columns
- Skips objects that already have redshift data

#### **Option 3: Fetch Detection Date Ranges**
- Processes existing objects in CSV
- Queries ALeRCE API for all detections per object
- Extracts first and last detection dates (MJD format)
- Calculates observation duration (days)
- Updates CSV with date columns: `first_mjd`, `last_mjd`, `duration_days`
- Skips objects that already have date data
- Useful for analyzing the temporal span of TDE events

## Output Format:

The `ztf_objects_summary.csv` file contains:
```csv
oid,num_detections,redshift,distance_mpc,first_mjd,last_mjd,duration_days
ZTF17aaaaaby,24,0.123456,527.45,58509.199,59815.444,1306.245
ZTF17aaaaaco,45,0.234567,1004.12,58348.446,60565.421,2216.975
```

**Column Descriptions:**
- `oid`: ZTF object identifier
- `num_detections`: Total number of detections/observations
- `redshift`: Redshift value from cross-matching (if available)
- `distance_mpc`: Distance in megaparsecs calculated from redshift
- `first_mjd`: Modified Julian Date of first detection (astronomical time format)
- `last_mjd`: Modified Julian Date of last detection
- `duration_days`: Number of days between first and last detection

## Performance Improvements:

- **10x faster**: Queries 500 objects at a time vs. 2
- **Smart batching**: Stops when target reached
- **Resumable**: Can continue interrupted processes
- **Memory efficient**: Processes data in chunks

## Error Handling:
- Graceful API error handling
- Missing coordinate detection
- Network timeout management
- CSV file validation

## Development Setup:

### For Contributors:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with small datasets first
5. Submit a pull request

### Testing:
```bash
# Test with small dataset first
python main.py
# Choose option 1, fetch 2-3 objects
# Choose option 2, fetch redshift data
# Choose option 3, fetch detection dates
```

## Troubleshooting:

### Common Issues:
1. **"CSV file not found"**: Run option 1 first to create the file
2. **API timeouts**: The system includes automatic retries and delays
3. **Missing redshift data**: Some objects may not have redshift matches in external catalogs

### Performance Tips:
- Start with small numbers (2-5 objects) for testing
- Use option 2 to add redshift data to existing objects
- Use option 3 to add detection date ranges to existing objects
- The system automatically skips objects that already have data (redshift or dates)
- Date fetching includes API rate limiting (0.25s delay) to be respectful to ALeRCE servers

### About Modified Julian Date (MJD):
- MJD is a standard astronomical time format
- MJD = Julian Date - 2400000.5
- Easier to work with than full Julian Dates
- Example: MJD 58509.199 corresponds to approximately January 26, 2019
- Duration is automatically calculated in days between first and last detection
