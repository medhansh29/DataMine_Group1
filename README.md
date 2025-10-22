# DataMine_Group1: ZTF Object Data Fetcher with Redshift Analysis

## Overview
This repository contains tools for fetching ZTF (Zwicky Transient Facility) object data and performing redshift analysis. The system efficiently queries the ALeRCE API to collect object information and cross-match with external catalogs for redshift data.

## File Description:

### Core Files:
1. **`main.py`**: Main entry point with interactive menu system
2. **`datapoint_filter.py`**: Handles ZTF object fetching with efficient batching (500 objects at a time)
3. **`redshifts.py`**: Manages redshift data fetching via catsHTM cross-matching
4. **`ztf_objects_summary.csv`**: Output file containing object data with redshift and distance information
5. **`requirements.txt`**: Python dependencies

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
- Resume interrupted processes

### 📊 **Comprehensive Output**
- Object IDs (oid)
- Number of detections (num_detections)
- Redshift values (redshift)
- Distance in megaparsecs (distance_mpc)

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
3. Exit
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

## Output Format:

The `ztf_objects_summary.csv` file contains:
```csv
oid,num_detections,redshift,distance_mpc
ZTF17aaaaaby,24,0.123456,527.45
ZTF17aaaaaco,45,0.234567,1004.12
```

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
```

## Troubleshooting:

### Common Issues:
1. **"CSV file not found"**: Run option 1 first to create the file
2. **API timeouts**: The system includes automatic retries and delays
3. **Missing redshift data**: Some objects may not have redshift matches in external catalogs

### Performance Tips:
- Start with small numbers (2-5 objects) for testing
- Use option 2 to add redshift data to existing objects
- The system automatically skips objects that already have redshift data
