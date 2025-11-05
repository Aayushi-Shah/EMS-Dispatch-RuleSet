# EMS Data Processing Project

This project processes EMS incident data for Pennsylvania, with a focus on Lancaster County for simulation purposes.

## Project Structure

```
project_root/
│
├── data/
│   ├── raw/                      # Original JSON (massive)
│   │   └── all_incidents.json
│   ├── processed/
│   │   ├── medical_calls_pa_clean.parquet      # statewide clean dataset
│   │   ├── medical_calls_pa_clean_preview.csv
│   │   ├── medical_calls_lancaster.parquet     # filtered subset for simulation
│   │   └── medical_calls_lancaster_preview.csv
│   └── reference/
│       ├── lancaster_ems_stations.csv          # to be added (station info)
│       └── lancaster_units_types.csv
│
└── scripts/
    ├── filter_medical_pa.py
    ├── extract_lancaster_and_audit.py
    └── validate_pa_medical.py
```

## Usage

### 1. Process Raw Data
Filter the raw JSON to medical calls in Pennsylvania:
```bash
cd scripts
python filter_medical_pa.py
```

### 2. Extract Lancaster County Data
Create a Lancaster-specific subset for simulation:
```bash
cd scripts
python extract_lancaster_and_audit.py --in ../data/processed/medical_calls_pa_clean.parquet
```

### 3. Validate Data Quality
Run validation checks on the processed data:
```bash
cd scripts
python validate_pa_medical.py --parquet ../data/processed/medical_calls_pa_clean.parquet --csv ../data/processed/medical_calls_pa_clean_preview.csv
```

## Data Files

- **all_incidents.json**: Raw EMS incident data (very large)
- **medical_calls_pa_clean.parquet**: Cleaned PA medical incidents with UTC timestamps
- **medical_calls_lancaster.parquet**: Lancaster County subset for simulation
- **lancaster_ems_stations.csv**: Placeholder for EMS station information
- **lancaster_units_types.csv**: Placeholder for unit type specifications

## Environment

The project uses a Python virtual environment (`ems-env/`) with dependencies for data processing including pandas, pyarrow, and pdfplumber.




