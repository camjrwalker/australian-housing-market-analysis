# Australian Housing Market Analytics - Python Scripts

## Overview

This package contains the Python ETL pipeline and analysis scripts for the Australian Housing Market Analytics project, as defined in Section 4.1 of the project specification.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐              │
│  │   Stage 1    │     │   Stage 2    │    │   Stage 3    │              │
│  │  Ingestion   │───▶│   Staging    │───▶│   Storage    │              │
│  │  (Python)    │     │   (Excel)    │    │    (SQL)     │              │
│  └──────────────┘     └──────────────┘    └──────────────┘              │
│         │                                        │                      │
│         │                                        ▼                      │
│         │                               ┌──────────────┐                │
│         │                               │   Stage 4    │                │
│         └─────────────────────────────▶│  Transform   │                │
│                                         │  (Python)    │                │
│                                         └──────────────┘                │
│                                                │                        │
│                                                ▼                        │
│                                        ┌──────────────┐                 │
│                                        │   Stage 5    │                 │
│                                        │   Analysis   │                 │
│                                        │  (Python)    │                 │
│                                        └──────────────┘                 │
│                                                │                        │
│                                                ▼                        │
│                                        ┌──────────────┐                 │
│                                        │  Dashboard   │                 │
│                                        │(Tableau/PBI) │                 │
│                                        └──────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Scripts

### 1. `01_data_ingestion.py`
**Purpose:** Automated data acquisition and validation

**Features:**
- Load CSV files from ABS and RBA sources
- Validate data against predefined schemas
- Log all ingestion operations
- Handle missing/malformed data

**Key Classes:**
- `DataIngestionPipeline`: Main ingestion orchestrator

**Usage:**
```python
from data_ingestion import DataIngestionPipeline
pipeline = DataIngestionPipeline(output_dir="raw_data")
dataframes = pipeline.run_ingestion(data_files)
```

---

### 2. `02_data_staging.py`
**Purpose:** Data profiling and cleansing (Excel/Power Query equivalent)

**Features:**
- Comprehensive data profiling
- Date format standardisation
- Missing value handling strategies
- State/territory name normalisation
- Duplicate removal

**Key Classes:**
- `DataStagingPipeline`: Staging and profiling operations

**Cleansing Rules Applied:**
1. Dates → ISO format (YYYY-MM-DD)
2. State codes → Standardised 3-letter codes
3. Missing values → Forward fill for time series
4. Duplicates → Remove keeping first occurrence

---

### 3. `03_database_storage.py`
**Purpose:** SQLite database with star schema design

**Schema:**

**Dimension Tables:**
- `dim_date` - Date dimension with fiscal year support
- `dim_location` - Australian capitals and regions
- `dim_property_type` - Property classifications

**Fact Tables:**
- `fact_housing_prices` - Price data by location/date
- `fact_dwelling_approvals` - Building approval statistics
- `fact_population` - Demographic data
- `fact_interest_rates` - RBA cash rate history

**Key Classes:**
- `HousingDatabaseManager`: Database operations

**Usage:**
```python
from database_storage import HousingDatabaseManager
db = HousingDatabaseManager("australian_housing.db")
db.connect()
db.execute_ddl()
db.populate_dim_date()
```

---

### 4. `04_data_transformation.py`
**Purpose:** Feature engineering for analysis

**Features Created:**
- Rolling averages (MA2, MA4, MA12)
- Year-on-year growth rates
- Quarter-on-quarter growth rates
- Affordability ratios (price-to-income)
- Seasonality indicators
- Volatility metrics
- Market phase classification
- Lagged features for modelling

**Key Classes:**
- `FeatureEngineeringPipeline`: All transformation logic

**Affordability Calculations:**
```
Price-to-Income Ratio = Median Price / Median Household Income
Years to Save Deposit = (Price × 20%) / (Income × 30% savings rate)
Mortgage-to-Income = (Annual Mortgage Payment / Income) × 100
```

---

### 5. `05_statistical_analysis.py`
**Purpose:** Descriptive, diagnostic, and predictive analytics

**Analytics Types:**

**Descriptive:**
- Summary statistics (mean, median, std, quartiles)
- Regional comparisons
- Trend analysis

**Diagnostic:**
- Correlation analysis
- Interest rate impact assessment
- Outlier detection (IQR and Z-score methods)

**Predictive:**
- Random Forest regression for price prediction
- Exponential smoothing forecasts
- Scenario analysis (bull/bear/base cases)

**Key Classes:**
- `HousingMarketAnalytics`: Complete analytics suite

**Models:**
1. **Random Forest Regressor**
   - Multi-factor price prediction
   - Feature importance ranking
   - Train/test split validation

2. **Exponential Smoothing**
   - Time series forecasting
   - 95% confidence intervals
   - Trend direction identification

---

### 6. `run_pipeline.py`
**Purpose:** Master orchestration script

**Usage:**
```bash
# Run full pipeline
python run_pipeline.py --data-dir /path/to/csv/files

# Run specific stage
python run_pipeline.py --stage 3 --data-dir /path/to/csv/files
```

---

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. Place CSV data files in a directory
2. Copy all Python scripts to the same directory
3. Run:
```bash
python run_pipeline.py --data-dir .
```

## Output Files

| Stage | Output Directory | Files |
|-------|------------------|-------|
| 1 | `raw_data/` | `ingestion_log.json` |
| 2 | `staged_data/` | `*_staged.csv`, `*_staged.xlsx`, `profiling_report.json` |
| 3 | Current | `australian_housing.db` |
| 4 | `transformed_data/` | `*_transformed.csv`, `*_transformed.xlsx` |
| 5 | `analysis_output/` | `analysis_results.json`, `analysis_report.txt` |

## Skills Demonstrated

| Technology | Application |
|------------|-------------|
| **Python** | ETL automation, statistical analysis, ML models |
| **pandas** | Data manipulation, profiling, transformation |
| **NumPy** | Numerical computations |
| **scikit-learn** | Random Forest, preprocessing, metrics |
| **SQLite** | Database design, DDL, complex queries |
| **SQL** | Star schema, views, indexes, CTEs |

## Dependencies

See `requirements.txt` for complete list. Key packages:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0
- openpyxl >= 3.1.0

## License

This project is part of a portfolio demonstration. Data sourced from Australian Government open data initiatives (CC BY 4.0).

---

*Created: January 2026*
