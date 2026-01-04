"""
Australian Housing Market Analytics - Data Ingestion Script
============================================================
Script 1 of 5: Data Ingestion (Python)

Purpose: Automated scripts using pandas and requests to download CSV/Excel files 
         from ABS and RBA APIs. Data validated against schema definitions.

Author: Cam Walker
Date: January 2026
"""

import pandas as pd
import requests
import os
import json
from datetime import datetime
from pathlib import Path
import logging
import sys

# =============================================================================
# CONFIGURATION - UPDATE THIS PATH FOR YOUR LOCAL ENVIRONMENT
# =============================================================================
DATA_DIRECTORY = r"D:\Users\CameronWalker\Documents\15. Work\New Roles - Data Science\Projects\Housing\raw_data"
# =============================================================================

# Configure logging with UTF-8 encoding to handle special characters
# Using ASCII-only characters in log messages to avoid Windows console encoding issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Handles automated data ingestion from Australian government sources.
    Validates data against predefined schemas and logs all operations.
    """
    
    def __init__(self, data_dir: str = None, output_dir: str = None):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            data_dir: Directory containing the CSV files
            output_dir: Directory for output files (defaults to data_dir)
        """
        # Use provided path or fall back to configured default
        self.data_dir = Path(data_dir) if data_dir else Path(DATA_DIRECTORY)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Validate that the data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion_log = []
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Define data source configurations
        self.data_sources = {
            "abs_dwelling_value": {
                "name": "ABS Total Value of Dwellings",
                "url": "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/total-value-dwellings/latest-release",
                "type": "web_scrape",
                "frequency": "quarterly"
            },
            "rba_cash_rate": {
                "name": "RBA Cash Rate Target",
                "url": "https://www.rba.gov.au/statistics/cash-rate/",
                "type": "web_scrape", 
                "frequency": "as_announced"
            },
            "abs_building_approvals": {
                "name": "ABS Building Approvals",
                "url": "https://www.abs.gov.au/statistics/industry/building-and-construction/building-approvals-australia/latest-release",
                "type": "web_scrape",
                "frequency": "monthly"
            },
            "abs_population": {
                "name": "ABS Regional Population",
                "url": "https://www.abs.gov.au/statistics/people/population/regional-population/latest-release",
                "type": "web_scrape",
                "frequency": "annual"
            }
        }
        
        # Define schema validations
        self.schemas = {
            "dwelling_value": {
                "required_columns": ["Quarter", "Total_Value_Dwelling_Stock_Billion_AUD", "Year"],
                "data_types": {
                    "Quarter": "string",
                    "Total_Value_Dwelling_Stock_Billion_AUD": "float64",
                    "Year": "int64"
                },
                "value_ranges": {
                    "Total_Value_Dwelling_Stock_Billion_AUD": (5000, 15000),
                    "Year": (2015, 2030)
                }
            },
            "cash_rate": {
                "required_columns": ["Effective_Date", "Change_Percent_Points", "Cash_Rate_Target_Percent"],
                "data_types": {
                    "Effective_Date": "datetime64",
                    "Change_Percent_Points": "float64",
                    "Cash_Rate_Target_Percent": "float64"
                },
                "value_ranges": {
                    "Cash_Rate_Target_Percent": (0, 20),
                    "Change_Percent_Points": (-2, 2)
                }
            },
            "dwelling_approvals": {
                "required_columns": ["Month", "Year", "Total_Dwellings_Approved"],
                "data_types": {
                    "Month": "string",
                    "Year": "int64",
                    "Total_Dwellings_Approved": "int64"
                },
                "value_ranges": {
                    "Total_Dwellings_Approved": (5000, 30000),
                    "Year": (2015, 2030)
                }
            },
            "population": {
                "required_columns": ["Capital_City", "Population_2023_24", "Growth_Rate_2023_24_Percent"],
                "data_types": {
                    "Capital_City": "string",
                    "Population_2023_24": "int64",
                    "Growth_Rate_2023_24_Percent": "float64"
                },
                "value_ranges": {
                    "Growth_Rate_2023_24_Percent": (-5, 10)
                }
            },
            "median_prices": {
                "required_columns": ["Quarter", "Year", "Sydney_Median_000", "National_Median_000"],
                "data_types": {
                    "Quarter": "string",
                    "Year": "int64",
                    "Sydney_Median_000": "float64",
                    "National_Median_000": "float64"
                },
                "value_ranges": {
                    "Sydney_Median_000": (500, 3000),
                    "National_Median_000": (300, 2000)
                }
            },
            "home_value_index": {
                "required_columns": ["Month", "Year", "National_MoM_Percent"],
                "data_types": {
                    "Month": "string",
                    "Year": "int64",
                    "National_MoM_Percent": "float64"
                },
                "value_ranges": {
                    "National_MoM_Percent": (-5, 5),
                    "Year": (2020, 2030)
                }
            }
        }
    
    def validate_schema(self, df: pd.DataFrame, schema_name: str) -> dict:
        """
        Validate DataFrame against predefined schema.
        
        Args:
            df: DataFrame to validate
            schema_name: Name of schema to validate against
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        if schema_name not in self.schemas:
            results["warnings"].append(f"No schema defined for '{schema_name}', skipping validation")
            return results
        
        schema = self.schemas[schema_name]
        
        # Check required columns
        missing_cols = set(schema["required_columns"]) - set(df.columns)
        if missing_cols:
            results["valid"] = False
            results["errors"].append(f"Missing required columns: {missing_cols}")
        
        # Check value ranges
        for col, (min_val, max_val) in schema.get("value_ranges", {}).items():
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min < min_val or col_max > max_val:
                    results["warnings"].append(
                        f"Column '{col}' has values outside expected range "
                        f"[{min_val}, {max_val}]: found [{col_min}, {col_max}]"
                    )
        
        # Check for null values
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if len(cols_with_nulls) > 0:
            results["warnings"].append(f"Columns with null values: {dict(cols_with_nulls)}")
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            results["warnings"].append(f"Found {duplicate_count} duplicate rows")
        
        return results
    
    def load_local_csv(self, filename: str, schema_name: str = None) -> pd.DataFrame:
        """
        Load and validate a local CSV file.
        
        Args:
            filename: Name of CSV file (will be loaded from data_dir)
            schema_name: Optional schema name for validation
            
        Returns:
            Validated DataFrame
        """
        # Construct full path
        filepath = self.data_dir / filename
        
        logger.info(f"Loading file: {filepath}")
        
        # Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            if schema_name:
                validation = self.validate_schema(df, schema_name)
                if not validation["valid"]:
                    logger.error(f"Validation failed: {validation['errors']}")
                for warning in validation["warnings"]:
                    logger.warning(warning)
            
            self.ingestion_log.append({
                "timestamp": datetime.now().isoformat(),
                "source": str(filepath),
                "rows": len(df),
                "columns": len(df.columns),
                "status": "success"
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {str(e)}")
            self.ingestion_log.append({
                "timestamp": datetime.now().isoformat(),
                "source": str(filepath),
                "status": "failed",
                "error": str(e)
            })
            raise
    
    def list_available_files(self) -> list:
        """List all CSV files available in the data directory."""
        csv_files = list(self.data_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {self.data_dir}")
        for f in csv_files:
            logger.info(f"  - {f.name}")
        return csv_files
    
    def fetch_web_data(self, url: str, source_name: str) -> str:
        """
        Fetch data from web URL with error handling.
        
        Args:
            url: URL to fetch
            source_name: Name of data source for logging
            
        Returns:
            Response content as string
        """
        logger.info(f"Fetching data from: {source_name}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            logger.info(f"Successfully fetched {len(response.content)} bytes")
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {source_name}: {str(e)}")
            raise
    
    def save_ingestion_log(self):
        """Save ingestion log to JSON file."""
        log_path = self.output_dir / "ingestion_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.ingestion_log, f, indent=2)
        logger.info(f"Ingestion log saved to {log_path}")
    
    def run_ingestion(self, data_files: dict = None) -> dict:
        """
        Run the complete data ingestion pipeline.
        
        Args:
            data_files: Dictionary mapping schema names to file names
                        If None, uses default file mapping
            
        Returns:
            Dictionary of loaded DataFrames
        """
        logger.info("=" * 60)
        logger.info("Starting Data Ingestion Pipeline")
        logger.info("=" * 60)
        
        # Default file mapping if not provided
        if data_files is None:
            data_files = {
                "dwelling_value": "01_abs_total_value_dwellings.csv",
                "cash_rate": "03_rba_cash_rate_history.csv",
                "dwelling_approvals": "05_abs_dwelling_approvals_monthly.csv",
                "population": "04_abs_capital_city_population.csv",
                "median_prices": "07_capital_city_median_prices_quarterly.csv",
                "home_value_index": "06_corelogic_home_value_index.csv"
            }
        
        # List available files first
        self.list_available_files()
        
        dataframes = {}
        
        for schema_name, filename in data_files.items():
            try:
                df = self.load_local_csv(filename, schema_name)
                dataframes[schema_name] = df
                # Using ASCII-compatible characters instead of Unicode checkmarks
                logger.info(f"[OK] Successfully loaded: {schema_name}")
            except FileNotFoundError:
                logger.warning(f"[MISSING] File not found: {filename}")
            except Exception as e:
                logger.error(f"[FAILED] Failed to load: {schema_name} - {str(e)}")
        
        self.save_ingestion_log()
        
        logger.info("=" * 60)
        logger.info(f"Ingestion Complete: {len(dataframes)}/{len(data_files)} files loaded")
        logger.info("=" * 60)
        
        return dataframes


def main():
    """Main entry point for data ingestion."""
    
    print("\n" + "=" * 60)
    print("AUSTRALIAN HOUSING MARKET ANALYTICS")
    print("Data Ingestion Pipeline")
    print("=" * 60 + "\n")
    
    try:
        # Initialize pipeline with configured data directory
        pipeline = DataIngestionPipeline()
        
        # Run ingestion with default file mapping
        dataframes = pipeline.run_ingestion()
        
        # Display summary
        print("\n" + "=" * 60)
        print("DATA INGESTION SUMMARY")
        print("=" * 60)
        
        if len(dataframes) == 0:
            print("\n[WARNING] No files were loaded. Please check:")
            print(f"  1. Data directory exists: {DATA_DIRECTORY}")
            print("  2. CSV files are present in the directory")
            print("  3. File names match expected pattern (e.g., 01_abs_total_value_dwellings.csv)")
        else:
            for name, df in dataframes.items():
                print(f"\n{name.upper()}")
                print(f"  Rows: {len(df):,}")
                print(f"  Columns: {len(df.columns)}")
                print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                print(f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        
        return dataframes
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease update the DATA_DIRECTORY variable at the top of this script")
        print("to point to your local folder containing the CSV files.")
        return {}
    
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()