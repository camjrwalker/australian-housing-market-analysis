"""
Australian Housing Market Analytics - Data Staging Script
==========================================================
Script 2 of 5: Staging (Excel/Power Query Equivalent)

Purpose: Initial data profiling and cleansing. Documented cleansing rules applied:
         handling missing values, standardising date formats, normalising suburb 
         names across datasets.

Author: Cam Walker
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import json
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataStagingPipeline:
    """
    Data staging and profiling pipeline - equivalent to Excel Power Query operations.
    Implements documented cleansing rules for Australian housing market data.
    """
    
    def __init__(self, input_dir: str = "raw_data", output_dir: str = "staged_data"):
        """Initialize the staging pipeline."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiling_results = {}
        self.cleansing_log = []
        
        # Standardised state/territory mappings
        self.state_mappings = {
            'NSW': 'New South Wales',
            'VIC': 'Victoria', 
            'QLD': 'Queensland',
            'SA': 'South Australia',
            'WA': 'Western Australia',
            'TAS': 'Tasmania',
            'NT': 'Northern Territory',
            'ACT': 'Australian Capital Territory',
            'New South Wales': 'NSW',
            'Victoria': 'VIC',
            'Queensland': 'QLD',
            'South Australia': 'SA',
            'Western Australia': 'WA',
            'Tasmania': 'TAS',
            'Northern Territory': 'NT',
            'Australian Capital Territory': 'ACT'
        }
        
        # Capital city standardisation
        self.city_mappings = {
            'Sydney': 'Greater Sydney',
            'Melbourne': 'Greater Melbourne',
            'Brisbane': 'Greater Brisbane',
            'Perth': 'Greater Perth',
            'Adelaide': 'Greater Adelaide',
            'Hobart': 'Greater Hobart',
            'Darwin': 'Greater Darwin',
            'Canberra': 'Australian Capital Territory',
            'ACT': 'Australian Capital Territory'
        }
        
        # Month name mappings
        self.month_mappings = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12',
            'January': '01', 'February': '02', 'March': '03', 'April': '04',
            'June': '06', 'July': '07', 'August': '08',
            'September': '09', 'October': '10', 'November': '11', 'December': '12'
        }
    
    def profile_dataframe(self, df: pd.DataFrame, name: str) -> dict:
        """
        Generate comprehensive data profile (like Power Query profiling).
        
        Args:
            df: DataFrame to profile
            name: Name for the profile
            
        Returns:
            Dictionary containing profile statistics
        """
        logger.info(f"Profiling dataset: {name}")
        
        profile = {
            "name": name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "memory_usage_kb": df.memory_usage(deep=True).sum() / 1024,
            "columns": {},
            "duplicate_rows": int(df.duplicated().sum()),
            "complete_rows": int((~df.isnull().any(axis=1)).sum())
        }
        
        for col in df.columns:
            col_profile = {
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "null_percent": round(df[col].isna().sum() / len(df) * 100, 2),
                "unique_count": int(df[col].nunique()),
                "sample_values": df[col].dropna().head(3).tolist()
            }
            
            # Add numeric statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update({
                    "min": float(df[col].min()) if not df[col].isna().all() else None,
                    "max": float(df[col].max()) if not df[col].isna().all() else None,
                    "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                    "median": float(df[col].median()) if not df[col].isna().all() else None,
                    "std": float(df[col].std()) if not df[col].isna().all() else None
                })
            
            profile["columns"][col] = col_profile
        
        self.profiling_results[name] = profile
        return profile
    
    def standardise_date_format(self, df: pd.DataFrame, date_col: str, 
                                 output_format: str = "%Y-%m-%d") -> pd.DataFrame:
        """
        Standardise date column to consistent format.
        
        Args:
            df: DataFrame containing date column
            date_col: Name of date column
            output_format: Desired output format
            
        Returns:
            DataFrame with standardised dates
        """
        df = df.copy()
        
        original_sample = df[date_col].head(3).tolist()
        
        # Try multiple date parsing strategies
        try:
            df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=True)
            df[date_col] = df[date_col].dt.strftime(output_format)
        except Exception:
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                df[date_col] = df[date_col].dt.strftime(output_format)
            except Exception as e:
                logger.warning(f"Could not parse dates in {date_col}: {e}")
        
        self.cleansing_log.append({
            "operation": "standardise_date",
            "column": date_col,
            "before_sample": original_sample,
            "after_sample": df[date_col].head(3).tolist(),
            "format": output_format
        })
        
        return df
    
    def parse_quarter_to_date(self, df: pd.DataFrame, quarter_col: str) -> pd.DataFrame:
        """
        Convert quarter string (e.g., 'Q1-2025') to date.
        
        Args:
            df: DataFrame containing quarter column
            quarter_col: Name of quarter column
            
        Returns:
            DataFrame with additional date column
        """
        df = df.copy()
        
        def quarter_to_date(q_str):
            if pd.isna(q_str):
                return None
            
            # Handle formats like 'Sep-2020', 'Q1-2025', etc.
            q_str = str(q_str)
            
            # Format: 'Mon-YYYY' (e.g., 'Sep-2020')
            match = re.match(r'(\w+)-(\d{4})', q_str)
            if match:
                month_str, year = match.groups()
                if month_str in self.month_mappings:
                    month = self.month_mappings[month_str]
                    return f"{year}-{month}-01"
                elif month_str.startswith('Q'):
                    # Quarter format
                    quarter_num = int(month_str[1])
                    month = str((quarter_num - 1) * 3 + 1).zfill(2)
                    return f"{year}-{month}-01"
            
            return None
        
        df['Period_Date'] = df[quarter_col].apply(quarter_to_date)
        
        self.cleansing_log.append({
            "operation": "parse_quarter_to_date",
            "source_column": quarter_col,
            "new_column": "Period_Date"
        })
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: dict = None) -> pd.DataFrame:
        """
        Handle missing values according to specified strategy.
        
        Args:
            df: DataFrame with missing values
            strategy: Dictionary mapping columns to strategies
                      Options: 'drop', 'mean', 'median', 'mode', 'ffill', 'bfill', value
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        if strategy is None:
            strategy = {}
        
        # Default strategy: forward fill for time series, drop for others
        for col in df.columns:
            if df[col].isna().any():
                col_strategy = strategy.get(col, 'ffill')
                original_null_count = df[col].isna().sum()
                
                if col_strategy == 'drop':
                    df = df.dropna(subset=[col])
                elif col_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif col_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif col_strategy == 'mode':
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else np.nan)
                elif col_strategy == 'ffill':
                    df[col] = df[col].fillna(method='ffill')
                elif col_strategy == 'bfill':
                    df[col] = df[col].fillna(method='bfill')
                else:
                    df[col] = df[col].fillna(col_strategy)
                
                self.cleansing_log.append({
                    "operation": "handle_missing",
                    "column": col,
                    "strategy": str(col_strategy),
                    "original_null_count": int(original_null_count),
                    "remaining_null_count": int(df[col].isna().sum())
                })
        
        return df
    
    def normalise_state_names(self, df: pd.DataFrame, state_col: str, 
                               output_format: str = 'code') -> pd.DataFrame:
        """
        Normalise state/territory names to consistent format.
        
        Args:
            df: DataFrame containing state column
            state_col: Name of state column
            output_format: 'code' for abbreviations, 'full' for full names
            
        Returns:
            DataFrame with normalised state names
        """
        df = df.copy()
        
        def normalise(value):
            if pd.isna(value):
                return value
            value = str(value).strip()
            
            if output_format == 'code':
                # Convert to code if full name given
                if value in self.state_mappings and len(self.state_mappings[value]) <= 3:
                    return self.state_mappings[value]
                return value if len(value) <= 3 else value[:3].upper()
            else:
                # Convert to full name
                if value in self.state_mappings:
                    result = self.state_mappings[value]
                    if len(result) > 3:
                        return result
                return value
        
        original_values = df[state_col].unique().tolist()
        df[state_col] = df[state_col].apply(normalise)
        
        self.cleansing_log.append({
            "operation": "normalise_state_names",
            "column": state_col,
            "output_format": output_format,
            "original_values": original_values[:10],
            "normalised_values": df[state_col].unique().tolist()[:10]
        })
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: list = None, 
                          keep: str = 'first') -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: DataFrame with potential duplicates
            subset: Columns to consider for identifying duplicates
            keep: 'first', 'last', or False
            
        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep)
        removed_count = original_count - len(df)
        
        self.cleansing_log.append({
            "operation": "remove_duplicates",
            "subset_columns": subset,
            "keep": keep,
            "original_count": original_count,
            "removed_count": removed_count,
            "final_count": len(df)
        })
        
        logger.info(f"Removed {removed_count} duplicate rows")
        return df
    
    def add_calculated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add calculated columns for analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional calculated columns
        """
        df = df.copy()
        
        # Add year-over-year calculations if applicable columns exist
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if 'Price' in col or 'Value' in col or 'Median' in col:
                # Add percentage change
                df[f'{col}_PctChange'] = df[col].pct_change() * 100
                
                # Add 4-period (quarterly = YoY) moving average
                if len(df) >= 4:
                    df[f'{col}_MA4'] = df[col].rolling(window=4).mean()
        
        return df
    
    def export_to_excel_format(self, df: pd.DataFrame, name: str) -> str:
        """
        Export DataFrame to Excel-compatible CSV with formatting.
        
        Args:
            df: DataFrame to export
            name: Output file name
            
        Returns:
            Path to exported file
        """
        output_path = self.output_dir / f"{name}_staged.csv"
        df.to_csv(output_path, index=False)
        
        # Also create an Excel file
        excel_path = self.output_dir / f"{name}_staged.xlsx"
        df.to_excel(excel_path, index=False, sheet_name='Data')
        
        logger.info(f"Exported staged data to {output_path}")
        return str(output_path)
    
    def save_profiling_report(self):
        """Save profiling results to JSON."""
        report_path = self.output_dir / "profiling_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.profiling_results, f, indent=2, default=str)
        logger.info(f"Profiling report saved to {report_path}")
    
    def save_cleansing_log(self):
        """Save cleansing operations log."""
        log_path = self.output_dir / "cleansing_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.cleansing_log, f, indent=2, default=str)
        logger.info(f"Cleansing log saved to {log_path}")
    
    def stage_dwelling_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage dwelling values dataset."""
        logger.info("Staging dwelling values data...")
        
        # Profile
        self.profile_dataframe(df, "dwelling_values")
        
        # Parse quarter to date
        df = self.parse_quarter_to_date(df, 'Quarter')
        
        # Handle missing values
        df = self.handle_missing_values(df, {
            'Total_Value_Dwelling_Stock_Billion_AUD': 'ffill'
        })
        
        # Add calculated columns
        df = self.add_calculated_columns(df)
        
        # Remove duplicates
        df = self.remove_duplicates(df, subset=['Quarter'])
        
        return df
    
    def stage_cash_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage RBA cash rate dataset."""
        logger.info("Staging cash rate data...")
        
        # Profile
        self.profile_dataframe(df, "cash_rate")
        
        # Standardise date format
        df = self.standardise_date_format(df, 'Effective_Date')
        
        # Sort by date
        df = df.sort_values('Effective_Date', ascending=False)
        
        # Remove duplicates
        df = self.remove_duplicates(df, subset=['Effective_Date'])
        
        return df
    
    def stage_median_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stage median prices dataset."""
        logger.info("Staging median prices data...")
        
        # Profile
        self.profile_dataframe(df, "median_prices")
        
        # Parse quarter to date
        df = self.parse_quarter_to_date(df, 'Quarter')
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Add calculated columns
        df = self.add_calculated_columns(df)
        
        return df


def main():
    """Main entry point for data staging."""
    
    # Initialize pipeline
    pipeline = DataStagingPipeline(
        input_dir=".",  # Current directory with CSV files
        output_dir="staged_data"
    )
    
    # Load and stage each dataset
    staged_data = {}
    
    # Stage dwelling values
    try:
        df = pd.read_csv("01_abs_total_value_dwellings.csv")
        staged_data['dwelling_values'] = pipeline.stage_dwelling_values(df)
        pipeline.export_to_excel_format(staged_data['dwelling_values'], 'dwelling_values')
    except FileNotFoundError:
        logger.warning("Dwelling values file not found")
    
    # Stage cash rate
    try:
        df = pd.read_csv("03_rba_cash_rate_history.csv")
        staged_data['cash_rate'] = pipeline.stage_cash_rate(df)
        pipeline.export_to_excel_format(staged_data['cash_rate'], 'cash_rate')
    except FileNotFoundError:
        logger.warning("Cash rate file not found")
    
    # Stage median prices
    try:
        df = pd.read_csv("07_capital_city_median_prices_quarterly.csv")
        staged_data['median_prices'] = pipeline.stage_median_prices(df)
        pipeline.export_to_excel_format(staged_data['median_prices'], 'median_prices')
    except FileNotFoundError:
        logger.warning("Median prices file not found")
    
    # Save reports
    pipeline.save_profiling_report()
    pipeline.save_cleansing_log()
    
    # Display summary
    print("\n" + "="*60)
    print("DATA STAGING SUMMARY")
    print("="*60)
    
    for name, df in staged_data.items():
        print(f"\n{name.upper()}")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Columns: {list(df.columns)}")
    
    print(f"\nCleansing operations logged: {len(pipeline.cleansing_log)}")
    
    return staged_data


if __name__ == "__main__":
    main()
