"""
Australian Housing Market Analytics - Data Transformation Script
=================================================================
Script 4 of 5: Transform (Python/SQL)

Purpose: Feature engineering including: rolling price averages, year-on-year 
         growth rates, affordability ratios (median price to median income), 
         and seasonality indicators.

Author: Cam Walker
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline for Australian housing market data.
    Creates derived metrics for analysis and modelling.
    """
    
    def __init__(self, output_dir: str = "transformed_data"):
        """Initialize the transformation pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Australian median household income by capital city (ABS 2024 estimates)
        self.median_income_by_city = {
            'Sydney': 115000,
            'Melbourne': 105000,
            'Brisbane': 100000,
            'Adelaide': 95000,
            'Perth': 105000,
            'Hobart': 90000,
            'Darwin': 110000,
            'Canberra': 130000,
            'National': 105000
        }
        
        # Seasonal patterns (based on historical data)
        self.seasonal_factors = {
            1: 0.95,   # January - Summer holidays, lower activity
            2: 0.97,   # February - Market warming up
            3: 1.02,   # March - Strong autumn market
            4: 1.03,   # April - Peak autumn
            5: 1.01,   # May - Autumn continuing
            6: 0.98,   # June - Winter slowdown begins
            7: 0.96,   # July - Mid-winter low
            8: 0.97,   # August - Winter continuing
            9: 1.01,   # September - Spring begins
            10: 1.04,  # October - Peak spring market
            11: 1.05,  # November - Strong spring
            12: 0.98   # December - Holiday slowdown
        }
    
    def calculate_rolling_averages(self, df: pd.DataFrame, 
                                    value_columns: List[str],
                                    windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        Calculate rolling averages for specified columns.
        
        Args:
            df: Input DataFrame
            value_columns: Columns to calculate rolling averages for
            windows: Rolling window sizes
            
        Returns:
            DataFrame with rolling average columns
        """
        df = df.copy()
        
        for col in value_columns:
            if col in df.columns:
                for window in windows:
                    new_col = f"{col}_MA{window}"
                    df[new_col] = df[col].rolling(window=window, min_periods=1).mean()
                    logger.info(f"Created {new_col}")
        
        return df
    
    def calculate_yoy_growth(self, df: pd.DataFrame, 
                              value_columns: List[str],
                              periods: int = 4) -> pd.DataFrame:
        """
        Calculate year-on-year growth rates.
        
        Args:
            df: Input DataFrame (quarterly or monthly data)
            value_columns: Columns to calculate YoY growth for
            periods: Number of periods for YoY (4 for quarterly, 12 for monthly)
            
        Returns:
            DataFrame with YoY growth columns
        """
        df = df.copy()
        
        for col in value_columns:
            if col in df.columns:
                new_col = f"{col}_YoY_Growth"
                df[new_col] = ((df[col] - df[col].shift(periods)) / 
                               df[col].shift(periods) * 100)
                logger.info(f"Created {new_col}")
        
        return df
    
    def calculate_mom_growth(self, df: pd.DataFrame,
                              value_columns: List[str]) -> pd.DataFrame:
        """
        Calculate month-on-month or quarter-on-quarter growth rates.
        
        Args:
            df: Input DataFrame
            value_columns: Columns to calculate growth for
            
        Returns:
            DataFrame with growth columns
        """
        df = df.copy()
        
        for col in value_columns:
            if col in df.columns:
                new_col = f"{col}_QoQ_Growth"
                df[new_col] = df[col].pct_change() * 100
                logger.info(f"Created {new_col}")
        
        return df
    
    def calculate_affordability_ratios(self, df: pd.DataFrame,
                                        price_column: str,
                                        city_column: str = None) -> pd.DataFrame:
        """
        Calculate housing affordability ratios.
        
        Price-to-Income Ratio = Median House Price / Median Household Income
        
        Args:
            df: Input DataFrame with price data
            price_column: Column containing median prices
            city_column: Column containing city names (optional)
            
        Returns:
            DataFrame with affordability metrics
        """
        df = df.copy()
        
        if city_column and city_column in df.columns:
            df['Median_Income'] = df[city_column].map(self.median_income_by_city)
            df['Median_Income'] = df['Median_Income'].fillna(self.median_income_by_city['National'])
        else:
            df['Median_Income'] = self.median_income_by_city['National']
        
        if price_column in df.columns:
            # Price to Income Ratio (prices in thousands)
            df['Price_to_Income_Ratio'] = (df[price_column] * 1000) / df['Median_Income']
            
            # Years to save for deposit (20% deposit, 30% savings rate)
            deposit_percent = 0.20
            savings_rate = 0.30
            df['Years_to_Save_Deposit'] = (
                (df[price_column] * 1000 * deposit_percent) / 
                (df['Median_Income'] * savings_rate)
            )
            
            # Monthly mortgage payment (30-year loan at 6% interest)
            loan_amount = df[price_column] * 1000 * 0.80
            interest_rate = 0.06 / 12
            num_payments = 360
            
            df['Est_Monthly_Mortgage'] = (
                loan_amount * 
                (interest_rate * (1 + interest_rate)**num_payments) /
                ((1 + interest_rate)**num_payments - 1)
            )
            
            df['Mortgage_to_Income_Percent'] = (
                df['Est_Monthly_Mortgage'] * 12 / df['Median_Income'] * 100
            )
            
            logger.info("Created affordability metrics")
        
        return df
    
    def add_seasonality_indicators(self, df: pd.DataFrame,
                                    date_column: str = None,
                                    month_column: str = None) -> pd.DataFrame:
        """
        Add seasonality indicators based on month.
        """
        df = df.copy()
        
        if date_column and date_column in df.columns:
            df['_temp_date'] = pd.to_datetime(df[date_column], errors='coerce')
            df['Month_Num'] = df['_temp_date'].dt.month
            df = df.drop('_temp_date', axis=1)
        elif month_column and month_column in df.columns:
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            df['Month_Num'] = df[month_column].str[:3].map(month_map)
        
        if 'Month_Num' in df.columns:
            df['Seasonal_Factor'] = df['Month_Num'].map(self.seasonal_factors)
            
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Summer'
                elif month in [3, 4, 5]:
                    return 'Autumn'
                elif month in [6, 7, 8]:
                    return 'Winter'
                else:
                    return 'Spring'
            
            df['Season'] = df['Month_Num'].apply(get_season)
            df['Market_Activity'] = df['Seasonal_Factor'].apply(
                lambda x: 'High' if x > 1.02 else ('Low' if x < 0.97 else 'Normal')
            )
            
            logger.info("Added seasonality indicators")
        
        return df
    
    def calculate_volatility_metrics(self, df: pd.DataFrame,
                                      value_columns: List[str],
                                      window: int = 12) -> pd.DataFrame:
        """Calculate volatility metrics (rolling standard deviation)."""
        df = df.copy()
        
        for col in value_columns:
            if col in df.columns:
                df[f"{col}_Volatility"] = df[col].rolling(window=window).std()
                rolling_mean = df[col].rolling(window=window).mean()
                df[f"{col}_CoV"] = df[f"{col}_Volatility"] / rolling_mean * 100
                logger.info(f"Created volatility metrics for {col}")
        
        return df
    
    def calculate_market_indicators(self, df: pd.DataFrame,
                                     price_column: str) -> pd.DataFrame:
        """Calculate market health indicators."""
        df = df.copy()
        
        if price_column in df.columns:
            df['Price_Momentum_3M'] = df[price_column].pct_change(3) * 100
            df['Price_Momentum_6M'] = df[price_column].pct_change(6) * 100
            df['Price_Acceleration'] = df['Price_Momentum_3M'].diff()
            
            def classify_market_phase(row):
                if pd.isna(row.get('Price_Momentum_3M')) or pd.isna(row.get('Price_Acceleration')):
                    return 'Unknown'
                momentum = row['Price_Momentum_3M']
                accel = row['Price_Acceleration']
                
                if momentum > 2 and accel > 0:
                    return 'Accelerating Growth'
                elif momentum > 2 and accel <= 0:
                    return 'Decelerating Growth'
                elif momentum < -2 and accel < 0:
                    return 'Accelerating Decline'
                elif momentum < -2 and accel >= 0:
                    return 'Decelerating Decline'
                else:
                    return 'Stable'
            
            df['Market_Phase'] = df.apply(classify_market_phase, axis=1)
            logger.info("Created market indicators")
        
        return df
    
    def calculate_comparative_metrics(self, df: pd.DataFrame,
                                       city_price_columns: List[str],
                                       national_column: str) -> pd.DataFrame:
        """Calculate city vs national comparison metrics."""
        df = df.copy()
        
        for city_col in city_price_columns:
            if city_col in df.columns and national_column in df.columns:
                city_name = city_col.replace('_Median_000', '').replace('_', ' ')
                df[f"{city_name}_vs_National_Percent"] = (
                    (df[city_col] - df[national_column]) / df[national_column] * 100
                )
        
        return df
    
    def create_lagged_features(self, df: pd.DataFrame,
                                value_columns: List[str],
                                lags: List[int] = [1, 2, 3, 4]) -> pd.DataFrame:
        """Create lagged features for time series modelling."""
        df = df.copy()
        
        for col in value_columns:
            if col in df.columns:
                for lag in lags:
                    df[f"{col}_Lag{lag}"] = df[col].shift(lag)
        
        logger.info(f"Created lagged features with lags: {lags}")
        return df
    
    def transform_prices_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations to prices dataset."""
        logger.info("Transforming prices dataset...")
        
        price_cols = [col for col in df.columns if 'Median' in col and '000' in col]
        
        df = self.calculate_rolling_averages(df, price_cols, windows=[2, 4])
        df = self.calculate_yoy_growth(df, price_cols, periods=4)
        df = self.calculate_mom_growth(df, price_cols)
        df = self.calculate_volatility_metrics(df, price_cols, window=8)
        
        if 'National_Median_000' in df.columns:
            df = self.calculate_market_indicators(df, 'National_Median_000')
            city_cols = [c for c in price_cols if c != 'National_Median_000']
            df = self.calculate_comparative_metrics(df, city_cols, 'National_Median_000')
        
        df = self.create_lagged_features(df, ['National_Median_000'], lags=[1, 2, 4])
        
        return df
    
    def transform_dwelling_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dwelling values dataset."""
        logger.info("Transforming dwelling values dataset...")
        
        value_col = 'Total_Value_Dwelling_Stock_Billion_AUD'
        
        if value_col in df.columns:
            df = self.calculate_rolling_averages(df, [value_col], windows=[2, 4])
            df = self.calculate_yoy_growth(df, [value_col], periods=4)
            df = self.calculate_mom_growth(df, [value_col])
            df = self.calculate_market_indicators(df, value_col)
        
        return df
    
    def transform_interest_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform interest rates dataset."""
        logger.info("Transforming interest rates dataset...")
        
        rate_col = 'Cash_Rate_Target_Percent'
        
        if rate_col in df.columns:
            def classify_rate_regime(rate):
                if rate < 1:
                    return 'Ultra Low'
                elif rate < 2:
                    return 'Low'
                elif rate < 4:
                    return 'Moderate'
                elif rate < 6:
                    return 'High'
                else:
                    return 'Very High'
            
            df['Rate_Regime'] = df[rate_col].apply(classify_rate_regime)
            df['Cumulative_Change'] = df[rate_col] - df[rate_col].iloc[-1]
            df['Rate_Direction'] = df['Change_Percent_Points'].apply(
                lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'Hold')
            )
        
        return df
    
    def export_transformed_data(self, df: pd.DataFrame, name: str):
        """Export transformed data to CSV and Excel."""
        csv_path = self.output_dir / f"{name}_transformed.csv"
        excel_path = self.output_dir / f"{name}_transformed.xlsx"
        
        df.to_csv(csv_path, index=False)
        df.to_excel(excel_path, index=False)
        
        logger.info(f"Exported transformed data to {csv_path}")


def main():
    """Main entry point for data transformation."""
    
    pipeline = FeatureEngineeringPipeline(output_dir="transformed_data")
    
    transformed_data = {}
    
    # Transform median prices
    try:
        df = pd.read_csv("07_capital_city_median_prices_quarterly.csv")
        transformed = pipeline.transform_prices_dataset(df)
        pipeline.export_transformed_data(transformed, "median_prices")
        transformed_data['median_prices'] = transformed
        print(f"\nMedian Prices: {len(transformed.columns)} columns after transformation")
    except FileNotFoundError:
        logger.warning("Median prices file not found")
    
    # Transform dwelling values
    try:
        df = pd.read_csv("01_abs_total_value_dwellings.csv")
        transformed = pipeline.transform_dwelling_values(df)
        pipeline.export_transformed_data(transformed, "dwelling_values")
        transformed_data['dwelling_values'] = transformed
        print(f"Dwelling Values: {len(transformed.columns)} columns after transformation")
    except FileNotFoundError:
        logger.warning("Dwelling values file not found")
    
    # Transform interest rates
    try:
        df = pd.read_csv("03_rba_cash_rate_history.csv")
        transformed = pipeline.transform_interest_rates(df)
        pipeline.export_transformed_data(transformed, "interest_rates")
        transformed_data['interest_rates'] = transformed
        print(f"Interest Rates: {len(transformed.columns)} columns after transformation")
    except FileNotFoundError:
        logger.warning("Interest rates file not found")
    
    # Display summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    for name, df in transformed_data.items():
        print(f"\n{name.upper()}")
        print(f"  Original + New Columns: {len(df.columns)}")
        print(f"  Sample new features:")
        new_cols = [c for c in df.columns if any(x in c for x in ['_MA', '_YoY', '_QoQ', 'Momentum', 'Phase', 'Regime'])]
        for col in new_cols[:5]:
            print(f"    - {col}")
    
    return transformed_data


if __name__ == "__main__":
    main()
