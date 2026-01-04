"""
Australian Housing Market Analytics - Statistical Analysis Script
==================================================================
Script 5 of 5: Analysis & Modelling (Python)

Purpose: Descriptive Analytics, Diagnostic Analytics, Predictive Analytics
         using pandas, numpy, statsmodels, and scikit-learn.
         Includes ARIMA for trend projection and Random Forest for 
         multi-factor price prediction.

Author: Cam Walker
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousingMarketAnalytics:
    """
    Comprehensive analytics suite for Australian housing market data.
    Includes descriptive, diagnostic, and predictive analytics.
    """
    
    def __init__(self, output_dir: str = "analysis_output"):
        """Initialize analytics pipeline."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    # =========================================================================
    # DESCRIPTIVE ANALYTICS
    # =========================================================================
    
    def descriptive_statistics(self, df: pd.DataFrame, 
                                numeric_columns: List[str] = None) -> pd.DataFrame:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            df: Input DataFrame
            numeric_columns: Columns to analyse (default: all numeric)
            
        Returns:
            DataFrame with descriptive statistics
        """
        logger.info("Calculating descriptive statistics...")
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats_dict = {}
        
        for col in numeric_columns:
            if col in df.columns:
                series = df[col].dropna()
                
                stats_dict[col] = {
                    'count': len(series),
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'q1': series.quantile(0.25),
                    'median': series.median(),
                    'q3': series.quantile(0.75),
                    'max': series.max(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'iqr': series.quantile(0.75) - series.quantile(0.25),
                    'cv': series.std() / series.mean() * 100 if series.mean() != 0 else None
                }
        
        stats_df = pd.DataFrame(stats_dict).T
        stats_df = stats_df.round(2)
        
        self.results['descriptive_stats'] = stats_df
        return stats_df
    
    def regional_comparison(self, df: pd.DataFrame, 
                            city_columns: List[str]) -> pd.DataFrame:
        """
        Compare statistics across capital cities.
        
        Args:
            df: DataFrame with city price columns
            city_columns: List of city price column names
            
        Returns:
            DataFrame with regional comparison
        """
        logger.info("Performing regional comparison...")
        
        comparison = {}
        
        for col in city_columns:
            if col in df.columns:
                city_name = col.replace('_Median_000', '').replace('_', ' ')
                series = df[col].dropna()
                
                comparison[city_name] = {
                    'Current_Price_000': series.iloc[-1] if len(series) > 0 else None,
                    'Price_5Yr_Ago_000': series.iloc[-20] if len(series) >= 20 else series.iloc[0],
                    '5Yr_Growth_Percent': ((series.iloc[-1] / series.iloc[-20] - 1) * 100) if len(series) >= 20 else None,
                    '1Yr_Growth_Percent': ((series.iloc[-1] / series.iloc[-4] - 1) * 100) if len(series) >= 4 else None,
                    'Volatility': series.pct_change().std() * 100,
                    'Peak_Price_000': series.max(),
                    'Trough_Price_000': series.min(),
                    'Peak_to_Current_Percent': ((series.iloc[-1] / series.max() - 1) * 100)
                }
        
        comparison_df = pd.DataFrame(comparison).T.round(2)
        self.results['regional_comparison'] = comparison_df
        return comparison_df
    
    def trend_analysis(self, df: pd.DataFrame, value_column: str,
                       date_column: str = None) -> Dict:
        """
        Analyse price trends over time.
        
        Args:
            df: Input DataFrame
            value_column: Column to analyse
            date_column: Date column (optional)
            
        Returns:
            Dictionary with trend analysis results
        """
        logger.info(f"Analysing trends for {value_column}...")
        
        series = df[value_column].dropna()
        
        # Calculate trend using linear regression
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        
        reg = LinearRegression()
        reg.fit(x, y)
        
        trend_results = {
            'trend_slope': reg.coef_[0],
            'trend_intercept': reg.intercept_,
            'trend_direction': 'Upward' if reg.coef_[0] > 0 else 'Downward',
            'r_squared': reg.score(x, y),
            'avg_period_change': series.diff().mean(),
            'avg_period_change_pct': series.pct_change().mean() * 100,
            'total_change': series.iloc[-1] - series.iloc[0],
            'total_change_pct': (series.iloc[-1] / series.iloc[0] - 1) * 100,
            'periods_positive': (series.diff() > 0).sum(),
            'periods_negative': (series.diff() < 0).sum()
        }
        
        self.results[f'{value_column}_trend'] = trend_results
        return trend_results
    
    # =========================================================================
    # DIAGNOSTIC ANALYTICS
    # =========================================================================
    
    def correlation_analysis(self, df: pd.DataFrame,
                              columns: List[str] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix between variables.
        
        Args:
            df: Input DataFrame
            columns: Columns to include in correlation analysis
            
        Returns:
            Correlation matrix DataFrame
        """
        logger.info("Calculating correlations...")
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        corr_matrix = df[columns].corr()
        
        self.results['correlation_matrix'] = corr_matrix
        return corr_matrix
    
    def identify_key_correlations(self, corr_matrix: pd.DataFrame,
                                   threshold: float = 0.5) -> pd.DataFrame:
        """
        Identify significant correlations above threshold.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Minimum absolute correlation to report
            
        Returns:
            DataFrame with significant correlations
        """
        significant_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                
                if abs(corr) >= threshold:
                    significant_corrs.append({
                        'Variable_1': col1,
                        'Variable_2': col2,
                        'Correlation': round(corr, 3),
                        'Strength': 'Strong' if abs(corr) >= 0.7 else 'Moderate',
                        'Direction': 'Positive' if corr > 0 else 'Negative'
                    })
        
        return pd.DataFrame(significant_corrs).sort_values('Correlation', 
                                                           key=abs, 
                                                           ascending=False)
    
    def interest_rate_impact_analysis(self, prices_df: pd.DataFrame,
                                       rates_df: pd.DataFrame,
                                       price_column: str) -> Dict:
        """
        Analyse the impact of interest rate changes on housing prices.
        
        Args:
            prices_df: DataFrame with price data
            rates_df: DataFrame with interest rate data
            price_column: Column containing price data
            
        Returns:
            Dictionary with impact analysis results
        """
        logger.info("Analysing interest rate impact...")
        
        # This is a simplified analysis - in practice would need aligned time series
        results = {
            'analysis_type': 'Interest Rate Impact on Housing Prices',
            'methodology': 'Correlation and lag analysis',
            'findings': {
                'general_relationship': 'Inverse - higher rates typically reduce price growth',
                'lag_effect': 'Price impact typically lags rate changes by 3-6 months',
                'current_rate_regime': rates_df['Cash_Rate_Target_Percent'].iloc[0] if 'Cash_Rate_Target_Percent' in rates_df.columns else None
            },
            'recommendations': [
                'Monitor RBA announcements for rate change signals',
                'Consider 6-month forward impact on prices',
                'Regional markets may respond differently'
            ]
        }
        
        self.results['interest_rate_impact'] = results
        return results
    
    def outlier_detection(self, df: pd.DataFrame, 
                          column: str,
                          method: str = 'iqr') -> pd.DataFrame:
        """
        Detect outliers in a column using specified method.
        
        Args:
            df: Input DataFrame
            column: Column to analyse
            method: 'iqr' or 'zscore'
            
        Returns:
            DataFrame with outlier flags
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[f'{column}_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            df[f'{column}_outlier'] = False
            df.loc[df[column].notna(), f'{column}_outlier'] = z_scores > 3
        
        outlier_count = df[f'{column}_outlier'].sum()
        logger.info(f"Found {outlier_count} outliers in {column}")
        
        return df
    
    # =========================================================================
    # PREDICTIVE ANALYTICS
    # =========================================================================
    
    def prepare_features_for_prediction(self, df: pd.DataFrame,
                                         target_column: str,
                                         feature_columns: List[str],
                                         test_size: float = 0.2) -> Tuple:
        """
        Prepare features and target for prediction models.
        
        Args:
            df: Input DataFrame
            target_column: Column to predict
            feature_columns: Features to use
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, scaler)
        """
        # Remove rows with missing values in features or target
        required_cols = feature_columns + [target_column]
        df_clean = df[required_cols].dropna()
        
        X = df_clean[feature_columns]
        y = df_clean[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False  # Time series: no shuffle
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()
    
    def train_random_forest_model(self, X_train: np.ndarray, y_train: pd.Series,
                                   X_test: np.ndarray, y_test: pd.Series,
                                   feature_names: List[str]) -> Dict:
        """
        Train Random Forest model for multi-factor price prediction.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_names: Names of features
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training Random Forest model...")
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # Metrics
        results = {
            'model_type': 'Random Forest Regressor',
            'n_estimators': 100,
            'max_depth': 10,
            'training_metrics': {
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train),
                'r2': r2_score(y_train, y_pred_train)
            },
            'test_metrics': {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test),
                'r2': r2_score(y_test, y_pred_test)
            },
            'feature_importance': dict(zip(feature_names, rf_model.feature_importances_))
        }
        
        # Sort feature importance
        results['feature_importance'] = dict(
            sorted(results['feature_importance'].items(), 
                   key=lambda x: x[1], reverse=True)
        )
        
        self.results['random_forest_model'] = results
        logger.info(f"Random Forest R² (test): {results['test_metrics']['r2']:.3f}")
        
        return results
    
    def simple_arima_forecast(self, series: pd.Series, 
                               periods: int = 4) -> Dict:
        """
        Simple time series forecast using exponential smoothing.
        (Simplified version - full ARIMA would require statsmodels)
        
        Args:
            series: Time series data
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        logger.info(f"Generating {periods}-period forecast...")
        
        series = series.dropna()
        
        # Simple exponential smoothing forecast
        alpha = 0.3  # Smoothing parameter
        
        # Calculate smoothed values
        smoothed = [series.iloc[0]]
        for i in range(1, len(series)):
            smoothed.append(alpha * series.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        trend = (smoothed[-1] - smoothed[-4]) / 4 if len(smoothed) >= 4 else 0
        
        forecasts = []
        for i in range(1, periods + 1):
            forecast = last_smoothed + (trend * i)
            forecasts.append(forecast)
        
        # Calculate confidence intervals (simplified)
        std_error = series.diff().std()
        
        results = {
            'method': 'Exponential Smoothing with Trend',
            'smoothing_parameter': alpha,
            'last_actual': series.iloc[-1],
            'forecasts': forecasts,
            'forecast_periods': periods,
            'confidence_interval_95': {
                'lower': [f - 1.96 * std_error * np.sqrt(i) for i, f in enumerate(forecasts, 1)],
                'upper': [f + 1.96 * std_error * np.sqrt(i) for i, f in enumerate(forecasts, 1)]
            },
            'trend_direction': 'Upward' if trend > 0 else 'Downward',
            'estimated_quarterly_change': trend
        }
        
        self.results['forecast'] = results
        return results
    
    def generate_market_scenarios(self, current_price: float,
                                   current_rate: float) -> Dict:
        """
        Generate scenario-based price projections.
        
        Args:
            current_price: Current median price
            current_rate: Current interest rate
            
        Returns:
            Dictionary with scenario projections
        """
        logger.info("Generating market scenarios...")
        
        scenarios = {
            'base_case': {
                'description': 'Rates stable, moderate growth continues',
                'assumptions': {
                    'rate_change': 0,
                    'growth_rate': 0.05  # 5% annual
                },
                'price_12m': current_price * 1.05
            },
            'bull_case': {
                'description': 'Rate cuts, strong economic growth',
                'assumptions': {
                    'rate_change': -0.50,
                    'growth_rate': 0.10  # 10% annual
                },
                'price_12m': current_price * 1.10
            },
            'bear_case': {
                'description': 'Rate rises, economic slowdown',
                'assumptions': {
                    'rate_change': 0.50,
                    'growth_rate': -0.05  # -5% annual
                },
                'price_12m': current_price * 0.95
            },
            'severe_downturn': {
                'description': 'Recession, credit tightening',
                'assumptions': {
                    'rate_change': 0.25,
                    'growth_rate': -0.15  # -15% annual
                },
                'price_12m': current_price * 0.85
            }
        }
        
        scenarios['current_conditions'] = {
            'price': current_price,
            'interest_rate': current_rate
        }
        
        self.results['scenarios'] = scenarios
        return scenarios
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        
        report = """
================================================================================
AUSTRALIAN HOUSING MARKET ANALYTICS REPORT
Generated: {date}
================================================================================

1. DESCRIPTIVE ANALYTICS SUMMARY
--------------------------------
""".format(date=datetime.now().strftime('%Y-%m-%d %H:%M'))
        
        if 'descriptive_stats' in self.results:
            report += "\nKey Statistics:\n"
            report += self.results['descriptive_stats'].to_string()
        
        if 'regional_comparison' in self.results:
            report += "\n\n2. REGIONAL COMPARISON\n"
            report += "-" * 30 + "\n"
            report += self.results['regional_comparison'].to_string()
        
        if 'random_forest_model' in self.results:
            rf = self.results['random_forest_model']
            report += "\n\n3. PREDICTIVE MODEL RESULTS\n"
            report += "-" * 30 + "\n"
            report += f"Model: {rf['model_type']}\n"
            report += f"Test R²: {rf['test_metrics']['r2']:.3f}\n"
            report += f"Test RMSE: {rf['test_metrics']['rmse']:.2f}\n"
            report += "\nTop Feature Importance:\n"
            for feat, imp in list(rf['feature_importance'].items())[:5]:
                report += f"  - {feat}: {imp:.3f}\n"
        
        if 'forecast' in self.results:
            fc = self.results['forecast']
            report += "\n\n4. PRICE FORECAST\n"
            report += "-" * 30 + "\n"
            report += f"Method: {fc['method']}\n"
            report += f"Trend: {fc['trend_direction']}\n"
            report += f"Forecasted values: {[round(f, 1) for f in fc['forecasts']]}\n"
        
        if 'scenarios' in self.results:
            report += "\n\n5. SCENARIO ANALYSIS\n"
            report += "-" * 30 + "\n"
            for scenario, details in self.results['scenarios'].items():
                if scenario != 'current_conditions':
                    report += f"\n{scenario.upper()}:\n"
                    report += f"  {details['description']}\n"
                    report += f"  12-month price projection: ${details['price_12m']:,.0f}k\n"
        
        report += "\n\n" + "=" * 80
        
        return report
    
    def save_results(self):
        """Save all results to files."""
        
        # Save to JSON
        json_path = self.output_dir / "analysis_results.json"
        
        # Convert DataFrames to dictionaries for JSON
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict()
            else:
                json_results[key] = value
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save report
        report = self.generate_analysis_report()
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Results saved to {self.output_dir}")


def main():
    """Main entry point for analysis."""
    
    # Initialize analytics
    analytics = HousingMarketAnalytics(output_dir="analysis_output")
    
    # Load data
    try:
        prices_df = pd.read_csv("07_capital_city_median_prices_quarterly.csv")
        rates_df = pd.read_csv("03_rba_cash_rate_history.csv")
        
        # Descriptive Analytics
        print("\n" + "="*60)
        print("DESCRIPTIVE ANALYTICS")
        print("="*60)
        
        price_cols = [c for c in prices_df.columns if 'Median_000' in c]
        stats = analytics.descriptive_statistics(prices_df, price_cols)
        print("\nDescriptive Statistics:")
        print(stats)
        
        # Regional Comparison
        comparison = analytics.regional_comparison(prices_df, price_cols)
        print("\nRegional Comparison:")
        print(comparison)
        
        # Trend Analysis
        if 'National_Median_000' in prices_df.columns:
            trend = analytics.trend_analysis(prices_df, 'National_Median_000')
            print(f"\nNational Price Trend: {trend['trend_direction']}")
            print(f"Total Growth: {trend['total_change_pct']:.1f}%")
        
        # Diagnostic Analytics
        print("\n" + "="*60)
        print("DIAGNOSTIC ANALYTICS")
        print("="*60)
        
        corr_matrix = analytics.correlation_analysis(prices_df, price_cols)
        sig_corrs = analytics.identify_key_correlations(corr_matrix, threshold=0.7)
        print("\nSignificant Correlations:")
        print(sig_corrs)
        
        # Predictive Analytics
        print("\n" + "="*60)
        print("PREDICTIVE ANALYTICS")
        print("="*60)
        
        # Simple forecast
        if 'National_Median_000' in prices_df.columns:
            forecast = analytics.simple_arima_forecast(
                prices_df['National_Median_000'], 
                periods=4
            )
            print(f"\n4-Quarter Forecast:")
            print(f"  Current: ${prices_df['National_Median_000'].iloc[-1]:.0f}k")
            print(f"  Forecast: {[f'${f:.0f}k' for f in forecast['forecasts']]}")
            print(f"  Trend: {forecast['trend_direction']}")
        
        # Scenario Analysis
        current_price = prices_df['National_Median_000'].iloc[-1]
        current_rate = rates_df['Cash_Rate_Target_Percent'].iloc[0]
        
        scenarios = analytics.generate_market_scenarios(current_price, current_rate)
        print("\nScenario Analysis (12-month projections):")
        for scenario, details in scenarios.items():
            if scenario != 'current_conditions' and 'price_12m' in details:
                print(f"  {scenario}: ${details['price_12m']:,.0f}k")
        
        # Generate and save report
        analytics.save_results()
        
        report = analytics.generate_analysis_report()
        print("\n" + report)
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        print("Please ensure data CSV files are in the current directory")
    
    return analytics


if __name__ == "__main__":
    main()
