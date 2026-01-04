# Australian Housing Market Analytics - Data Package

## Project Overview
This data package contains Australian housing market data sourced from official government repositories for use in the Data Science Portfolio Project: **Australian Housing Market Analytics**.

## Data Sources
All data has been sourced from authoritative Australian government and industry sources:

1. **Australian Bureau of Statistics (ABS)** - data.gov.au / abs.gov.au
   - Total Value of Dwellings
   - Mean Dwelling Prices by State/Territory
   - Regional Population Statistics
   - Building/Dwelling Approvals

2. **Reserve Bank of Australia (RBA)** - rba.gov.au
   - Cash Rate Target History
   - Monetary Policy Decisions

3. **CoreLogic/Cotality** - corelogic.com.au
   - Home Value Index (Monthly Changes)
   - Capital City Price Trends

---

## File Descriptions

### 00_data_sources_documentation.csv
Metadata file documenting all data sources, URLs, descriptions, and reference periods.

### 01_abs_total_value_dwellings.csv
**Source:** ABS Total Value of Dwellings Release  
**Fields:**
- `Quarter` - Reference quarter (e.g., Sep-2020)
- `Total_Value_Dwelling_Stock_Billion_AUD` - Total value in billions of AUD
- `Year` - Calendar year

**Coverage:** Sep 2020 - Sep 2025 (Quarterly)

### 02_abs_mean_dwelling_prices_by_state.csv
**Source:** ABS Total Value of Dwellings Release  
**Fields:**
- `State_Territory` - Australian state/territory code
- `Mar_2025_Mean_Price_000` - Mean price in thousands AUD
- `Jun_2025_Mean_Price_000` - Mean price in thousands AUD
- `Sep_2025_Mean_Price_000` - Mean price in thousands AUD

**Coverage:** Most recent three quarters

### 03_rba_cash_rate_history.csv
**Source:** RBA Cash Rate Target Statistics  
**Fields:**
- `Effective_Date` - Date rate change took effect
- `Change_Percent_Points` - Rate change in percentage points
- `Cash_Rate_Target_Percent` - New cash rate target

**Coverage:** Feb 2015 - Dec 2025

### 04_abs_capital_city_population.csv
**Source:** ABS Regional Population Release  
**Fields:**
- `Capital_City` - Capital city name
- `Population_2021_22` - Population at end of FY 2021-22
- `Population_Growth_2021_22` - Annual growth (persons)
- `Population_2022_23` - Population at end of FY 2022-23
- `Population_Growth_2022_23` - Annual growth (persons)
- `Population_2023_24` - Population at end of FY 2023-24
- `Population_Growth_2023_24` - Annual growth (persons)
- `Growth_Rate_2023_24_Percent` - Growth rate percentage

**Coverage:** FY 2021-22 to FY 2023-24

### 05_abs_dwelling_approvals_monthly.csv
**Source:** ABS Building Approvals Release  
**Fields:**
- `Month` - Reference month
- `Year` - Calendar year
- `Total_Dwellings_Approved` - Total dwelling units approved
- `Private_Sector_Houses` - Private sector house approvals
- `Private_Sector_Other_Dwellings` - Private sector other dwellings
- `Total_Residential_Value_Billion` - Total value in billions AUD
- `Seasonally_Adjusted` - Indicates seasonally adjusted data

**Coverage:** Dec 2020 - Oct 2025 (Monthly)

### 06_corelogic_home_value_index.csv
**Source:** CoreLogic/Cotality Home Value Index  
**Fields:**
- `Month`, `Year` - Reference period
- `National_MoM_Percent` - National month-on-month change
- `[City]_MoM_Percent` - City-specific month-on-month change
- `National_Annual_Percent` - National annual change

**Coverage:** Jan 2023 - Nov 2025 (Monthly)

### 07_capital_city_median_prices_quarterly.csv
**Source:** ABS and CoreLogic  
**Fields:**
- `Quarter`, `Year` - Reference period
- `[City]_Median_000` - Median house price in thousands AUD
- `National_Median_000` - National median price

**Coverage:** Q1 2015 - Q3 2025 (Quarterly)

### 08_housing_affordability_metrics.csv
**Source:** ABS and CoreLogic  
**Fields:**
- `City` - Capital city name
- `Median_House_Price_2025_000` - Current median price (thousands)
- `Median_Household_Income_000` - Estimated median household income
- `Price_to_Income_Ratio` - Affordability ratio
- `Annual_Price_Growth_Percent` - 12-month price growth
- `Rental_Yield_Percent` - Gross rental yield
- `Days_on_Market_Median` - Median days to sell

**Coverage:** 2025 (Point-in-time)

---

## Data Quality Notes

1. **ABS Data**: Official government statistics, subject to standard revisions. Seasonally adjusted where noted.

2. **RBA Data**: Official monetary policy decisions, no revisions.

3. **CoreLogic Data**: Based on hedonic valuation methodology. Monthly data released on last business day of each month.

4. **Population Data**: Annual estimates based on census benchmarks with quarterly updates.

---

## Suggested Analyses

- **Time Series Analysis**: Price trends over 10+ years by capital city
- **Correlation Analysis**: Interest rates vs. housing prices
- **Affordability Analysis**: Price-to-income ratios across cities
- **Supply-Demand**: Building approvals vs. population growth
- **Market Cycle Analysis**: Identifying peaks and troughs

---

## License and Attribution

Data sourced from Australian Government open data initiatives:
- ABS: Creative Commons Attribution 4.0 International (CC BY 4.0)
- RBA: Crown Copyright, available for public use with attribution
- CoreLogic: Publicly available summary data

**Citation:**
Australian Bureau of Statistics, Reserve Bank of Australia, CoreLogic/Cotality. Australian Housing Market Data 2015-2025.

---

## Contact
For questions about this data package or the Australian Housing Market Analytics project, please refer to the project specification document.

*Data compiled: January 2026*
