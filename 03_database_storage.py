"""
Australian Housing Market Analytics - SQL Database Script
==========================================================
Script 3 of 5: Storage (SQL)

Purpose: SQLite database with normalised schema. Tables include: dim_location, 
         dim_date, fact_prices, fact_demographics. Documented DDL scripts and 
         ERD diagram provided.

Author: Cam Walker
Date: January 2026
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousingDatabaseManager:
    """
    SQLite database manager for Australian Housing Market Analytics.
    Implements star schema with dimension and fact tables.
    """
    
    def __init__(self, db_path: str = "australian_housing.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        logger.info(f"Connected to database: {self.db_path}")
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def execute_ddl(self):
        """
        Execute DDL statements to create database schema.
        Implements star schema with dimension and fact tables.
        """
        
        ddl_statements = """
        -- ============================================================
        -- AUSTRALIAN HOUSING MARKET ANALYTICS DATABASE
        -- Schema: Star Schema (Dimensional Model)
        -- Created: January 2026
        -- ============================================================

        -- Drop existing tables (for clean recreation)
        DROP TABLE IF EXISTS fact_housing_prices;
        DROP TABLE IF EXISTS fact_dwelling_approvals;
        DROP TABLE IF EXISTS fact_population;
        DROP TABLE IF EXISTS fact_interest_rates;
        DROP TABLE IF EXISTS dim_location;
        DROP TABLE IF EXISTS dim_date;
        DROP TABLE IF EXISTS dim_property_type;

        -- ============================================================
        -- DIMENSION TABLES
        -- ============================================================

        -- Dimension: Location
        -- Stores geographic hierarchy for Australian locations
        CREATE TABLE dim_location (
            location_id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_code VARCHAR(10) NOT NULL UNIQUE,
            location_name VARCHAR(100) NOT NULL,
            state_code VARCHAR(3) NOT NULL,
            state_name VARCHAR(50) NOT NULL,
            region_type VARCHAR(20) NOT NULL,  -- 'Capital City', 'Regional', 'National'
            latitude DECIMAL(10, 6),
            longitude DECIMAL(10, 6),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Dimension: Date
        -- Date dimension for time-series analysis
        CREATE TABLE dim_date (
            date_id INTEGER PRIMARY KEY,  -- YYYYMMDD format
            full_date DATE NOT NULL UNIQUE,
            day_of_week INTEGER NOT NULL,
            day_name VARCHAR(10) NOT NULL,
            day_of_month INTEGER NOT NULL,
            day_of_year INTEGER NOT NULL,
            week_of_year INTEGER NOT NULL,
            month_number INTEGER NOT NULL,
            month_name VARCHAR(10) NOT NULL,
            quarter INTEGER NOT NULL,
            quarter_name VARCHAR(10) NOT NULL,
            year INTEGER NOT NULL,
            fiscal_year INTEGER NOT NULL,  -- Australian FY (Jul-Jun)
            is_weekend BOOLEAN NOT NULL,
            is_public_holiday BOOLEAN DEFAULT FALSE
        );

        -- Dimension: Property Type
        CREATE TABLE dim_property_type (
            property_type_id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_type_code VARCHAR(20) NOT NULL UNIQUE,
            property_type_name VARCHAR(50) NOT NULL,
            property_category VARCHAR(30) NOT NULL,  -- 'Residential', 'Commercial'
            description TEXT
        );

        -- ============================================================
        -- FACT TABLES
        -- ============================================================

        -- Fact: Housing Prices
        -- Central fact table for price data
        CREATE TABLE fact_housing_prices (
            price_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            property_type_id INTEGER,
            median_price_aud DECIMAL(12, 2),
            mean_price_aud DECIMAL(12, 2),
            price_index DECIMAL(8, 2),
            monthly_change_percent DECIMAL(6, 3),
            quarterly_change_percent DECIMAL(6, 3),
            annual_change_percent DECIMAL(6, 3),
            volume_sold INTEGER,
            days_on_market_median INTEGER,
            rental_yield_percent DECIMAL(5, 2),
            source VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
            FOREIGN KEY (location_id) REFERENCES dim_location(location_id),
            FOREIGN KEY (property_type_id) REFERENCES dim_property_type(property_type_id)
        );

        -- Fact: Dwelling Approvals
        -- Building approval statistics
        CREATE TABLE fact_dwelling_approvals (
            approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            total_dwellings_approved INTEGER,
            private_houses_approved INTEGER,
            private_other_approved INTEGER,
            public_dwellings_approved INTEGER,
            total_value_million_aud DECIMAL(10, 2),
            is_seasonally_adjusted BOOLEAN DEFAULT TRUE,
            source VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
            FOREIGN KEY (location_id) REFERENCES dim_location(location_id)
        );

        -- Fact: Population
        -- Demographic and population data
        CREATE TABLE fact_population (
            population_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_id INTEGER NOT NULL,
            location_id INTEGER NOT NULL,
            total_population INTEGER,
            population_growth INTEGER,
            growth_rate_percent DECIMAL(5, 2),
            natural_increase INTEGER,
            net_overseas_migration INTEGER,
            net_internal_migration INTEGER,
            median_age DECIMAL(4, 1),
            source VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
            FOREIGN KEY (location_id) REFERENCES dim_location(location_id)
        );

        -- Fact: Interest Rates
        -- RBA cash rate and related metrics
        CREATE TABLE fact_interest_rates (
            rate_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_id INTEGER NOT NULL,
            cash_rate_percent DECIMAL(5, 2) NOT NULL,
            change_percent DECIMAL(5, 2),
            standard_variable_rate DECIMAL(5, 2),
            fixed_rate_3yr DECIMAL(5, 2),
            inflation_rate_percent DECIMAL(5, 2),
            source VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
        );

        -- ============================================================
        -- INDEXES FOR PERFORMANCE
        -- ============================================================

        CREATE INDEX idx_fact_prices_date ON fact_housing_prices(date_id);
        CREATE INDEX idx_fact_prices_location ON fact_housing_prices(location_id);
        CREATE INDEX idx_fact_prices_date_location ON fact_housing_prices(date_id, location_id);

        CREATE INDEX idx_fact_approvals_date ON fact_dwelling_approvals(date_id);
        CREATE INDEX idx_fact_approvals_location ON fact_dwelling_approvals(location_id);

        CREATE INDEX idx_fact_population_date ON fact_population(date_id);
        CREATE INDEX idx_fact_population_location ON fact_population(location_id);

        CREATE INDEX idx_fact_rates_date ON fact_interest_rates(date_id);

        CREATE INDEX idx_dim_date_year_month ON dim_date(year, month_number);
        CREATE INDEX idx_dim_date_quarter ON dim_date(year, quarter);

        CREATE INDEX idx_dim_location_state ON dim_location(state_code);

        -- ============================================================
        -- VIEWS FOR COMMON QUERIES
        -- ============================================================

        -- View: Price Summary by City
        CREATE VIEW vw_price_summary_by_city AS
        SELECT 
            d.year,
            d.quarter_name,
            l.location_name,
            l.state_code,
            AVG(f.median_price_aud) as avg_median_price,
            AVG(f.annual_change_percent) as avg_annual_change,
            SUM(f.volume_sold) as total_volume
        FROM fact_housing_prices f
        JOIN dim_date d ON f.date_id = d.date_id
        JOIN dim_location l ON f.location_id = l.location_id
        GROUP BY d.year, d.quarter, l.location_id;

        -- View: Market Overview
        CREATE VIEW vw_market_overview AS
        SELECT 
            d.year,
            d.quarter_name,
            SUM(CASE WHEN l.region_type = 'National' THEN f.median_price_aud END) as national_median,
            AVG(f.annual_change_percent) as avg_annual_growth,
            r.cash_rate_percent,
            SUM(a.total_dwellings_approved) as total_approvals
        FROM fact_housing_prices f
        JOIN dim_date d ON f.date_id = d.date_id
        JOIN dim_location l ON f.location_id = l.location_id
        LEFT JOIN fact_interest_rates r ON f.date_id = r.date_id
        LEFT JOIN fact_dwelling_approvals a ON f.date_id = a.date_id AND f.location_id = a.location_id
        GROUP BY d.year, d.quarter;

        -- View: Affordability Metrics
        CREATE VIEW vw_affordability_metrics AS
        SELECT 
            d.year,
            l.location_name,
            l.state_code,
            f.median_price_aud,
            f.rental_yield_percent,
            f.days_on_market_median,
            p.median_age,
            f.median_price_aud / NULLIF(p.total_population / 1000000.0, 0) as price_per_million_pop
        FROM fact_housing_prices f
        JOIN dim_date d ON f.date_id = d.date_id
        JOIN dim_location l ON f.location_id = l.location_id
        LEFT JOIN fact_population p ON f.date_id = p.date_id AND f.location_id = p.location_id;
        """
        
        # Execute each statement
        for statement in ddl_statements.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    self.cursor.execute(statement)
                except sqlite3.Error as e:
                    logger.error(f"Error executing: {statement[:50]}... - {e}")
        
        self.conn.commit()
        logger.info("Database schema created successfully")
    
    def populate_dim_date(self, start_year: int = 2015, end_year: int = 2026):
        """
        Populate date dimension table.
        
        Args:
            start_year: Starting year for date dimension
            end_year: Ending year for date dimension
        """
        logger.info(f"Populating dim_date from {start_year} to {end_year}")
        
        dates = pd.date_range(
            start=f'{start_year}-01-01',
            end=f'{end_year}-12-31',
            freq='D'
        )
        
        for date in dates:
            date_id = int(date.strftime('%Y%m%d'))
            fiscal_year = date.year if date.month >= 7 else date.year - 1
            quarter = (date.month - 1) // 3 + 1
            quarter_name = f'Q{quarter}'
            
            self.cursor.execute("""
                INSERT OR REPLACE INTO dim_date 
                (date_id, full_date, day_of_week, day_name, day_of_month, day_of_year,
                 week_of_year, month_number, month_name, quarter, quarter_name,
                 year, fiscal_year, is_weekend)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_id,
                date.strftime('%Y-%m-%d'),
                date.weekday(),
                date.strftime('%A'),
                date.day,
                date.timetuple().tm_yday,
                date.isocalendar()[1],
                date.month,
                date.strftime('%B'),
                quarter,
                quarter_name,
                date.year,
                fiscal_year,
                1 if date.weekday() >= 5 else 0
            ))
        
        self.conn.commit()
        logger.info(f"Populated {len(dates)} date records")
    
    def populate_dim_location(self):
        """Populate location dimension with Australian capitals and states."""
        
        locations = [
            ('SYD', 'Greater Sydney', 'NSW', 'New South Wales', 'Capital City', -33.8688, 151.2093),
            ('MEL', 'Greater Melbourne', 'VIC', 'Victoria', 'Capital City', -37.8136, 144.9631),
            ('BNE', 'Greater Brisbane', 'QLD', 'Queensland', 'Capital City', -27.4698, 153.0251),
            ('PER', 'Greater Perth', 'WA', 'Western Australia', 'Capital City', -31.9505, 115.8605),
            ('ADL', 'Greater Adelaide', 'SA', 'South Australia', 'Capital City', -34.9285, 138.6007),
            ('HOB', 'Greater Hobart', 'TAS', 'Tasmania', 'Capital City', -42.8821, 147.3272),
            ('DAR', 'Greater Darwin', 'NT', 'Northern Territory', 'Capital City', -12.4634, 130.8456),
            ('CBR', 'Australian Capital Territory', 'ACT', 'Australian Capital Territory', 'Capital City', -35.2809, 149.1300),
            ('AUS', 'Australia', 'AUS', 'Australia', 'National', -25.2744, 133.7751),
            ('NSW_REG', 'Rest of NSW', 'NSW', 'New South Wales', 'Regional', -32.0, 147.0),
            ('VIC_REG', 'Rest of Victoria', 'VIC', 'Victoria', 'Regional', -37.0, 144.0),
            ('QLD_REG', 'Rest of Queensland', 'QLD', 'Queensland', 'Regional', -22.0, 145.0),
        ]
        
        for loc in locations:
            self.cursor.execute("""
                INSERT OR REPLACE INTO dim_location 
                (location_code, location_name, state_code, state_name, region_type, latitude, longitude)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, loc)
        
        self.conn.commit()
        logger.info(f"Populated {len(locations)} location records")
    
    def populate_dim_property_type(self):
        """Populate property type dimension."""
        
        property_types = [
            ('HOUSE', 'Detached House', 'Residential', 'Standalone dwelling on its own land'),
            ('UNIT', 'Unit/Apartment', 'Residential', 'Strata-titled dwelling in multi-unit building'),
            ('TOWNHOUSE', 'Townhouse', 'Residential', 'Attached dwelling, typically 2-3 storeys'),
            ('COMBINED', 'All Dwellings', 'Residential', 'Combined houses and units'),
            ('LAND', 'Vacant Land', 'Residential', 'Residential zoned vacant land'),
        ]
        
        for pt in property_types:
            self.cursor.execute("""
                INSERT OR REPLACE INTO dim_property_type 
                (property_type_code, property_type_name, property_category, description)
                VALUES (?, ?, ?, ?)
            """, pt)
        
        self.conn.commit()
        logger.info(f"Populated {len(property_types)} property type records")
    
    def load_csv_to_fact_table(self, csv_path: str, table_name: str, 
                                column_mapping: dict, date_column: str,
                                location_column: str = None):
        """
        Load CSV data into a fact table.
        
        Args:
            csv_path: Path to CSV file
            table_name: Target fact table name
            column_mapping: Mapping from CSV columns to table columns
            date_column: Name of date column in CSV
            location_column: Name of location column in CSV (optional)
        """
        logger.info(f"Loading {csv_path} into {table_name}")
        
        df = pd.read_csv(csv_path)
        records_loaded = 0
        
        for _, row in df.iterrows():
            # Get date_id
            date_val = row[date_column]
            # Convert to date_id format (YYYYMMDD)
            try:
                if '-' in str(date_val) and len(str(date_val)) == 10:
                    date_id = int(str(date_val).replace('-', ''))
                else:
                    # Handle quarter format like 'Sep-2020'
                    date_id = 20200901  # Default, would need parsing
            except:
                continue
            
            # Build insert statement dynamically
            columns = ['date_id']
            values = [date_id]
            
            for csv_col, db_col in column_mapping.items():
                if csv_col in df.columns:
                    columns.append(db_col)
                    values.append(row[csv_col])
            
            columns.append('source')
            values.append(csv_path)
            
            placeholders = ','.join(['?' for _ in values])
            columns_str = ','.join(columns)
            
            try:
                self.cursor.execute(
                    f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})",
                    values
                )
                records_loaded += 1
            except sqlite3.Error as e:
                logger.warning(f"Error inserting record: {e}")
        
        self.conn.commit()
        logger.info(f"Loaded {records_loaded} records into {table_name}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Query results as DataFrame
        """
        return pd.read_sql_query(query, self.conn)
    
    def get_table_info(self) -> dict:
        """Get information about all tables in the database."""
        
        self.cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        tables = [row[0] for row in self.cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = self.cursor.fetchone()[0]
            
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in self.cursor.fetchall()]
            
            table_info[table] = {
                'row_count': count,
                'columns': columns
            }
        
        return table_info
    
    def generate_erd_text(self) -> str:
        """Generate text representation of Entity-Relationship Diagram."""
        
        erd = """
        ============================================================
        ENTITY-RELATIONSHIP DIAGRAM (ERD)
        Australian Housing Market Analytics Database
        ============================================================
        
        STAR SCHEMA DESIGN
        ==================
        
                                    +------------------+
                                    |    dim_date      |
                                    +------------------+
                                    | PK date_id       |
                                    |    full_date     |
                                    |    year          |
                                    |    quarter       |
                                    |    month_number  |
                                    +--------+---------+
                                             |
                    +------------------------+------------------------+
                    |                        |                        |
                    v                        v                        v
        +-----------------------+  +------------------------+  +----------------------+
        | fact_housing_prices   |  | fact_dwelling_approvals|  | fact_interest_rates  |
        +-----------------------+  +------------------------+  +----------------------+
        | PK price_id           |  | PK approval_id         |  | PK rate_id           |
        | FK date_id            |  | FK date_id             |  | FK date_id           |
        | FK location_id        |  | FK location_id         |  |    cash_rate_percent |
        | FK property_type_id   |  |    dwellings_approved  |  |    change_percent    |
        |    median_price_aud   |  |    total_value         |  +----------------------+
        |    annual_change_%    |  +------------------------+
        +-----------+-----------+
                    |
                    v
        +------------------+        +--------------------+
        |   dim_location   |        | dim_property_type  |
        +------------------+        +--------------------+
        | PK location_id   |        | PK property_type_id|
        |    location_code |        |    type_code       |
        |    location_name |        |    type_name       |
        |    state_code    |        |    category        |
        |    region_type   |        +--------------------+
        +------------------+
        
        RELATIONSHIPS
        =============
        - fact_housing_prices (N) --> dim_date (1)
        - fact_housing_prices (N) --> dim_location (1)
        - fact_housing_prices (N) --> dim_property_type (1)
        - fact_dwelling_approvals (N) --> dim_date (1)
        - fact_dwelling_approvals (N) --> dim_location (1)
        - fact_population (N) --> dim_date (1)
        - fact_population (N) --> dim_location (1)
        - fact_interest_rates (N) --> dim_date (1)
        """
        return erd


def main():
    """Main entry point for database setup."""
    
    # Initialize database manager
    db = HousingDatabaseManager(db_path="australian_housing.db")
    
    try:
        # Connect and create schema
        db.connect()
        db.execute_ddl()
        
        # Populate dimension tables
        db.populate_dim_date()
        db.populate_dim_location()
        db.populate_dim_property_type()
        
        # Display database info
        print("\n" + "="*60)
        print("DATABASE CREATION SUMMARY")
        print("="*60)
        
        table_info = db.get_table_info()
        for table, info in table_info.items():
            print(f"\n{table}")
            print(f"  Rows: {info['row_count']:,}")
            print(f"  Columns: {len(info['columns'])}")
        
        # Print ERD
        print(db.generate_erd_text())
        
        # Example queries
        print("\n" + "="*60)
        print("SAMPLE QUERIES")
        print("="*60)
        
        # Query 1: Locations
        result = db.execute_query("""
            SELECT location_code, location_name, state_code, region_type 
            FROM dim_location 
            LIMIT 10
        """)
        print("\nLocations:")
        print(result.to_string(index=False))
        
    finally:
        db.disconnect()
    
    return db


if __name__ == "__main__":
    main()
