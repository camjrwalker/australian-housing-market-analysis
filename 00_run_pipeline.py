#!/usr/bin/env python3
"""
Australian Housing Market Analytics - Master Pipeline Runner
=============================================================

This script orchestrates the complete data pipeline from ingestion
through to analysis. Run this to execute all pipeline stages.

Usage:
    python run_pipeline.py [--stage STAGE] [--data-dir DIR]
    
Arguments:
    --stage     Run specific stage only (1-5 or 'all')
    --data-dir  Directory containing input CSV files

Author: Cam Walker
Date: January 2026
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print pipeline banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║     AUSTRALIAN HOUSING MARKET ANALYTICS PIPELINE              ║
    ║     Data Science Portfolio Project                            ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Stage 1: Data Ingestion (Python/pandas)                      ║
    ║  Stage 2: Data Staging (Excel/Power Query equivalent)         ║
    ║  Stage 3: Database Storage (SQL/SQLite)                       ║
    ║  Stage 4: Data Transformation (Feature Engineering)           ║
    ║  Stage 5: Statistical Analysis & Modelling                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_stage_1_ingestion(data_dir: str):
    """Run data ingestion stage."""
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA INGESTION")
    logger.info("=" * 60)
    
    try:
        from importlib import import_module
        
        # Change to data directory
        original_dir = os.getcwd()
        os.chdir(data_dir)
        
        # Import and run
        ingestion = import_module('01_data_ingestion')
        result = ingestion.main()
        
        os.chdir(original_dir)
        logger.info("Stage 1 completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")
        raise


def run_stage_2_staging(data_dir: str):
    """Run data staging stage."""
    logger.info(" ")
    logger.info("=" * 60)
    logger.info("STAGE 2: DATA STAGING")
    logger.info("=" * 60)
    
    try:
        original_dir = os.getcwd()
        os.chdir(data_dir)
        
        from importlib import import_module
        staging = import_module('02_data_staging')
        result = staging.main()
        
        os.chdir(original_dir)
        logger.info("Stage 2 completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")
        raise


def run_stage_3_database(data_dir: str):
    """Run database creation stage."""
    logger.info("=" * 60)
    logger.info("STAGE 3: DATABASE STORAGE")
    logger.info("=" * 60)
    
    try:
        original_dir = os.getcwd()
        os.chdir(data_dir)
        
        from importlib import import_module
        database = import_module('03_database_storage')
        result = database.main()
        
        os.chdir(original_dir)
        logger.info("Stage 3 completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")
        raise


def run_stage_4_transformation(data_dir: str):
    """Run data transformation stage."""
    logger.info("=" * 60)
    logger.info("STAGE 4: DATA TRANSFORMATION")
    logger.info("=" * 60)
    
    try:
        original_dir = os.getcwd()
        os.chdir(data_dir)
        
        from importlib import import_module
        transform = import_module('04_data_transformation')
        result = transform.main()
        
        os.chdir(original_dir)
        logger.info("Stage 4 completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        raise


def run_stage_5_analysis(data_dir: str):
    """Run statistical analysis stage."""
    logger.info("=" * 60)
    logger.info("STAGE 5: STATISTICAL ANALYSIS")
    logger.info("=" * 60)
    
    try:
        original_dir = os.getcwd()
        os.chdir(data_dir)
        
        from importlib import import_module
        analysis = import_module('05_statistical_analysis')
        result = analysis.main()
        
        os.chdir(original_dir)
        logger.info("Stage 5 completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Stage 5 failed: {e}")
        raise


def run_full_pipeline(data_dir: str):
    """Run complete pipeline."""
    start_time = datetime.now()
    
    print_banner()
    logger.info(f"Starting full pipeline at {start_time}")
    logger.info(f"Data directory: {data_dir}")
    
    results = {}
    
    try:
        # Stage 1: Ingestion
        results['ingestion'] = run_stage_1_ingestion(data_dir)
        
        # Stage 2: Staging
        results['staging'] = run_stage_2_staging(data_dir)
        
        # Stage 3: Database
        results['database'] = run_stage_3_database(data_dir)
        
        # Stage 4: Transformation
        results['transformation'] = run_stage_4_transformation(data_dir)
        
        # Stage 5: Analysis
        results['analysis'] = run_stage_5_analysis(data_dir)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total duration: {duration}")
    
    print(f"\n✓ Pipeline completed successfully in {duration}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Australian Housing Market Analytics Pipeline'
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['1', '2', '3', '4', '5', 'all'],
        help='Pipeline stage to run (1-5 or all)'
    )
    parser.add_argument(
    '--data-dir',
    type=str,
    default=r'D:\Users\CameronWalker\Documents\15. Work\New Roles - Data Science\Projects\Housing\raw_data',
    help='Directory containing input CSV files'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Add scripts directory to path
    scripts_dir = Path(__file__).parent.resolve()
    sys.path.insert(0, str(scripts_dir))
    sys.path.insert(0, str(data_dir))
    
    try:
        if args.stage == 'all':
            run_full_pipeline(str(data_dir))
        elif args.stage == '1':
            run_stage_1_ingestion(str(data_dir))
        elif args.stage == '2':
            run_stage_2_staging(str(data_dir))
        elif args.stage == '3':
            run_stage_3_database(str(data_dir))
        elif args.stage == '4':
            run_stage_4_transformation(str(data_dir))
        elif args.stage == '5':
            run_stage_5_analysis(str(data_dir))
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
