#!/usr/bin/env python3
"""
Anomaly Detection Toolkit - Command Line Interface
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import yaml
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.anomaly_detector import AnomalyDetector
from src.data_processor import DataProcessor


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Anomaly Detection Toolkit - Detect anomalies in your data using multiple algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default methods
  python main.py --input data/input/dataset.csv --output data/output/results.csv
  
  # Specify columns and methods
  python main.py --input data.csv --columns col1,col2,col3 --methods isolation_forest,z_score
  
  # Use intersection logic (all methods must agree)
  python main.py --input data.csv --methods isolation_forest,lof,z_score --logic intersection
  
  # Interactive mode
  python main.py --input data.csv --interactive
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to input data file (CSV, Excel, etc.)')
    parser.add_argument('--output', '-o', 
                       help='Path to output file (default: auto-generated)')
    
    # Data processing arguments
    parser.add_argument('--columns', '-c', 
                       help='Comma-separated list of columns to analyze (default: all numeric)')
    parser.add_argument('--exclude-columns', 
                       help='Comma-separated list of columns to exclude')
    
    # Detection arguments
    parser.add_argument('--methods', '-m', 
                       help='Comma-separated list of detection methods (default: isolation_forest,z_score)')
    parser.add_argument('--logic', choices=['union', 'intersection'], default='union',
                       help='Logic for combining multiple methods (default: union)')
    parser.add_argument('--contamination', type=float, default=0.05,
                       help='Expected proportion of anomalies (default: 0.05)')
    
    # Configuration
    parser.add_argument('--config', 
                       help='Path to configuration YAML file')
    parser.add_argument('--list-methods', action='store_true',
                       help='List available detection methods and exit')
    
    # Mode arguments
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    # Output options
    parser.add_argument('--include-scores', action='store_true',
                       help='Include anomaly scores in output')
    parser.add_argument('--only-anomalies', action='store_true',
                       help='Output only anomalous records')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    if args.contamination != 0.05:  # Override config if specified
        config['contamination'] = args.contamination
    
    # Initialize detector
    detector = AnomalyDetector(config)
    
    # List methods if requested
    if args.list_methods:
        print("\nAvailable Anomaly Detection Methods:")
        print("=" * 50)
        for method, description in detector.get_available_methods().items():
            print(f"{method:20} : {description}")
        return
    
    # Interactive mode
    if args.interactive:
        run_interactive_mode(detector, args.input)
        return
    
    # Batch mode
    run_batch_mode(args, detector, logger)


def run_interactive_mode(detector: AnomalyDetector, input_path: str):
    """Run the tool in interactive mode"""
    print("\n🔍 Anomaly Detection Toolkit - Interactive Mode")
    print("=" * 60)
    
    # Load data
    processor = DataProcessor()
    try:
        data = processor.load_data(input_path)
        print(f"✅ Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Show data info
    print(f"\nData Overview:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    numeric_columns = data.select_dtypes(include=['number']).columns.tolist()
    print(f"Numeric columns: {numeric_columns}")
    
    # Column selection
    print(f"\n📊 Select columns to analyze:")
    print("Available numeric columns:")
    for i, col in enumerate(numeric_columns, 1):
        print(f"  {i}. {col}")
    
    selection = input("\nEnter column numbers (comma-separated) or 'all' for all numeric columns: ").strip()
    
    if selection.lower() == 'all':
        columns = numeric_columns
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            columns = [numeric_columns[i] for i in indices if 0 <= i < len(numeric_columns)]
        except:
            print("❌ Invalid selection. Using all numeric columns.")
            columns = numeric_columns
    
    print(f"Selected columns: {columns}")
    
    # Method selection
    print(f"\n🔧 Select detection methods:")
    methods = list(detector.get_available_methods().keys())
    for i, method in enumerate(methods, 1):
        desc = detector.get_available_methods()[method]
        print(f"  {i}. {method} - {desc}")
    
    method_selection = input("\nEnter method numbers (comma-separated) or press Enter for default (isolation_forest,z_score): ").strip()
    
    if not method_selection:
        selected_methods = ['isolation_forest', 'z_score']
    else:
        try:
            indices = [int(x.strip()) - 1 for x in method_selection.split(',')]
            selected_methods = [methods[i] for i in indices if 0 <= i < len(methods)]
        except:
            print("❌ Invalid selection. Using default methods.")
            selected_methods = ['isolation_forest', 'z_score']
    
    print(f"Selected methods: {selected_methods}")
    
    # Logic selection
    logic = input("\nCombination logic (union/intersection) [union]: ").strip().lower() or 'union'
    
    # Run detection
    print(f"\n🚀 Running anomaly detection...")
    try:
        results = detector.detect_anomalies(
            data=data,
            columns=columns,
            methods=selected_methods,
            logic=logic
        )
        
        anomaly_count = results['is_anomaly'].sum()
        print(f"✅ Detection completed!")
        print(f"Found {anomaly_count} anomalies out of {len(results)} records ({anomaly_count/len(results)*100:.2f}%)")
        
        # Save results
        output_path = input(f"\nEnter output file path [data/output/anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv]: ").strip()
        if not output_path:
            Path("data/output").mkdir(parents=True, exist_ok=True)
            output_path = f"data/output/anomalies_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        results.to_csv(output_path, index=False)
        print(f"✅ Results saved to: {output_path}")
        
        # Show sample anomalies
        if anomaly_count > 0:
            print(f"\nSample anomalies:")
            print(results[results['is_anomaly']].head())
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")


def run_batch_mode(args, detector: AnomalyDetector, logger):
    """Run the tool in batch mode"""
    logger.info("Starting anomaly detection in batch mode")
    
    # Load data
    processor = DataProcessor()
    try:
        data = processor.load_data(args.input)
        logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Process columns
    if args.columns:
        columns = [col.strip() for col in args.columns.split(',')]
    else:
        columns = data.select_dtypes(include=['number']).columns.tolist()
        logger.info(f"Auto-selected numeric columns: {columns}")
    
    if args.exclude_columns:
        exclude_cols = [col.strip() for col in args.exclude_columns.split(',')]
        columns = [col for col in columns if col not in exclude_cols]
        logger.info(f"Excluded columns: {exclude_cols}")
    
    # Process methods
    if args.methods:
        methods = [method.strip() for method in args.methods.split(',')]
    else:
        methods = ['isolation_forest', 'z_score']
        logger.info(f"Using default methods: {methods}")
    
    # Run detection
    try:
        logger.info("Running anomaly detection...")
        results = detector.detect_anomalies(
            data=data,
            columns=columns,
            methods=methods,
            logic=args.logic,
            return_scores=args.include_scores
        )
        
        if args.include_scores:
            results, scores = results
        
        anomaly_count = results['is_anomaly'].sum()
        logger.info(f"Found {anomaly_count} anomalies out of {len(results)} records ({anomaly_count/len(results)*100:.2f}%)")
        
        # Filter results if requested
        if args.only_anomalies:
            results = results[results['is_anomaly']]
            logger.info(f"Filtered to {len(results)} anomalous records only")
        
        # Generate output path
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_dir = Path("data/output")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_dir / f"anomalies_{input_path.stem}_{timestamp}.csv"
        
        # Save results
        results.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")
        
        if args.include_scores:
            scores_path = str(output_path).replace('.csv', '_scores.csv')
            scores.to_csv(scores_path, index=False)
            logger.info(f"Scores saved to: {scores_path}")
        
        print(f"✅ Anomaly detection completed successfully!")
        print(f"Found {anomaly_count} anomalies ({anomaly_count/len(data)*100:.2f}%)")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()