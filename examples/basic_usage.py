#!/usr/bin/env python3
"""
Basic Usage Example for Anomaly Detection Toolkit

This example shows how to use the toolkit for basic anomaly detection
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.anomaly_detector import AnomalyDetector
from src.data_processor import DataProcessor


def create_sample_data():
    """Create sample data with some anomalies for demonstration"""
    np.random.seed(42)
    
    # Normal data
    normal_data = {
        'feature1': np.random.normal(50, 10, 950),
        'feature2': np.random.normal(100, 15, 950),
        'feature3': np.random.normal(25, 5, 950),
        'category': np.random.choice(['A', 'B', 'C'], 950)
    }
    
    # Anomalous data
    anomaly_data = {
        'feature1': np.random.normal(150, 20, 50),  # Much higher values
        'feature2': np.random.normal(10, 5, 50),    # Much lower values
        'feature3': np.random.normal(80, 10, 50),   # Much higher values
        'category': np.random.choice(['A', 'B', 'C'], 50)
    }
    
    # Combine data
    data = pd.DataFrame({
        'feature1': np.concatenate([normal_data['feature1'], anomaly_data['feature1']]),
        'feature2': np.concatenate([normal_data['feature2'], anomaly_data['feature2']]),
        'feature3': np.concatenate([normal_data['feature3'], anomaly_data['feature3']]),
        'category': np.concatenate([normal_data['category'], anomaly_data['category']])
    })
    
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    
    return data


def basic_example():
    """Basic anomaly detection example"""
    print("🔍 Basic Anomaly Detection Example")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample data...")
    data = create_sample_data()
    print(f"Created dataset with {len(data)} records and {len(data.columns)} columns")
    print("\nData overview:")
    print(data.describe())
    
    # Initialize detector
    print("\n🔧 Initializing anomaly detector...")
    detector = AnomalyDetector()
    
    # Show available methods
    print("\nAvailable detection methods:")
    for method, description in detector.get_available_methods().items():
        print(f"  - {method}: {description}")
    
    # Detect anomalies using default methods
    print("\n🚀 Detecting anomalies with default methods (isolation_forest + z_score)...")
    
    # Select numeric columns
    numeric_columns = ['feature1', 'feature2', 'feature3']
    
    results = detector.detect_anomalies(
        data=data,
        columns=numeric_columns,
        methods=['isolation_forest', 'z_score'],
        logic='union'  # Any method can flag as anomaly
    )
    
    # Analyze results
    anomaly_count = results['is_anomaly'].sum()
    total_count = len(results)
    percentage = (anomaly_count / total_count) * 100
    
    print(f"✅ Detection completed!")
    print(f"Found {anomaly_count} anomalies out of {total_count} records ({percentage:.2f}%)")
    
    # Show some anomalies
    if anomaly_count > 0:
        print(f"\nSample anomalies:")
        anomalies = results[results['is_anomaly']][['feature1', 'feature2', 'feature3', 'is_anomaly']].head(10)
        print(anomalies)
    
    # Show some normal records for comparison
    print(f"\nSample normal records:")
    normal = results[~results['is_anomaly']][['feature1', 'feature2', 'feature3', 'is_anomaly']].head(5)
    print(normal)
    
    return results


def multiple_methods_example():
    """Example using multiple detection methods"""
    print("\n\n🔍 Multiple Methods Comparison Example")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    detector = AnomalyDetector()
    
    # List of methods to compare
    methods_to_test = ['z_score', 'iqr', 'isolation_forest', 'lof']
    
    print(f"🔧 Testing methods: {', '.join(methods_to_test)}")
    
    # Test with union logic (any method flags as anomaly)
    print(f"\n🚀 Testing with UNION logic (any method can flag anomaly)...")
    
    results_union = detector.detect_anomalies(
        data=data,
        columns=['feature1', 'feature2', 'feature3'],
        methods=methods_to_test,
        logic='union'
    )
    
    union_anomalies = results_union['is_anomaly'].sum()
    print(f"Union logic found: {union_anomalies} anomalies ({union_anomalies/len(data)*100:.2f}%)")
    
    # Test with intersection logic (all methods must agree)
    print(f"\n🚀 Testing with INTERSECTION logic (all methods must agree)...")
    
    results_intersection = detector.detect_anomalies(
        data=data,
        columns=['feature1', 'feature2', 'feature3'],
        methods=methods_to_test,
        logic='intersection'
    )
    intersection_anomalies = results_intersection['is_anomaly'].sum()
    print(f"Intersection logic found: {intersection_anomalies} anomalies ({intersection_anomalies/len(data)*100:.2f}%)")


def file_processing_example():
    """Example with file processing"""
    print("\n\n🔍 File Processing Example")
    print("=" * 50)
    
    # Create sample data and save to file
    data = create_sample_data()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/input")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data
    sample_file = data_dir / "sample_data.csv"
    data.to_csv(sample_file, index=False)
    print(f"📁 Sample data saved to: {sample_file}")
    
    # Initialize processor and detector
    processor = DataProcessor()
    detector = AnomalyDetector()
    
    # Load data using processor
    print(f"📂 Loading data from file...")
    loaded_data = processor.load_data(sample_file)
    
    # Get data info
    data_info = processor.get_column_info(loaded_data)
    print(f"Numeric columns: {data_info['numeric_columns']}")
    
    # Clean data
    cleaned_data = processor.clean_data(loaded_data)
    
    # Detect anomalies
    results = detector.detect_anomalies(
        data=cleaned_data,
        columns=data_info['numeric_columns'],
        methods=['isolation_forest', 'z_score'],
        logic='union'
    )
    
    # Save results
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "anomaly_results.csv"
    results.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    return results


def advanced_example():
    """Advanced example with multiple configurations"""
    print("\n\n🔍 Advanced Configuration Example")
    print("=" * 50)
    
    # Create sample data
    data = create_sample_data()
    
    # Custom configuration
    custom_config = {
        'contamination': 0.1,  # Expect 10% anomalies
        'z_score_threshold': 2.5,  # More sensitive
        'scaler_type': 'robust'  # Use robust scaler
    }
    
    detector = AnomalyDetector(config=custom_config)
    
    # Test all available methods
    all_methods = list(detector.get_available_methods().keys())
    print(f"Testing all methods: {all_methods}")
    
    results = detector.detect_anomalies(
        data=data,
        columns=['feature1', 'feature2', 'feature3'],
        methods=all_methods,
        logic='union',
        return_scores=True
    )
    
    result_df, scores_df = results
    
    anomaly_count = result_df['is_anomaly'].sum()
    print(f"Found {anomaly_count} anomalies with custom configuration")
    
    # Show method comparison
    method_columns = [col for col in result_df.columns if col.endswith('_anomaly') and col != 'is_anomaly']
    method_counts = result_df[method_columns].sum()
    
    print("\nMethod comparison:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} anomalies")
    
    return result_df, scores_df


if __name__ == "__main__":
    # Run all examples
    try:
        # Basic example
        basic_results = basic_example()
        
        # Multiple methods example
        multiple_methods_example()
        
        # File processing example
        file_results = file_processing_example()
        
        # Advanced example
        advanced_results, scores = advanced_example()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()