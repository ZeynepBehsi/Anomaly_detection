"""
Data Processing Module for Anomaly Detection Toolkit
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
import logging


class DataProcessor:
    """
    Data processing utilities for loading and preparing data for anomaly detection
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.logger = logging.getLogger(__name__)
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
    
    def load_data(
        self, 
        filepath: Union[str, Path], 
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            filepath: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        suffix = filepath.suffix.lower()
        
        try:
            if suffix == '.csv':
                df = self._load_csv(filepath, **kwargs)
            elif suffix in ['.xlsx', '.xls']:
                df = self._load_excel(filepath, **kwargs)
            elif suffix == '.json':
                df = pd.read_json(filepath, **kwargs)
            elif suffix == '.parquet':
                df = pd.read_parquet(filepath, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
            
            self.logger.info(f"Successfully loaded data from {filepath}")
            self.logger.info(f"Data shape: {df.shape}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
    
    def _load_csv(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file with automatic delimiter detection"""
        # Try common delimiters
        delimiters = [',', ';', '\t', '|']
        
        for delimiter in delimiters:
            try:
                df = pd.read_csv(filepath, delimiter=delimiter, nrows=5, **kwargs)
                if df.shape[1] > 1:  # Found proper delimiter
                    df = pd.read_csv(filepath, delimiter=delimiter, **kwargs)
                    return df
            except:
                continue
        
        # Fallback to default
        return pd.read_csv(filepath, **kwargs)
    
    def _load_excel(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Load Excel file with header detection"""
        # First, check if there are multiple header rows (like your original data)
        df_preview = pd.read_excel(filepath, header=None, nrows=10)
        
        # Check for multi-level headers
        if self._has_multi_level_headers(df_preview):
            return self._process_multi_level_headers(filepath, **kwargs)
        else:
            return pd.read_excel(filepath, **kwargs)
    
    def _has_multi_level_headers(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has multi-level headers"""
        # Simple heuristic: if first few rows have mostly string values
        # and subsequent rows have mostly numeric values
        if df.shape[0] < 3:
            return False
        
        first_rows_string = df.iloc[:2].applymap(lambda x: isinstance(x, str)).sum().sum()
        total_cells_first_rows = df.iloc[:2].size
        
        return (first_rows_string / total_cells_first_rows) > 0.5
    
    def _process_multi_level_headers(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Process files with multi-level headers (like your original data)"""
        df = pd.read_excel(filepath, header=None, **kwargs)
        
        if df.shape[0] < 2:
            return df
        
        # Get header rows
        upper_header = df.iloc[0].tolist()
        lower_header = df.iloc[1].tolist()
        
        # Fill forward NaN values in upper header
        for i in range(len(upper_header)):
            if pd.isna(upper_header[i]) and i > 0:
                upper_header[i] = upper_header[i-1]
        
        # Create combined column names
        new_columns = []
        for i in range(len(upper_header)):
            upper = str(upper_header[i]) if not pd.isna(upper_header[i]) else 'Col'
            lower = str(lower_header[i]) if not pd.isna(lower_header[i]) else f'{i}'
            new_columns.append(f"{upper}_{lower}")
        
        # Remove header rows and set new columns
        df = df.drop([0, 1]).reset_index(drop=True)
        df.columns = new_columns[:len(df.columns)]
        
        return df
    
    def clean_data(
        self, 
        df: pd.DataFrame, 
        remove_duplicates: bool = True,
        handle_missing: str = 'median',
        numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clean and preprocess the data
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: How to handle missing values ('median', 'mean', 'drop', 'forward_fill')
            numeric_columns: List of columns to treat as numeric
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Clean string columns (remove extra spaces)
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        # Clean column names
        df_clean.columns = df_clean.columns.str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
        
        # Convert numeric columns
        if numeric_columns:
            for col in numeric_columns:
                if col in df_clean.columns:
                    df_clean[col] = self._convert_to_numeric(df_clean[col])
        else:
            # Auto-detect and convert numeric columns
            for col in df_clean.columns:
                df_clean[col] = self._try_convert_numeric(df_clean[col])
        
        # Handle missing values
        if handle_missing != 'none':
            df_clean = self._handle_missing_values(df_clean, method=handle_missing)
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                self.logger.info(f"Removed {removed_rows} duplicate rows")
        
        self.logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def _convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """Convert series to numeric, handling common formatting issues"""
        if series.dtype in ['int64', 'float64']:
            return series
        
        # Convert to string first
        series_str = series.astype(str)
        
        # Replace common formatting issues
        series_str = series_str.str.replace(' ', '')  # Remove spaces
        series_str = series_str.str.replace(',', '.')  # Replace comma with dot
        series_str = series_str.replace('', np.nan)    # Replace empty strings with NaN
        series_str = series_str.replace('nan', np.nan) # Replace 'nan' strings
        
        # Convert to numeric
        return pd.to_numeric(series_str, errors='coerce')
    
    def _try_convert_numeric(self, series: pd.Series) -> pd.Series:
        """Try to convert series to numeric if it looks numeric"""
        if series.dtype in ['int64', 'float64']:
            return series
        
        # Check if the series looks numeric
        try:
            # Convert to string and clean
            series_str = series.astype(str)
            series_clean = series_str.str.replace(' ', '').str.replace(',', '.')
            
            # Try converting a sample
            sample = series_clean.head(10).replace('nan', np.nan)
            pd.to_numeric(sample, errors='raise')
            
            # If successful, convert the whole series
            return pd.to_numeric(series_clean.replace('nan', np.nan), errors='coerce')
        except:
            # Not numeric, return as is
            return series
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str = 'median') -> pd.DataFrame:
        """Handle missing values in the DataFrame"""
        df_filled = df.copy()
        
        for col in df_filled.columns:
            if df_filled[col].isnull().any():
                if df_filled[col].dtype in ['int64', 'float64']:
                    # Numeric column
                    if method == 'median':
                        df_filled[col] = df_filled[col].fillna(df_filled[col].median())
                    elif method == 'mean':
                        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
                    elif method == 'forward_fill':
                        df_filled[col] = df_filled[col].fillna(method='ffill')
                    elif method == 'drop':
                        df_filled = df_filled.dropna(subset=[col])
                else:
                    # Non-numeric column
                    if method == 'forward_fill':
                        df_filled[col] = df_filled[col].fillna(method='ffill')
                    elif method != 'drop':
                        # Fill with mode (most frequent value)
                        mode_value = df_filled[col].mode()
                        if len(mode_value) > 0:
                            df_filled[col] = df_filled[col].fillna(mode_value[0])
                    elif method == 'drop':
                        df_filled = df_filled.dropna(subset=[col])
        
        return df_filled
    
    def get_column_info(self, df: pd.DataFrame) -> Dict:
        """Get information about DataFrame columns"""
        info = {
            'total_columns': len(df.columns),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        return info
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of the DataFrame"""
        summary = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict()
        }
        return summary