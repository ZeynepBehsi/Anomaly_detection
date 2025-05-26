"""
Anomaly Detection Toolkit - Main Detection Module
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis
import logging
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    A comprehensive anomaly detection toolkit supporting multiple algorithms
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the anomaly detector
        
        Args:
            config: Configuration dictionary for detector parameters
        """
        self.config = config or self._get_default_config()
        self.scaler = None
        self.scaled_data = None
        self.feature_names = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Available detection methods
        self.methods = {
            'z_score': {
                'func': self._detect_z_score,
                'multivariate': False,
                'description': 'Statistical outlier detection using Z-scores'
            },
            'iqr': {
                'func': self._detect_iqr,
                'multivariate': False,
                'description': 'Interquartile Range based outlier detection'
            },
            'isolation_forest': {
                'func': self._detect_isolation_forest,
                'multivariate': True,
                'description': 'Tree-based anomaly detection'
            },
            'lof': {
                'func': self._detect_lof,
                'multivariate': True,
                'description': 'Local Outlier Factor density-based detection'
            },
            'one_class_svm': {
                'func': self._detect_one_class_svm,
                'multivariate': True,
                'description': 'Support Vector Machine novelty detection'
            },
            'dbscan': {
                'func': self._detect_dbscan,
                'multivariate': True,
                'description': 'DBSCAN clustering-based outlier detection'
            },
            'kmeans': {
                'func': self._detect_kmeans,
                'multivariate': True,
                'description': 'K-Means distance-based outlier detection'
            },
            'mahalanobis': {
                'func': self._detect_mahalanobis,
                'multivariate': True,
                'description': 'Mahalanobis distance-based detection'
            },
            'elliptic_envelope': {
                'func': self._detect_elliptic_envelope,
                'multivariate': True,
                'description': 'Robust covariance estimation'
            },
            'pca': {
                'func': self._detect_pca,
                'multivariate': True,
                'description': 'PCA reconstruction error based detection'
            }
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'contamination': 0.05,
            'z_score_threshold': 3,
            'iqr_multiplier': 1.5,
            'mahalanobis_threshold': 3,
            'pca_components': 0.95,
            'pca_threshold': 2,
            'scaler_type': 'standard',  # 'standard', 'robust', 'minmax'
            'random_state': 42
        }
    
    def get_available_methods(self) -> Dict[str, str]:
        """Get available detection methods with descriptions"""
        return {name: info['description'] for name, info in self.methods.items()}
    
    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> 'AnomalyDetector':
        """
        Fit the detector to the data
        
        Args:
            data: Input DataFrame
            columns: List of column names to use. If None, uses all numeric columns
            
        Returns:
            self
        """
        if columns is None:
            # Select numeric columns automatically
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_names = columns
        feature_data = data[columns].copy()
        
        # Handle missing values
        feature_data = self._handle_missing_values(feature_data)
        
        # Scale the data
        self._fit_scaler(feature_data)
        self.scaled_data = self.scaler.transform(feature_data)
        
        self.logger.info(f"Fitted detector with {len(columns)} features: {columns}")
        return self
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        methods: Optional[List[str]] = None,
        logic: str = 'union',
        return_scores: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Detect anomalies using specified methods
        
        Args:
            data: Input DataFrame
            columns: Columns to analyze
            methods: List of method names to use
            logic: 'union' (any method) or 'intersection' (all methods)
            return_scores: Whether to return anomaly scores
            
        Returns:
            DataFrame with anomaly flags, optionally with scores
        """
        if methods is None:
            methods = ['isolation_forest', 'z_score']
        
        # Validate methods
        invalid_methods = set(methods) - set(self.methods.keys())
        if invalid_methods:
            raise ValueError(f"Invalid methods: {invalid_methods}")
        
        # Fit if not already fitted
        if self.scaled_data is None or columns != self.feature_names:
            self.fit(data, columns)
        
        results = pd.DataFrame(index=data.index)
        scores = pd.DataFrame(index=data.index)
        
        # Apply each method
        for method_name in methods:
            self.logger.info(f"Applying {method_name}...")
            
            method_info = self.methods[method_name]
            
            if method_info['multivariate'] and len(self.feature_names) > 1:
                # Multivariate method
                anomalies, method_scores = method_info['func'](self.scaled_data)
                results[f'{method_name}_anomaly'] = anomalies
                if method_scores is not None:
                    scores[f'{method_name}_score'] = method_scores
            else:
                # Apply to each feature separately
                for i, feature in enumerate(self.feature_names):
                    feature_data = data[feature].values
                    anomalies, method_scores = method_info['func'](feature_data)
                    results[f'{feature}_{method_name}_anomaly'] = anomalies
                    if method_scores is not None:
                        scores[f'{feature}_{method_name}_score'] = method_scores
        
        # Combine results based on logic
        anomaly_columns = [col for col in results.columns if col.endswith('_anomaly')]
        
        if logic == 'intersection':
            results['is_anomaly'] = results[anomaly_columns].all(axis=1)
        else:  # union
            results['is_anomaly'] = results[anomaly_columns].any(axis=1)
        
        # Add original data
        result_df = pd.concat([data, results], axis=1)
        
        if return_scores:
            return result_df, scores
        return result_df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        # Use median for numerical columns (robust to outliers)
        for col in data.columns:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
        return data
    
    def _fit_scaler(self, data: pd.DataFrame):
        """Fit the appropriate scaler"""
        scaler_type = self.config.get('scaler_type', 'standard')
        
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:  # default to standard
            self.scaler = StandardScaler()
        
        self.scaler.fit(data)
    
    # Detection Methods
    def _detect_z_score(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score based anomaly detection"""
        if data.ndim > 1:
            data = data.flatten()
        
        threshold = self.config.get('z_score_threshold', 3)
        z_scores = np.abs(zscore(data, nan_policy='omit'))
        anomalies = z_scores > threshold
        return anomalies, z_scores
    
    def _detect_iqr(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """IQR based anomaly detection"""
        if data.ndim > 1:
            data = data.flatten()
        
        multiplier = self.config.get('iqr_multiplier', 1.5)
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        scores = np.maximum(lower_bound - data, data - upper_bound)
        scores = np.maximum(scores, 0)  # Only positive scores
        
        return anomalies, scores
    
    def _detect_isolation_forest(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Isolation Forest anomaly detection"""
        contamination = self.config.get('contamination', 0.05)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        model = IsolationForest(
            contamination=contamination,
            random_state=self.config.get('random_state', 42)
        )
        predictions = model.fit_predict(data)
        scores = -model.score_samples(data)  # Higher scores = more anomalous
        
        return predictions == -1, scores
    
    def _detect_lof(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Local Outlier Factor detection"""
        contamination = self.config.get('contamination', 0.05)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        model = LocalOutlierFactor(contamination=contamination)
        predictions = model.fit_predict(data)
        scores = -model.negative_outlier_factor_
        
        return predictions == -1, scores
    
    def _detect_one_class_svm(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One-Class SVM detection"""
        nu = self.config.get('contamination', 0.05)
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        model = OneClassSVM(nu=nu)
        predictions = model.fit_predict(data)
        scores = -model.score_samples(data)
        
        return predictions == -1, scores
    
    def _detect_dbscan(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """DBSCAN clustering based detection"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(data)
        anomalies = labels == -1
        
        # Calculate distance to nearest cluster as score
        scores = np.zeros(len(data))
        if len(np.unique(labels)) > 1:  # If clusters found
            for i, point in enumerate(data):
                if labels[i] == -1:  # Outlier
                    # Distance to nearest non-outlier point
                    non_outliers = data[labels != -1]
                    if len(non_outliers) > 0:
                        distances = np.linalg.norm(non_outliers - point, axis=1)
                        scores[i] = np.min(distances)
        
        return anomalies, scores
    
    def _detect_kmeans(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """K-Means distance based detection"""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        n_clusters = min(5, len(data) // 2)  # Adaptive cluster number
        model = KMeans(n_clusters=n_clusters, random_state=self.config.get('random_state', 42))
        model.fit(data)
        
        distances = np.min(model.transform(data), axis=1)
        threshold = np.percentile(distances, 95)  # Top 5% as anomalies
        
        return distances > threshold, distances
    
    def _detect_mahalanobis(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mahalanobis distance based detection"""
        try:
            mean = np.mean(data, axis=0)
            cov = np.cov(data, rowvar=False)
            
            # Handle singular covariance matrix
            if np.linalg.det(cov) == 0:
                cov += np.eye(cov.shape[0]) * 1e-6
            
            inv_cov = np.linalg.inv(cov)
            distances = np.array([
                mahalanobis(row, mean, inv_cov) for row in data
            ])
            
            threshold = self.config.get('mahalanobis_threshold', 3)
            anomalies = distances > threshold * np.mean(distances)
            
            return anomalies, distances
        except Exception as e:
            self.logger.warning(f"Mahalanobis detection failed: {e}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    
    def _detect_elliptic_envelope(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Elliptic Envelope detection"""
        contamination = self.config.get('contamination', 0.05)
        
        try:
            model = EllipticEnvelope(contamination=contamination)
            predictions = model.fit_predict(data)
            scores = -model.score_samples(data)
            
            return predictions == -1, scores
        except Exception as e:
            self.logger.warning(f"Elliptic Envelope detection failed: {e}")
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
    
    def _detect_pca(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """PCA reconstruction error based detection"""
        n_components = self.config.get('pca_components', 0.95)
        threshold = self.config.get('pca_threshold', 2)
        
        # Determine number of components
        if isinstance(n_components, float) and n_components < 1:
            pca_temp = PCA()
            pca_temp.fit(data)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= n_components) + 1
        
        n_components = min(n_components, data.shape[1], data.shape[0] - 1)
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(reduced)
        
        # Reconstruction error
        errors = np.sqrt(np.sum((data - reconstructed) ** 2, axis=1))
        anomalies = errors > threshold * np.mean(errors)
        
        return anomalies, errors