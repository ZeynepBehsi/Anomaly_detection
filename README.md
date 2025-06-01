# 🏗️ System Architecture
```mermaid
flowchart TB
    subgraph Proje Mimarisi
        A[Veri Girişi] --> B[DataProcessor]
        B --> C[AnomalyDetector]
        C --> D[Algoritmalar]
        D --> E[Sonuçlar]
    end

    subgraph Algoritmalar
        D --> F[İstatistiksel\nZ-Score, IQR]
        D --> G[Makine Öğrenmesi\nIsolation Forest, One-Class SVM]
        D --> H[Kümeleme\nDBSCAN, K-Means]
        D --> I[Çok Değişkenli\nMahalanobis, PCA]
    end

    E --> J[Raporlar]
    E --> K[Görselleştirmeler]
    E --> L[Metrics]
```


# 🚀 Features

## Multiple Anomaly Detection Algorithms

Statistical Methods: Z-Score, IQR
Machine Learning: Isolation Forest, One-Class SVM, LOF
Clustering: DBSCAN, K-Means
Multivariate: Mahalanobis Distance, Elliptic Envelope, PCA-based


## Flexible Data Input

Support for CSV, Excel files
Automatic column detection
Custom column selection
Data preprocessing and cleaning


## Interactive Interface

Command-line interface
Jupyter notebook examples
Configuration file support


## Comprehensive Output

Detailed anomaly reports
Visualization plots
Performance metrics
Export results in multiple formats

# 🔄 Workflow

```mermaid
journey
    title Anomali Tespit İş Akışı
    section Veri Yükleme
      Kullanıcı veri dosyası yükler: 5: Kullanıcı
      Sistem veriyi ön işler: 4: Sistem
    section Tespit
      Kullanıcı algoritma seçer: 3: Kullanıcı
      Sistem anomali tespit eder: 5: Sistem
    section Çıktı
      Sonuçlar gösterilir: 4: Sistem
      Kullanıcı sonuçları export eder: 4: Kullanıcı
```

## 📋 Requirements
See requirements.txt for full dependencies.

## Install dependencies
pip install -r requirements.txt

## Or install as a package
pip install -e .
🎯 Quick Start
Basic Usage
pythonfrom src.anomaly_detector import AnomalyDetector
from src.data_processor import DataProcessor

## Load and process data
processor = DataProcessor()
data = processor.load_data('data/input/your_dataset.csv')

## Initialize detector
detector = AnomalyDetector()

## Detect anomalies
results = detector.detect_anomalies(
    data=data,
    columns=['column1', 'column2'],
    methods=['isolation_forest', 'z_score'],
    logic='union'
)

## Save results
results.to_csv('data/output/anomalies.csv')
Command Line Usage
bashpython -m src.anomaly_detector \
    --input data/input/dataset.csv \
    --columns col1,col2,col3 \
    --methods isolation_forest,z_score \
    --output data/output/results.csv

# 📊 Supported Algorithms

## Univariate Methods
- Z-Score: Statistical outlier detection based on standard deviations
- IQR (Interquartile Range): Outliers beyond Q1-1.5IQR or Q3+1.5IQR
- Isolation Forest: Tree-based anomaly detection
- Local Outlier Factor (LOF): Density-based outlier detection
- One-Class SVM: Support Vector Machine for novelty detection

## Multivariate Methods
- Mahalanobis Distance: Distance-based detection considering correlations
- Elliptic Envelope: Robust covariance estimation
- PCA-based: Principal Component Analysis reconstruction error

## Clustering Methods
- DBSCAN: Density-based clustering outliers
- K-Means: Distance from cluster centers

# 🔧 Configuration
Customize detection parameters in config/config.yaml:
yamldetection:
  default_contamination: 0.05
  z_score_threshold: 3
  iqr_multiplier: 1.5
  
visualization:
  figure_size: [12, 8]
  color_palette: "viridis"
  
output:
  include_scores: true
  export_plots: true

# 📈 Examples
See the examples/ directory and Jupyter notebooks for detailed usage examples.
🤝 Contributing

Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

# 📧 Contact
zeynepbehsi@gmail.com


🙏 Acknowledgments
Scikit-learn community for excellent ML algorithms
Pandas team for data manipulation tools
All contributors who help improve this toolkit
