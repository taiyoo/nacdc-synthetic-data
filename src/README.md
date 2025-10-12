# Source Code Reference

## 📄 File Descriptions

### `nacdc_synthetic_dp.py` 
**Main synthetic data generator with differential privacy**
- Generates realistic NACDC residential care datasets
- Implements properly calibrated differential privacy mechanisms
- Produces high-quality synthetic data suitable for research
- **Usage**: `python nacdc_synthetic_dp.py`

### `nacdc_analysis.py`
**Dataset analysis and validation toolkit**
- Compares datasets across different privacy levels
- Analyzes privacy-utility tradeoffs
- Generates quality assessment reports and visualizations
- **Usage**: `python nacdc_analysis.py`

### `ultra_concise_zk.py`
**Zero-Knowledge SNARKs demonstration**
- Privacy-preserving access control for aged care data
- ZK proof generation and verification
- Cryptographic authentication without credential disclosure
- **Usage**: `python ultra_concise_zk.py`

## 🔧 Key Classes

### `NACDCResidentialCareSynthesizer`
Main class for generating synthetic data with differential privacy.

**Key Methods:**
- `generate_synthetic_dataset()`: Create synthetic NACDC dataset
- `apply_differential_privacy_refined()`: Apply calibrated DP mechanisms
- `validate_privacy_properties()`: Verify privacy guarantees

### `NACDCDatasetAnalyzer`
Analysis toolkit for comparing and validating synthetic datasets.

**Key Methods:**
- `compare_distributions()`: Compare statistical distributions
- `privacy_utility_analysis()`: Analyze privacy-utility tradeoffs
- `generate_data_quality_report()`: Assess data quality metrics

### `AgedCareZKSystem`
Zero-knowledge proof system for aged care access control.

**Key Methods:**
- `generate_access_challenge()`: Create cryptographic challenges
- `compute_zk_proof()`: Generate zero-knowledge proofs
- `verify_zk_proof()`: Verify proofs without revealing credentials

## 🚀 Quick Examples

### Generate Synthetic Data
```python
from nacdc_synthetic_dp import NACDCResidentialCareSynthesizer

synthesizer = NACDCResidentialCareSynthesizer(epsilon=1.0)
data = synthesizer.generate_synthetic_dataset(n_records=5000)
```

### Analyze Datasets
```python
from nacdc_analysis import NACDCDatasetAnalyzer

analyzer = NACDCDatasetAnalyzer()
datasets = analyzer.load_datasets()
analyzer.compare_distributions(datasets)
```

### ZK Proof Demo
```python
from ultra_concise_zk import demo_aged_care_zk

demo_aged_care_zk()  # Run complete demonstration
```
