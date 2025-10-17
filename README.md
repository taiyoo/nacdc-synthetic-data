# Aged Care Synthetic Data Generator

Australian aged care synthetic data generation with formal differential privacy protection and comprehensive privacy-utility analysis.

## 🎯 Overview

This repository provides a **research-focused synthetic data generator** for Australian aged care data with:

### 🔬 Aged Care Dataset Generator  
- **Purpose**: Academic research on aged care populations
- **Features**: Realistic demographics, health assessments, care episodes with ACFI scores
- **Privacy**: Advanced differential privacy with optimized utility preservation
- **Analysis**: Comprehensive privacy-utility trade-off evaluation
- **Output**: Baseline + multiple privacy levels (ε = 0.5, 1.0, 2.0)

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/taiyoo/nacdc-synthetic-data.git
cd nacdc-synthetic-data

# Setup
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate and Analyze Data
```bash
# Generate baseline + multiple DP levels (ε = 0.5, 1.0, 2.0) automatically
python src/agedcare_synthetic_dp.py

# Comprehensive privacy-utility trade-off analysis
python src/agedcare_tradeoff_analysis.py

# Demonstrate zero-knowledge access control for aged care data
python src/agedcare_zk_snarks.py
```

## 🔒 Privacy Protection

The generator implements **formal differential privacy** providing mathematical privacy guarantees:

### Privacy Parameters
- **ε (epsilon)**: Privacy budget - lower values = stronger privacy
  - ε = 0.5: High privacy (research-grade protection)
  - ε = 1.0: Balanced privacy-utility (recommended)
  - ε = 2.0: Moderate privacy (good utility)

- **δ (delta)**: Privacy failure probability (fixed at 1e-5)

### Privacy Mechanisms
- **Numerical data**: Calibrated Laplace noise with realistic sensitivity
- **Categorical data**: Randomized response mechanism
- **Budget allocation**: Optimized 85% numerical, 15% categorical

## 📈 Privacy-Utility Analysis

The `agedcare_tradeoff_analysis.py` provides comprehensive evaluation of privacy-utility trade-offs:

### Analysis Features
- **Multi-metric evaluation**: MAE-based utility, correlation preservation, relative error
- **Visual trade-off charts**: Privacy vs utility curves across epsilon values  
- **Statistical validation**: Distribution comparisons and fidelity metrics
- **Automated reporting**: Detailed analysis summaries and recommendations

### Key Metrics
1. **MAE Utility**: `1 - (MAE / variable_range)` - measures value preservation
2. **Correlation Utility**: Pearson correlation between baseline and DP data
3. **Relative Error**: `1 - (MAE / baseline_mean)` - normalized accuracy measure

### Generated Outputs
- **Privacy-utility charts**: `analysis/agedcare_privacy_utility_chart.png`
- **Detailed reports**: Analysis summaries with utility scores per epsilon
- **Comparison tables**: Statistical fidelity across all privacy levels

## 📊 Dataset Structure

### Aged Care Dataset Variables

#### Demographics
- `person_id`: De-identified person identifier  
- `age_at_admission`: Age entering care (65-105)
- `sex`: Gender (M/F/X) 
- `indigenous_status`: Indigenous background
- `postcode`: Residential postcode
- `seifa_decile`: Socioeconomic index (1-10)

#### Care & Health
- `care_level`: Required care level (1-4)
- `length_of_stay_days`: Duration in residential care
- `acfi_*_domain`: ACFI assessment scores
- `dementia_status`: Dementia diagnosis
- `medication_count`: Number of medications
- `chronic_conditions_count`: Chronic health conditions

## 💻 Usage Examples

### Generate Synthetic Data
```python
# Generate baseline and privacy-protected datasets
from src.agedcare_synthetic_dp import AgedCareSynthesizer
import os

# Step 1: Generate baseline data (no privacy)
baseline_synthesizer = AgedCareSynthesizer(epsilon=0.0)
baseline_data = baseline_synthesizer.generate_base_demographics(n_records=3000)

# Step 2: Apply different privacy levels to same baseline
privacy_levels = [0.5, 1.0, 2.0]  # High, moderate, lower privacy
datasets = {'baseline': baseline_data}

for epsilon in privacy_levels:
    privacy_synthesizer = AgedCareSynthesizer(epsilon=epsilon, delta=1e-5)
    dp_data = privacy_synthesizer.apply_differential_privacy(baseline_data.copy())
    datasets[f'epsilon_{epsilon}'] = dp_data
    
    # Save DP dataset
    dp_data.to_csv(f'data/agedcare_synthetic_dp_{epsilon}_3000.csv', index=False)

# Save baseline
baseline_data.to_csv('data/agedcare_baseline_3000.csv', index=False)
print(f"Generated {len(datasets)} datasets with varying privacy levels")
```

### Privacy-Utility Analysis
```python
# Comprehensive analysis of privacy-utility trade-offs
from src.agedcare_tradeoff_analysis import AgedCareTradeoffAnalyzer

analyzer = AgedCareTradeoffAnalyzer()
analyzer.run_analysis()  # Generates charts and utility metrics
```

### Load and Analyze Data
```python
# Load datasets for custom analysis
import pandas as pd

# Load baseline and differentially private data
baseline_df = pd.read_csv('data/agedcare_baseline_3000.csv')
dp_05_df = pd.read_csv('data/agedcare_synthetic_dp_0.5_3000.csv')  # High privacy
dp_10_df = pd.read_csv('data/agedcare_synthetic_dp_1.0_3000.csv')  # Balanced
dp_20_df = pd.read_csv('data/agedcare_synthetic_dp_2.0_3000.csv')  # Lower privacy

# Compare key demographics
print(f"Baseline avg age: {baseline_df['age_at_admission'].mean():.1f}")
print(f"DP ε=1.0 avg age: {dp_10_df['age_at_admission'].mean():.1f}")
print(f"Care level preservation: {(dp_10_df['care_level'].mode()[0] == baseline_df['care_level'].mode()[0])}")
```

## 🔐 Zero-Knowledge Access Control

The `agedcare_zk_snarks.py` demonstrates **ZK-SNARKs** (Zero-Knowledge Succinct Non-Interactive Arguments of Knowledge) for privacy-preserving access control to resident data.

### Key Features
- **Zero-knowledge proof of access rights** - Staff prove they have valid credentials without revealing them
- **Policy-based access control** - Resident data protected by clearance level, department, and role requirements  
- **Cryptographic verification** - Mathematically secure proof verification without credential exposure
- **Complete audit trail** - All access attempts logged while preserving privacy
- **Consent management** - Time-limited access with automatic expiration

### ZK-SNARKs Workflow
1. **Access Request**: Staff member requests access to resident data
2. **Challenge Generation**: System creates cryptographic challenge based on data access policy
3. **Proof Computation**: Staff client generates ZK-SNARK proof of valid credentials (without revealing them)
4. **Proof Verification**: System verifies proof using cryptographic pairing operations
5. **Access Grant**: If proof valid, decrypt and provide data + log access

### Usage Example
```python
# Zero-knowledge access control demonstration
from src.agedcare_zk_snarks import AgedCareZKSystem

# Initialize ZK access control system
zk_system = AgedCareZKSystem()

# Register staff with credentials (stored securely)
zk_system.register_staff('dr_smith', 'doctor', clearance_level=5, department='medical')
zk_system.register_staff('nurse_alice', 'nurse', clearance_level=3, department='medical')

# Store resident data with access policy
resident_data = {
    'name': 'Margaret Johnson',
    'medical_conditions': ['diabetes', 'hypertension'],
    'care_plan': 'assisted living with monitoring'
}

zk_system.store_resident_data(
    resident_id='resident_001',
    data=resident_data,
    required_clearance=4,
    allowed_departments=['medical'],
    allowed_roles=['doctor', 'senior_nurse']
)

# ZK-SNARKs access workflow
challenge = zk_system.generate_access_challenge('resident_001')
proof = zk_system.compute_zk_proof('dr_smith', challenge)
accessed_data = zk_system.grant_access_and_log('dr_smith', 'resident_001', proof, challenge, 'medical_review')
```

### Benefits for Aged Care
- **Staff Privacy**: Credentials never exposed during verification process
- **Resident Protection**: Multi-layered access control with cryptographic guarantees
- **Compliance**: Detailed audit logs for regulatory requirements
- **Scalability**: Verification complexity independent of credential database size
- **Trust**: Mathematical proof of authorization without revealing sensitive information


## 📈 Research Applications

### Suitable Research Uses ✅
- **Population health studies** - Demographic and health trend analysis  
- **Healthcare policy research** - Care level requirements and resource planning
- **Machine learning** - Privacy-preserving model development and validation
- **Differential privacy research** - Privacy-utility trade-off evaluation
- **Zero-knowledge cryptography** - Access control and authentication systems
- **Educational purposes** - Teaching privacy-preserving data analysis and cryptographic protocols

### Important Limitations ❌  
- **Individual analysis** - Records don't represent real people
- **Longitudinal studies** - Generated at single time point
- **Rare conditions** - May not capture very low-prevalence cases
- **External linkage** - Cannot be linked to other datasets

## 🛠️ Development & Contribution

### Repository Structure
```
aged-care-synthetic-data/
├── src/                              # Source code
│   ├── agedcare_synthetic_dp.py          # Data generator with differential privacy
│   ├── agedcare_tradeoff_analysis.py     # Privacy-utility analysis
│   └── agedcare_zk_snarks.py             # Zero-knowledge access control
├── data/                             # Generated datasets
│   ├── agedcare_baseline_3000.csv       # Original synthetic data
│   ├── agedcare_synthetic_dp_0.5_3000.csv  # High privacy
│   ├── agedcare_synthetic_dp_1.0_3000.csv  # Balanced
│   └── agedcare_synthetic_dp_2.0_3000.csv  # Lower privacy
├── analysis/                         # Analysis results and charts
├── requirements.txt                  # Dependencies
└── setup.py                          # Automated setup
```

### Git Workflow
```bash
# Clone and setup
git clone https://github.com/taiyoo/nacdc-synthetic-data.git
cd nacdc-synthetic-data
python setup.py

# Make changes and commit
git add .
git commit -m "Description of changes"
git push origin main
```

### Contributing Guidelines
1. **Privacy preservation**: Maintain formal differential privacy guarantees
2. **Utility optimization**: Improve sensitivity calibration and noise mechanisms
3. **Analysis enhancement**: Add new utility metrics and evaluation methods
4. **Code quality**: Include tests and documentation for new features

## ⚖️ Ethics & Legal

### Privacy Guarantees
- **Formal differential privacy** provides mathematical privacy protection
- **Optimized utility preservation** through careful sensitivity tuning
- **Safe for research use** without individual privacy concerns

### Usage Responsibilities  
- Synthetic data only - not for clinical decisions
- Cannot represent real individuals or organizations  
- Follow institutional research ethics guidelines
- Acknowledge synthetic nature in research publications

## 📚 References & Citation

### Academic References
- Dwork, C., & Roth, A. (2014). The algorithmic foundations of differential privacy
- AIHW (2024). National Aged Care Data Clearinghouse data dictionary

### Citation
```bibtex
@software{agedcare_synthetic_2025,
  title = {Aged Care Synthetic Data Generator with Differential Privacy},
  author = {CSEC5614 Research Team},
  year = {2025},
  url = {https://github.com/taiyoo/nacdc-synthetic-data},
  note = {Privacy-utility optimized synthetic data for aged care research}
}
```

**⚠️ Disclaimer**: This synthetic data is generated for research purposes only. It does not represent real individuals and should not be used for clinical decision-making or individual care planning.
