# NACDC Synthetic Data Generation Project

## Overview

This repository contains **two distinct synthetic data generators** for Australian aged care data with formal differential privacy protection:

1. **🔬 Research Dataset Generator** (`src/nacdc_synthetic_dp.py`) - Simplified, research-focused data
2. **🏛️ Government-Compliant Generator** (`src/government_compliant_generator.py`) - **Following NACDC submission format**

### Key Features
- **Official Tables**: MAIN_RECIPIENT, RAC_EPISODE, RAC_ASSESSMENT_ACFI
- **Government Codes**: Uses actual NACDC variable names and code sets
- **Full Compliance**: Meets all 10/10 government submission requirements
- **Privacy Protected**: Formal differential privacy (ε = 1.0, 2.0, 5.0)
- **Realistic Data**: Australian demographic distributions, proper ACFI assessments

### Generated Tables
1. **MAIN_RECIPIENT** (18 variables) - Official demographics with AIHW_PPN, government codes
2. **RAC_EPISODE** (15 variables) - Care episodes with proper date relationships  
3. **RAC_ASSESSMENT_ACFI** (27+ variables) - Detailed ACFI assessments with Q01-Q14

**📂 Location**: `data/government_compliant/`  
**📖 Documentation**: See `data/government_compliant/README.md`

## � Quick Start

### Installation and Setup

```bash
# Clone the repository
git clone https://github.com/taiyoo/nacdc-synthetic-data
cd csec5614

# Run automated setup (creates virtual environment and installs dependencies)
python setup.py

# Or manually:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate Government-Compliant Data
```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Generate official government submission data
python src/government_compliant_generator.py

# Analyze compliance and quality
python src/government_compliance_analyzer.py
```

### Generate Research Data
```bash
# Generate simplified research datasets
python src/nacdc_synthetic_dp.py

# Analyze research datasets
python src/nacdc_analysis.py
```

## �🔬 Research Dataset Generator (Original)

Simplified synthetic data for research and academic purposes.

## 🔒 Privacy Protection

Both generators implement **differential privacy**, providing formal privacy guarantees:

### Privacy Parameters
- **ε (epsilon)**: Privacy budget - lower values provide stronger privacy protection
  - ε = 0.1-0.5: Very high privacy (recommended for sensitive research)
  - ε = 1.0: High privacy (good balance for most applications)
  - ε = 2.0-5.0: Moderate privacy (suitable for general research)
  - ε = 10.0+: Lower privacy (minimal protection, high utility)

- **δ (delta)**: Probability of privacy failure (set to 1e-5)

## 📊 Dataset Structure

The synthetic datasets follow the NACDC residential care data schema and include:

### Demographics
- `person_id`: De-identified person identifier
- `age_at_admission`: Age when entering residential care (65-105)
- `sex`: Gender (M/F/X)
- `indigenous_status`: Indigenous status
- `country_of_birth`: Country of birth
- `preferred_language`: Preferred language

### Care Information
- `provider_id`: De-identified care provider
- `service_id`: De-identified service identifier
- `admission_date`: Date of admission to residential care
- `discharge_date`: Date of discharge (null if still in care)
- `care_level`: Care level required (Level 1-4)
- `accommodation_type`: Permanent or Respite care
- `length_of_stay_days`: Duration in residential care

### Health & Assessment
- `acfi_care_domain`: Aged Care Funding Instrument - Care score
- `acfi_accommodation_domain`: ACFI - Accommodation score
- `acfi_complex_health_care`: ACFI - Complex health care score
- `dementia_status`: Dementia diagnosis status
- `falls_risk`: Risk of falls assessment
- `medication_count`: Number of medications
- `chronic_conditions_count`: Number of chronic conditions
- `mobility_assistance`: Level of mobility assistance required
- `personal_care_assistance`: Level of personal care assistance

### Geographic
- `postcode`: Postcode of residence
- `remoteness_area`: Geographic remoteness classification
- `seifa_decile`: Socioeconomic index (1-10)

## 🚀 Usage Instructions

### 1. Generate Synthetic Data

```python
from nacdc_synthetic_dp import NACDCResidentialCareSynthesizer

# Initialize synthesizer with desired privacy level
synthesizer = NACDCResidentialCareSynthesizer(epsilon=1.0, delta=1e-5)

# Generate synthetic dataset
synthetic_data = synthesizer.generate_synthetic_dataset(
    n_records=10000,
    apply_dp=True,
    save_to_file=True
)
```

### 2. Analyze Generated Datasets

```python
from nacdc_analysis import NACDCDatasetAnalyzer

# Initialize analyzer
analyzer = NACDCDatasetAnalyzer()

# Load and compare all generated datasets
datasets = analyzer.load_datasets()

# Compare distributions across privacy levels
analyzer.compare_distributions(datasets)

# Analyze privacy-utility tradeoff
analyzer.privacy_utility_analysis(datasets)

# Generate quality assessment
analyzer.generate_data_quality_report(datasets)
```

### 3. Load Dataset for Analysis

```python
import pandas as pd

# Load a specific dataset
df = pd.read_csv('nacdc_synthetic_dp_1.0_3000.csv')

# Basic statistics
print(df.describe())

# Demographic analysis
print(df['care_level'].value_counts())
print(f"Average age: {df['age_at_admission'].mean():.1f}")
print(f"Dementia prevalence: {(df['dementia_status'] == 'Yes').mean()*100:.1f}%")
```

## 📈 Research Applications

### Suitable Uses
✅ **Epidemiological research** - Population-level health trends  
✅ **Policy analysis** - Care level distributions and requirements  
✅ **Machine learning** - Predictive model development  
✅ **Resource planning** - Capacity and staffing analysis  
✅ **Quality improvement** - Care pathway optimization  
✅ **Education** - Training and teaching datasets  

### Limitations
❌ **Individual-level analysis** - Records don't represent real people  
❌ **Linkage studies** - Cannot be linked to external datasets  
❌ **Rare conditions** - May not accurately represent very rare cases  
❌ **Temporal trends** - Generated at single time point  

## 🔍 Quality Validation

Each generated dataset includes validation metrics:

- **Statistical Fidelity**: Comparison of distributions with baseline
- **Privacy Protection**: Formal differential privacy guarantees
- **Data Quality**: Missing values, range validation, logical consistency
- **Utility Preservation**: Maintenance of research-relevant patterns

## 📁 Generated Files

The system generates several files:

1. **Synthetic datasets**: `nacdc_synthetic_dp_{epsilon}_{n_records}.csv`
2. **Analysis plots**: Distribution comparisons and privacy-utility tradeoffs
3. **Quality report**: `nacdc_analysis_report.txt`
4. **Visualization**: Demographic and statistical analysis plots

## ⚖️ Ethical Considerations

This synthetic data:
- Provides formal privacy protection through differential privacy
- Enables research without compromising individual privacy
- Maintains statistical utility for population-level analysis
- Should not be used to infer information about real individuals

## 📚 Technical Details

### Differential Privacy Implementation
- Uses Laplace mechanism for numerical variables
- Calibrated sensitivity based on realistic variable bounds
- Categorical perturbation with privacy budget allocation
- Validation against baseline distributions

### Data Generation Process
1. **Base generation**: Realistic correlated demographic and health data
2. **Privacy application**: Differential privacy mechanisms with calibrated noise
3. **Validation**: Statistical and logical consistency checks
4. **Quality assessment**: Utility preservation and privacy validation

## 🤝 Contributing

To improve the synthetic data generator:
1. Enhance realism of generated correlations
2. Add new variables from NACDC specifications
3. Implement additional privacy mechanisms
4. Improve validation and quality metrics

## 📞 Support

For questions about:
- **Privacy guarantees**: Consult differential privacy literature
- **NACDC data**: Visit https://www.aihw.gov.au/reports-data/nacda/data
- **Implementation**: Review code documentation and comments

## 📄 Citation

If using this synthetic data for research, please cite:
- The NACDC data source: Australian Institute of Health and Welfare
- Differential privacy methodology: Dwork & Roth (2014)
- This synthetic data generator implementation

---

**Disclaimer**: This is synthetic data generated for research purposes. It does not represent real individuals and should not be used for clinical decision-making or individual care planning.
