# Data Directory

## 📁 Folder Structure

### `synthetic/`
Contains generated synthetic datasets with differential privacy protection.

**Files:**
- `nacdc_synthetic_dp_0.5_3000.csv` - High privacy (ε=0.5), 3,000 records
- `nacdc_synthetic_dp_1.0_3000.csv` - Moderate privacy (ε=1.0), 3,000 records  
- `nacdc_synthetic_dp_2.0_3000.csv` - Lower privacy (ε=2.0), 3,000 records

**Privacy Levels:**
- **ε = 0.5**: Strongest privacy protection, suitable for sensitive research
- **ε = 1.0**: Balanced privacy-utility, recommended for most applications
- **ε = 2.0**: Higher utility, suitable for general research

### `specifications/`
Contains original data specifications and documentation.

**Files:**
- `NACDC-table-specifcations-August-2025.xlsx` - Official NACDC table specifications

## 📊 Dataset Schema

Each synthetic dataset contains 29 variables:

### Demographics (6 variables)
- `person_id`: De-identified person identifier
- `age_at_admission`: Age when entering care (65-105)
- `sex`: Gender (M/F/X)
- `indigenous_status`: Indigenous status
- `country_of_birth`: Country of birth
- `preferred_language`: Preferred language

### Care Information (8 variables)
- `provider_id`: De-identified provider
- `service_id`: De-identified service
- `admission_date`: Admission date
- `discharge_date`: Discharge date (null if still in care)
- `care_level`: Care level (1-4)
- `accommodation_type`: Permanent/Respite
- `discharge_reason`: Reason for discharge
- `length_of_stay_days`: Duration in care

### Health Assessment (9 variables)
- `acfi_care_domain`: ACFI care score (0-100)
- `acfi_accommodation_domain`: ACFI accommodation score (0-100)
- `acfi_complex_health_care`: ACFI complex health score (0-100)
- `dementia_status`: Dementia diagnosis (Yes/No/Not assessed)
- `falls_risk`: Falls risk level (High/Medium/Low)
- `medication_count`: Number of medications (0-20)
- `chronic_conditions_count`: Number of chronic conditions (0-10)
- `mobility_assistance`: Mobility assistance level
- `personal_care_assistance`: Personal care assistance level

### Geographic (3 variables)
- `postcode`: Postcode of residence
- `remoteness_area`: Geographic remoteness classification
- `seifa_decile`: Socioeconomic index (1-10)

### Metadata (3 variables)
- `record_generated_date`: Date of synthetic record generation
- `privacy_applied`: Boolean indicating if differential privacy was applied
- `epsilon_used`: Privacy budget used (if applicable)

## 🔒 Privacy Protection

All synthetic datasets provide formal differential privacy guarantees:
- **Mathematical privacy protection** regardless of auxiliary information
- **No risk of re-identification** from synthetic records
- **Suitable for public sharing** and research collaboration
- **Maintains statistical utility** for population-level analysis

## 📈 Usage Examples

### Load Dataset
```python
import pandas as pd

# Load moderate privacy dataset
df = pd.read_csv('data/synthetic/nacdc_synthetic_dp_1.0_3000.csv')
print(f"Dataset shape: {df.shape}")
```

### Basic Analysis
```python
# Demographic overview
print(f"Average age: {df['age_at_admission'].mean():.1f}")
print(f"Sex distribution:\n{df['sex'].value_counts()}")
print(f"Care levels:\n{df['care_level'].value_counts()}")
```

### Research Applications
```python
# Analyze care patterns
care_analysis = df.groupby('care_level').agg({
    'length_of_stay_days': 'mean',
    'medication_count': 'mean',
    'dementia_status': lambda x: (x == 'Yes').mean()
})
print(care_analysis)
```
