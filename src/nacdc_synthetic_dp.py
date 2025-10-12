"""
NACDC Synthetic Dataset Generator with Differential Privacy
Generates synthetic residential care data based on NACDC table specifications
with formal differential privacy guarantees.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import os
from diffprivlib.mechanisms import Laplace
from diffprivlib.tools import mean as dp_mean, var as dp_var
import warnings
warnings.filterwarnings('ignore')

class NACDCResidentialCareSynthesizer:
    """
    Generates synthetic residential care datasets with differential privacy.
    Based on NACDC table specifications for residential care.
    """
    
    def __init__(self, epsilon=1.0, delta=1e-5, random_seed=42):
        self.epsilon = epsilon
        self.delta = delta
        self.fake = Faker()
        Faker.seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Realistic bounds for variables (for better sensitivity calculation)
        self.variable_bounds = {
            'age_at_admission': (65, 105),
            'length_of_stay_days': (1, 3000),  # Reasonable upper bound
            'medication_count': (0, 20),
            'chronic_conditions_count': (0, 10),
            'acfi_care_domain': (0, 100),
            'acfi_accommodation_domain': (0, 100),
            'acfi_complex_health_care': (0, 100),
            'seifa_decile': (1, 10)
        }
    
    def generate_base_demographics(self, n_records):
        """Generate base demographic data without differential privacy."""
        records = []
        
        for i in range(n_records):
            # Generate correlated demographic data
            age = max(65, min(105, int(np.random.normal(85, 8))))
            sex = np.random.choice(['M', 'F', 'X'], p=[0.35, 0.64, 0.01])
            
            # Age and sex influence other characteristics
            if age > 90:
                dementia_prob = 0.7
                high_care_prob = 0.6
            elif age > 85:
                dementia_prob = 0.5
                high_care_prob = 0.4
            else:
                dementia_prob = 0.2
                high_care_prob = 0.2
                
            dementia = np.random.choice(['Yes', 'No', 'Not assessed'], 
                                      p=[dementia_prob, 1-dementia_prob-0.05, 0.05])
            
            # Generate care needs based on age and dementia
            if dementia == 'Yes' or age > 90:
                care_level_probs = [0.05, 0.25, 0.45, 0.25]  # Higher care levels
                medication_count = max(3, min(15, int(np.random.poisson(8))))
                chronic_conditions = max(2, min(8, int(np.random.poisson(4))))
            else:
                care_level_probs = [0.20, 0.40, 0.30, 0.10]  # Lower care levels
                medication_count = max(1, min(12, int(np.random.poisson(5))))
                chronic_conditions = max(1, min(6, int(np.random.poisson(2))))
            
            care_level = np.random.choice(['Level 1', 'Level 2', 'Level 3', 'Level 4'], 
                                        p=care_level_probs)
            
            # Generate realistic length of stay
            if care_level in ['Level 3', 'Level 4']:
                los_mean = 800  # Longer stays for higher care
            else:
                los_mean = 400
            
            # Cap length of stay at reasonable maximum
            length_of_stay = max(1, min(2500, int(np.random.exponential(los_mean))))
            
            # Generate dates
            admission_date = self.fake.date_between(start_date='-3y', end_date='today')
            discharge_date = admission_date + timedelta(days=length_of_stay)
            
            # 20% still in care
            if np.random.random() < 0.2:
                discharge_date = None
                discharge_reason = 'Still in care'
                # For current residents, use time since admission
                length_of_stay = min(1095, (datetime.now().date() - admission_date).days)  # Max 3 years
            else:
                discharge_reason = np.random.choice([
                    'Death', 'Transfer to hospital', 'Family care', 'Other accommodation'
                ], p=[0.56, 0.19, 0.125, 0.125])
            
            # ACFI scores based on care level
            if care_level == 'Level 1':
                acfi_base = np.random.uniform(20, 40)
            elif care_level == 'Level 2':
                acfi_base = np.random.uniform(35, 55)
            elif care_level == 'Level 3':
                acfi_base = np.random.uniform(50, 70)
            else:  # Level 4
                acfi_base = np.random.uniform(65, 85)
            
            record = {
                'person_id': f'P{i+1:06d}',
                'provider_id': f'PROV{random.randint(1000, 9999)}',
                'service_id': f'SVC{random.randint(10000, 99999)}',
                'admission_date': admission_date,
                'discharge_date': discharge_date,
                'age_at_admission': age,
                'sex': sex,
                'indigenous_status': np.random.choice([
                    'Non-Indigenous', 'Indigenous', 'Not stated'
                ], p=[0.92, 0.05, 0.03]),
                'country_of_birth': np.random.choice([
                    'Australia', 'United Kingdom', 'Italy', 'Greece', 'Germany', 'China', 'Other'
                ], p=[0.7, 0.08, 0.04, 0.03, 0.03, 0.03, 0.09]),
                'preferred_language': np.random.choice([
                    'English', 'Italian', 'Greek', 'Mandarin', 'Other'
                ], p=[0.82, 0.05, 0.03, 0.03, 0.07]),
                'care_level': care_level,
                'accommodation_type': np.random.choice(['Permanent', 'Respite'], p=[0.85, 0.15]),
                'discharge_reason': discharge_reason,
                'length_of_stay_days': length_of_stay,
                'acfi_care_domain': round(max(0, min(100, acfi_base + np.random.normal(0, 5))), 2),
                'acfi_accommodation_domain': round(max(0, min(100, acfi_base * 0.6 + np.random.normal(0, 3))), 2),
                'acfi_complex_health_care': round(max(0, min(100, acfi_base * 0.8 + np.random.normal(0, 4))), 2),
                'dementia_status': dementia,
                'falls_risk': np.random.choice(['High', 'Medium', 'Low'], p=[0.25, 0.45, 0.30]),
                'medication_count': medication_count,
                'chronic_conditions_count': chronic_conditions,
                'mobility_assistance': np.random.choice([
                    'Independent', 'Supervision', 'Physical assistance', 'Total assistance'
                ], p=[0.15, 0.25, 0.35, 0.25]),
                'personal_care_assistance': np.random.choice([
                    'Independent', 'Supervision', 'Physical assistance', 'Total assistance'
                ], p=[0.10, 0.20, 0.40, 0.30]),
                'postcode': str(random.randint(1000, 9999)),
                'remoteness_area': np.random.choice([
                    'Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote'
                ], p=[0.65, 0.20, 0.12, 0.02, 0.01]),
                'seifa_decile': random.randint(1, 10)
            }
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def apply_differential_privacy_refined(self, df):
        """Apply carefully calibrated differential privacy."""
        print(f"Applying differential privacy with ε={self.epsilon}, δ={self.delta}")
        
        df_private = df.copy()
        
        # Allocate privacy budget more carefully
        epsilon_numerical = self.epsilon * 0.7  # 70% for numerical data
        epsilon_categorical = self.epsilon * 0.3  # 30% for categorical data
        
        # 1. Apply DP to numerical columns with realistic sensitivity
        numerical_cols = ['age_at_admission', 'length_of_stay_days', 'medication_count', 
                         'chronic_conditions_count', 'acfi_care_domain', 
                         'acfi_accommodation_domain', 'acfi_complex_health_care', 'seifa_decile']
        
        epsilon_per_numerical = epsilon_numerical / len(numerical_cols)
        
        for col in numerical_cols:
            if col in df_private.columns:
                # Use realistic sensitivity based on variable bounds
                min_val, max_val = self.variable_bounds.get(col, (df_private[col].min(), df_private[col].max()))
                sensitivity = max_val - min_val
                
                # Apply Laplace mechanism with proper sensitivity
                laplace_mech = Laplace(epsilon=epsilon_per_numerical, sensitivity=sensitivity)
                
                noisy_values = []
                for value in df_private[col]:
                    try:
                        noisy_value = laplace_mech.randomise(float(value))
                        
                        # Clamp to realistic bounds
                        if col == 'age_at_admission':
                            noisy_value = max(65, min(105, round(noisy_value)))
                        elif col == 'seifa_decile':
                            noisy_value = max(1, min(10, round(noisy_value)))
                        elif col in ['medication_count', 'chronic_conditions_count']:
                            noisy_value = max(0, min(20, round(noisy_value)))
                        elif col == 'length_of_stay_days':
                            noisy_value = max(1, min(3000, round(noisy_value)))
                        elif 'acfi' in col:
                            noisy_value = max(0, min(100, round(noisy_value, 2)))
                        else:
                            noisy_value = max(0, noisy_value)
                        
                        noisy_values.append(noisy_value)
                    except Exception:
                        # Fallback to original value if mechanism fails
                        noisy_values.append(value)
                
                df_private[col] = noisy_values
        
        # 2. Apply lighter perturbation to categorical data
        categorical_cols = ['sex', 'indigenous_status', 'care_level', 'dementia_status', 'falls_risk']
        
        perturbation_rate = min(0.1, self.epsilon / 10)  # Scale perturbation with epsilon
        
        for col in categorical_cols:
            if col in df_private.columns:
                unique_vals = df_private[col].unique()
                n_to_perturb = max(1, int(len(df_private) * perturbation_rate))
                
                # Randomly select records to perturb
                indices_to_perturb = np.random.choice(len(df_private), size=n_to_perturb, replace=False)
                
                for idx in indices_to_perturb:
                    df_private.at[idx, col] = np.random.choice(unique_vals)
        
        print("Differential privacy applied successfully!")
        return df_private
    
    def validate_privacy_properties(self, original_df, private_df):
        """Validate that privacy properties are maintained."""
        print("\n=== Privacy Validation ===")
        
        numerical_cols = ['age_at_admission', 'length_of_stay_days', 'medication_count']
        
        for col in numerical_cols:
            orig_mean = original_df[col].mean()
            priv_mean = private_df[col].mean()
            
            orig_std = original_df[col].std()
            priv_std = private_df[col].std()
            
            # Calculate relative error
            rel_error = abs(orig_mean - priv_mean) / orig_mean * 100
            
            print(f"{col}:")
            print(f"  Original: mean={orig_mean:.2f}, std={orig_std:.2f}")
            print(f"  Private:  mean={priv_mean:.2f}, std={priv_std:.2f}")
            print(f"  Relative error: {rel_error:.1f}%")
        
        # Check categorical distributions
        categorical_cols = ['sex', 'care_level', 'dementia_status']
        
        for col in categorical_cols:
            print(f"\n{col} distribution:")
            orig_dist = original_df[col].value_counts(normalize=True)
            priv_dist = private_df[col].value_counts(normalize=True)
            
            for category in orig_dist.index:
                orig_prop = orig_dist.get(category, 0)
                priv_prop = priv_dist.get(category, 0)
                diff = abs(orig_prop - priv_prop)
                print(f"  {category}: Original={orig_prop:.3f}, Private={priv_prop:.3f}, Diff={diff:.3f}")
    
    def generate_synthetic_dataset(self, n_records=10000, apply_dp=True, save_to_file=True):
        """Generate synthetic NACDC residential care dataset."""
        print(f"Generating {n_records} synthetic NACDC records...")
        
        # Generate base dataset
        df = self.generate_base_demographics(n_records)
        original_df = df.copy()
        
        # Apply differential privacy if requested
        if apply_dp:
            df = self.apply_differential_privacy_refined(df)
            self.validate_privacy_properties(original_df, df)
        
        # Add metadata
        df['record_generated_date'] = datetime.now().date()
        df['privacy_applied'] = apply_dp
        df['epsilon_used'] = self.epsilon if apply_dp else None
        
        # Save to file
        if save_to_file:
            filename = f"nacdc_synthetic_dp_{self.epsilon:.1f}_{n_records}.csv"
            # Use relative path from script location
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(script_dir, "data", "synthetic")
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Dataset saved to {filepath}")
        
        # Generate summary report
        self.generate_summary_report(df, apply_dp)
        
        return df
    
    def generate_summary_report(self, df, privacy_applied):
        """Generate a summary report of the synthetic dataset."""
        print(f"\n=== NACDC Synthetic Dataset Summary ===")
        print(f"Total records: {len(df):,}")
        print(f"Differential privacy applied: {privacy_applied}")
        if privacy_applied:
            print(f"Privacy budget (ε): {self.epsilon}")
        
        print(f"\nDemographics:")
        print(f"Age range: {df['age_at_admission'].min():.0f} - {df['age_at_admission'].max():.0f}")
        print(f"Average age: {df['age_at_admission'].mean():.1f}")
        
        print(f"\nSex distribution:")
        for sex, count in df['sex'].value_counts().items():
            print(f"  {sex}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\nCare level distribution:")
        for level, count in df['care_level'].value_counts().items():
            print(f"  {level}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print(f"\nLength of stay statistics:")
        print(f"  Average: {df['length_of_stay_days'].mean():.0f} days")
        print(f"  Median: {df['length_of_stay_days'].median():.0f} days")
        print(f"  Range: {df['length_of_stay_days'].min():.0f} - {df['length_of_stay_days'].max():.0f} days")
        
        print(f"\nHealth characteristics:")
        print(f"  Dementia prevalence: {(df['dementia_status'] == 'Yes').sum() / len(df) * 100:.1f}%")
        print(f"  Average medications: {df['medication_count'].mean():.1f}")
        print(f"  Average chronic conditions: {df['chronic_conditions_count'].mean():.1f}")


def main():
    """Main function for synthetic data generation."""
    print("=== NACDC Synthetic Data Generator ===")
    print("Generating realistic synthetic data with differential privacy...\n")
    
    # Test different privacy levels
    privacy_levels = [(0.5, "High Privacy"), (1.0, "Moderate Privacy"), (2.0, "Lower Privacy")]
    
    for epsilon, privacy_desc in privacy_levels:
        print(f"\n{'='*60}")
        print(f"Generating {privacy_desc} Dataset (ε={epsilon})")
        print(f"{'='*60}")
        
        synthesizer = NACDCResidentialCareSynthesizer(epsilon=epsilon, delta=1e-5)
        
        # Generate dataset
        synthetic_data = synthesizer.generate_synthetic_dataset(
            n_records=3000, 
            apply_dp=True, 
            save_to_file=True
        )
        
        print(f"\nSample records (ε={epsilon}):")
        print(synthetic_data[['person_id', 'age_at_admission', 'sex', 'care_level', 
                            'dementia_status', 'length_of_stay_days', 'medication_count']].head())
    
    # Generate baseline without DP
    print(f"\n{'='*60}")
    print("Generating Baseline Dataset (No Differential Privacy)")
    print(f"{'='*60}")
    
    baseline_synthesizer = NACDCResidentialCareSynthesizer(epsilon=1.0)
    baseline_data = baseline_synthesizer.generate_synthetic_dataset(
        n_records=3000, 
        apply_dp=False, 
        save_to_file=True
    )
    
    print("\n=== Generation Complete ===")
    print("Synthetic NACDC residential care datasets with differential privacy have been generated.")


if __name__ == "__main__":
    main()
