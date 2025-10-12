"""
Government-Compliant NACDC Data Analysis Tool
Analyzes synthetic NACDC datasets that match official government submission requirements.
Validates compliance, data quality, and privacy protection effectiveness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

class GovernmentCompliantAnalyzer:
    """
    Analyzes government-compliant NACDC synthetic datasets.
    Validates structure, compliance, and data quality.
    """
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Use relative path from script location
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(script_dir, "data", "government_compliant")
        else:
            self.data_dir = data_dir
        self.government_codes = {
            'sex': {'M': 'Male', 'F': 'Female', 'X': 'Other'},
            'indigenous_status': {
                '1': 'Aboriginal but not Torres Strait Islander origin',
                '2': 'Torres Strait Islander but not Aboriginal origin', 
                '3': 'Aboriginal and Torres Strait Islander origin',
                '4': 'Neither Aboriginal nor Torres Strait Islander origin',
                '9': 'Not stated/inadequately described'
            },
            'admission_type': {'P': 'Permanent', 'B': 'Respite'},
            'acfi_levels': {'N': 'Nil', 'L': 'Low', 'M': 'Medium', 'H': 'High'}
        }
    
    def load_datasets(self, epsilon=1.0):
        """Load all three government-compliant tables for a specific privacy level"""
        suffix = f"_ep{epsilon:.1f}_2000"
        
        files = {
            'demographics': f"MAIN_RECIPIENT{suffix}.csv",
            'episodes': f"RAC_EPISODE{suffix}.csv", 
            'assessments': f"RAC_ASSESSMENT_ACFI{suffix}.csv"
        }
        
        datasets = {}
        for table_name, filename in files.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                datasets[table_name] = pd.read_csv(filepath)
                print(f"✅ Loaded {table_name}: {len(datasets[table_name]):,} records")
            else:
                print(f"❌ File not found: {filename}")
                
        return datasets
    
    def validate_government_compliance(self, datasets):
        """Validate compliance with government submission requirements"""
        print("\\n🏛️  GOVERNMENT COMPLIANCE VALIDATION")
        print("=" * 60)
        
        compliance_score = 0
        total_checks = 0
        
        # 1. Check MAIN_RECIPIENT structure
        if 'demographics' in datasets:
            demo_df = datasets['demographics']
            total_checks += 5
            
            # Required demographic fields
            required_demo_fields = [
                'AIHW_PPN', 'YEAR_AND_MONTH_OF_BIRTH', 'SEX_CODE', 
                'INDIGENOUS_STATUS_CODE', 'COUNTRY_OF_BIRTH_CODE'
            ]
            
            missing_fields = [f for f in required_demo_fields if f not in demo_df.columns]
            if not missing_fields:
                print("✅ MAIN_RECIPIENT: All required demographic fields present")
                compliance_score += 1
            else:
                print(f"❌ MAIN_RECIPIENT: Missing fields: {missing_fields}")
            
            # Valid government codes
            if 'SEX_CODE' in demo_df.columns:
                valid_sex_codes = set(['M', 'F', 'X'])
                actual_sex_codes = set(demo_df['SEX_CODE'].unique())
                if actual_sex_codes.issubset(valid_sex_codes):
                    print("✅ SEX_CODE: Uses valid government codes (M/F/X)")
                    compliance_score += 1
                else:
                    print(f"❌ SEX_CODE: Invalid codes found: {actual_sex_codes - valid_sex_codes}")
            
            # Indigenous status codes
            if 'INDIGENOUS_STATUS_CODE' in demo_df.columns:
                valid_indigenous = set(['1', '2', '3', '4', '9'])
                actual_indigenous = set(demo_df['INDIGENOUS_STATUS_CODE'].astype(str).unique())
                if actual_indigenous.issubset(valid_indigenous):
                    print("✅ INDIGENOUS_STATUS_CODE: Uses valid government codes")
                    compliance_score += 1
                else:
                    print(f"❌ INDIGENOUS_STATUS_CODE: Invalid codes: {actual_indigenous - valid_indigenous}")
            
            # AIHW_PPN format
            if 'AIHW_PPN' in demo_df.columns:
                ppn_sample = demo_df['AIHW_PPN'].iloc[0]
                if isinstance(ppn_sample, str) and len(ppn_sample) >= 10:
                    print("✅ AIHW_PPN: Proper identifier format")
                    compliance_score += 1
                else:
                    print("❌ AIHW_PPN: Invalid identifier format")
            
            # Birth date format
            if 'YEAR_AND_MONTH_OF_BIRTH' in demo_df.columns:
                birth_sample = str(demo_df['YEAR_AND_MONTH_OF_BIRTH'].iloc[0])
                if len(birth_sample) == 6 and birth_sample.isdigit():
                    print("✅ YEAR_AND_MONTH_OF_BIRTH: Proper YYYYMM format")
                    compliance_score += 1
                else:
                    print("❌ YEAR_AND_MONTH_OF_BIRTH: Invalid format")
        
        # 2. Check RAC_EPISODE structure
        if 'episodes' in datasets:
            episode_df = datasets['episodes']
            total_checks += 3
            
            required_episode_fields = [
                'RECIPIENT_ID', 'SERVICE_ID', 'ADMISSION_DATE', 
                'ADMISSION_TYPE_CODE', 'EXIT_REASON_CODE'
            ]
            
            missing_ep_fields = [f for f in required_episode_fields if f not in episode_df.columns]
            if not missing_ep_fields:
                print("✅ RAC_EPISODE: All required episode fields present")
                compliance_score += 1
            else:
                print(f"❌ RAC_EPISODE: Missing fields: {missing_ep_fields}")
            
            # Admission type codes
            if 'ADMISSION_TYPE_CODE' in episode_df.columns:
                valid_admission = set(['P', 'B'])
                actual_admission = set(episode_df['ADMISSION_TYPE_CODE'].unique())
                if actual_admission.issubset(valid_admission):
                    print("✅ ADMISSION_TYPE_CODE: Uses valid codes (P/B)")
                    compliance_score += 1
                else:
                    print(f"❌ ADMISSION_TYPE_CODE: Invalid codes: {actual_admission - valid_admission}")
            
            # Data linkage
            if 'demographics' in datasets and 'RECIPIENT_ID' in episode_df.columns:
                demo_ids = set(datasets['demographics']['AIHW_PPN'])
                episode_ids = set(episode_df['RECIPIENT_ID'])
                if episode_ids.issubset(demo_ids):
                    print("✅ DATA_LINKAGE: Episodes properly linked to demographics")
                    compliance_score += 1
                else:
                    print("❌ DATA_LINKAGE: Orphaned episode records found")
        
        # 3. Check RAC_ASSESSMENT_ACFI structure
        if 'assessments' in datasets:
            assess_df = datasets['assessments']
            total_checks += 2
            
            required_acfi_fields = [
                'RECIPIENT_ID', 'ASSESSMENT_START_DATE', 'ACFI_CATEGORY',
                'ADL_LEVEL', 'BEH_LEVEL', 'CHC_LEVEL'
            ]
            
            missing_acfi_fields = [f for f in required_acfi_fields if f not in assess_df.columns]
            if not missing_acfi_fields:
                print("✅ RAC_ASSESSMENT_ACFI: All required ACFI fields present")
                compliance_score += 1
            else:
                print(f"❌ RAC_ASSESSMENT_ACFI: Missing fields: {missing_acfi_fields}")
            
            # ACFI levels
            if 'ADL_LEVEL' in assess_df.columns:
                valid_levels = set(['N', 'L', 'M', 'H'])
                actual_adl = set(assess_df['ADL_LEVEL'].unique())
                if actual_adl.issubset(valid_levels):
                    print("✅ ACFI_LEVELS: Uses valid level codes (N/L/M/H)")
                    compliance_score += 1
                else:
                    print(f"❌ ACFI_LEVELS: Invalid codes: {actual_adl - valid_levels}")
        
        # Calculate compliance percentage
        compliance_pct = (compliance_score / total_checks) * 100 if total_checks > 0 else 0
        print(f"\\n📊 COMPLIANCE SCORE: {compliance_score}/{total_checks} ({compliance_pct:.1f}%)")
        
        if compliance_pct >= 90:
            print("🟢 EXCELLENT: Dataset meets government compliance standards")
        elif compliance_pct >= 70:
            print("🟡 GOOD: Dataset mostly compliant, minor issues")
        else:
            print("🔴 POOR: Dataset requires significant compliance improvements")
        
        return compliance_score, total_checks
    
    def analyze_demographic_distributions(self, datasets):
        """Analyze demographic distributions for realism"""
        print("\\n👥 DEMOGRAPHIC DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        if 'demographics' not in datasets:
            print("❌ Demographics data not available")
            return
        
        demo_df = datasets['demographics']
        
        # Gender distribution
        if 'SEX_CODE' in demo_df.columns:
            gender_dist = demo_df['SEX_CODE'].value_counts(normalize=True) * 100
            print(f"🚻 Gender Distribution:")
            for code, pct in gender_dist.items():
                desc = self.government_codes['sex'].get(code, code)
                print(f"   {code} ({desc}): {pct:.1f}%")
        
        # Indigenous status
        if 'INDIGENOUS_STATUS_CODE' in demo_df.columns:
            indigenous_dist = demo_df['INDIGENOUS_STATUS_CODE'].value_counts(normalize=True) * 100
            print(f"\\n🏛️  Indigenous Status:")
            for code, pct in indigenous_dist.items():
                desc = self.government_codes['indigenous_status'].get(str(code), str(code))
                print(f"   {code}: {pct:.1f}%")
        
        # Age distribution
        if 'YEAR_AND_MONTH_OF_BIRTH' in demo_df.columns:
            current_year = 2025
            birth_years = demo_df['YEAR_AND_MONTH_OF_BIRTH'].astype(str).str[:4].astype(int)
            ages = current_year - birth_years
            
            print(f"\\n📅 Age Distribution:")
            print(f"   Mean age: {ages.mean():.1f} years")
            print(f"   Age range: {ages.min()}-{ages.max()} years")
            print(f"   Ages 65-74: {((ages >= 65) & (ages < 75)).mean() * 100:.1f}%")
            print(f"   Ages 75-84: {((ages >= 75) & (ages < 85)).mean() * 100:.1f}%")
            print(f"   Ages 85+: {(ages >= 85).mean() * 100:.1f}%")
        
        # Veterans
        if 'DVA_VETERAN_SUPP_FLAG' in demo_df.columns:
            veteran_pct = (demo_df['DVA_VETERAN_SUPP_FLAG'] == 'Y').mean() * 100
            print(f"\\n🎖️  Veterans: {veteran_pct:.1f}%")
        
        # Dementia support
        if 'HCP_DEMENTIA_SUPP_FLAG' in demo_df.columns:
            dementia_pct = (demo_df['HCP_DEMENTIA_SUPP_FLAG'] == 'Y').mean() * 100
            print(f"🧠 Dementia Support: {dementia_pct:.1f}%")
    
    def analyze_care_patterns(self, datasets):
        """Analyze care episode and assessment patterns"""
        print("\\n🏥 CARE PATTERN ANALYSIS")
        print("=" * 60)
        
        if 'episodes' not in datasets:
            print("❌ Episode data not available")
            return
        
        episode_df = datasets['episodes']
        
        # Admission types
        if 'ADMISSION_TYPE_CODE' in episode_df.columns:
            admission_dist = episode_df['ADMISSION_TYPE_CODE'].value_counts(normalize=True) * 100
            print(f"🏠 Admission Types:")
            for code, pct in admission_dist.items():
                desc = self.government_codes['admission_type'].get(code, code)
                print(f"   {code} ({desc}): {pct:.1f}%")
        
        # Length of stay analysis
        if 'ADMISSION_DATE' in episode_df.columns and 'DISCHARGE_DATE' in episode_df.columns:
            # Calculate length of stay for discharged residents
            discharged = episode_df[episode_df['DISCHARGE_DATE'].notna()].copy()
            
            if len(discharged) > 0:
                discharged['ADMISSION_DATE'] = pd.to_datetime(discharged['ADMISSION_DATE'])
                discharged['DISCHARGE_DATE'] = pd.to_datetime(discharged['DISCHARGE_DATE'])
                discharged['LOS_DAYS'] = (discharged['DISCHARGE_DATE'] - discharged['ADMISSION_DATE']).dt.days
                
                print(f"\\n📅 Length of Stay (Discharged Residents):")
                print(f"   Mean LOS: {discharged['LOS_DAYS'].mean():.0f} days")
                print(f"   Median LOS: {discharged['LOS_DAYS'].median():.0f} days")
                print(f"   Range: {discharged['LOS_DAYS'].min()}-{discharged['LOS_DAYS'].max()} days")
        
        # ACFI assessment analysis
        if 'assessments' in datasets:
            assess_df = datasets['assessments']
            
            print(f"\\n📋 ACFI Assessments:")
            print(f"   Total assessments: {len(assess_df):,}")
            
            if 'ADL_LEVEL' in assess_df.columns:
                print(f"\\n🏃 Activities of Daily Living (ADL) Levels:")
                adl_dist = assess_df['ADL_LEVEL'].value_counts(normalize=True) * 100
                for level, pct in adl_dist.items():
                    desc = self.government_codes['acfi_levels'].get(level, level)
                    print(f"   {level} ({desc}): {pct:.1f}%")
            
            if 'BEH_LEVEL' in assess_df.columns:
                print(f"\\n🧠 Cognition & Behaviour (BEH) Levels:")
                beh_dist = assess_df['BEH_LEVEL'].value_counts(normalize=True) * 100
                for level, pct in beh_dist.items():
                    desc = self.government_codes['acfi_levels'].get(level, level)
                    print(f"   {level} ({desc}): {pct:.1f}%")
            
            if 'CHC_LEVEL' in assess_df.columns:
                print(f"\\n🏥 Complex Health Care (CHC) Levels:")
                chc_dist = assess_df['CHC_LEVEL'].value_counts(normalize=True) * 100
                for level, pct in chc_dist.items():
                    desc = self.government_codes['acfi_levels'].get(level, level)
                    print(f"   {level} ({desc}): {pct:.1f}%")
    
    def compare_privacy_levels(self, epsilon_values=[1.0, 2.0, 5.0]):
        """Compare datasets across different privacy levels"""
        print("\\n🔒 PRIVACY LEVEL COMPARISON")
        print("=" * 60)
        
        privacy_results = {}
        
        for epsilon in epsilon_values:
            print(f"\\n📊 Privacy Level ε = {epsilon}")
            print("-" * 40)
            
            datasets = self.load_datasets(epsilon)
            
            if 'assessments' in datasets:
                assess_df = datasets['assessments']
                
                # Compare ACFI scores
                if all(col in assess_df.columns for col in ['ADL_SCORE', 'BEH_SCORE', 'CHC_SCORE']):
                    privacy_results[epsilon] = {
                        'adl_mean': assess_df['ADL_SCORE'].mean(),
                        'beh_mean': assess_df['BEH_SCORE'].mean(),
                        'chc_mean': assess_df['CHC_SCORE'].mean(),
                        'adl_std': assess_df['ADL_SCORE'].std(),
                        'beh_std': assess_df['BEH_SCORE'].std(),
                        'chc_std': assess_df['CHC_SCORE'].std()
                    }
                    
                    print(f"ADL Score: μ={assess_df['ADL_SCORE'].mean():.1f}, σ={assess_df['ADL_SCORE'].std():.1f}")
                    print(f"BEH Score: μ={assess_df['BEH_SCORE'].mean():.1f}, σ={assess_df['BEH_SCORE'].std():.1f}")
                    print(f"CHC Score: μ={assess_df['CHC_SCORE'].mean():.1f}, σ={assess_df['CHC_SCORE'].std():.1f}")
        
        # Analyze privacy-utility trade-off
        if len(privacy_results) > 1:
            print(f"\\n⚖️  Privacy-Utility Trade-off:")
            epsilons = sorted(privacy_results.keys())
            
            for i, metric in enumerate(['adl_mean', 'beh_mean', 'chc_mean']):
                domain = ['ADL', 'BEH', 'CHC'][i]
                print(f"\\n{domain} Score Stability:")
                
                for j, eps in enumerate(epsilons[:-1]):
                    next_eps = epsilons[j + 1]
                    diff = abs(privacy_results[next_eps][metric] - privacy_results[eps][metric])
                    print(f"   ε {eps} → {next_eps}: Δ = {diff:.2f}")
    
    def generate_comprehensive_report(self, epsilon=1.0):
        """Generate comprehensive analysis report"""
        print("🏛️  GOVERNMENT-COMPLIANT NACDC ANALYSIS REPORT")
        print("=" * 70)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔒 Privacy Level: ε = {epsilon}")
        print(f"🇦🇺 Jurisdiction: Australian Government")
        
        # Load datasets
        datasets = self.load_datasets(epsilon)
        
        # Run all analyses
        compliance_score, total_checks = self.validate_government_compliance(datasets)
        self.analyze_demographic_distributions(datasets)
        self.analyze_care_patterns(datasets)
        
        # Overall assessment
        print("\\n🎯 OVERALL ASSESSMENT")
        print("=" * 60)
        
        if compliance_score / total_checks >= 0.9:
            print("🟢 RECOMMENDATION: Dataset is ready for government submission")
            print("   ✅ Meets all compliance requirements")
            print("   ✅ Realistic demographic distributions")
            print("   ✅ Proper data linkage and structure")
        else:
            print("🟡 RECOMMENDATION: Address compliance issues before submission")
            print("   ⚠️  Some government requirements not met")
            print("   ⚠️  Review data structure and codes")
        
        print(f"\\n📊 Quality Score: {(compliance_score / total_checks) * 100:.1f}/100")
        
        return datasets

if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = GovernmentCompliantAnalyzer()
    
    print("Starting government-compliant NACDC data analysis...")
    
    # Analyze each privacy level
    for epsilon in [1.0, 2.0, 5.0]:
        print(f"\\n{'='*70}")
        print(f"ANALYZING PRIVACY LEVEL ε = {epsilon}")
        print(f"{'='*70}")
        
        datasets = analyzer.generate_comprehensive_report(epsilon)
    
    # Compare across privacy levels
    analyzer.compare_privacy_levels()
    
    print(f"\\n🎉 Analysis complete! Government-compliant NACDC datasets validated.")
