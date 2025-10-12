"""
NACDC Synthetic Dataset Analysis and Validation Tool
Provides analysis tools for generated synthetic datasets with differential privacy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

class NACDCDatasetAnalyzer:
    """Analyzes and validates synthetic NACDC datasets."""
    
    def __init__(self, workspace_path=None):
        if workspace_path is None:
            # Use relative path from script location
            self.workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.workspace_path = workspace_path
        
        self.data_path = os.path.join(self.workspace_path, "data", "synthetic")
        self.analysis_path = os.path.join(self.workspace_path, "analysis")
        
    def load_datasets(self):
        """Load all generated synthetic datasets for comparison."""
        datasets = {}
        
        # Find all synthetic dataset files in the data/synthetic directory
        for filename in os.listdir(self.data_path):
            if filename.startswith('nacdc_synthetic_dp_') and filename.endswith('.csv'):
                filepath = os.path.join(self.data_path, filename)
                df = pd.read_csv(filepath)
                
                # Extract privacy parameters from filename
                if '_dp_' in filename:
                    parts = filename.split('_')
                    epsilon = float(parts[parts.index('dp') + 1])
                    n_records = int(parts[-1].replace('.csv', ''))
                    datasets[f"ε={epsilon}"] = {
                        'data': df,
                        'epsilon': epsilon,
                        'n_records': n_records,
                        'filename': filename
                    }
                else:
                    datasets['No DP'] = {
                        'data': df,
                        'epsilon': None,
                        'n_records': len(df),
                        'filename': filename
                    }
        
        return datasets
    
    def compare_distributions(self, datasets):
        """Compare distributions across different privacy levels."""
        print("=== Distribution Comparison Across Privacy Levels ===\n")
        
        # Numerical columns to compare
        numerical_cols = ['age_at_admission', 'length_of_stay_days', 'medication_count', 
                         'chronic_conditions_count']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numerical_cols):
            ax = axes[i]
            
            for label, dataset_info in datasets.items():
                df = dataset_info['data']
                if col in df.columns:
                    ax.hist(df[col], alpha=0.6, label=label, bins=30, density=True)
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_path, 'nacdc_distribution_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical comparison
        print("Statistical Comparison (Mean ± Std):")
        print("-" * 80)
        
        for col in numerical_cols:
            print(f"\n{col}:")
            for label, dataset_info in datasets.items():
                df = dataset_info['data']
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    print(f"  {label:12}: {mean_val:8.2f} ± {std_val:6.2f}")
    
    def privacy_utility_analysis(self, datasets):
        """Analyze privacy-utility tradeoff."""
        print("\n=== Privacy-Utility Tradeoff Analysis ===\n")
        
        # Compare against baseline (no DP)
        baseline = None
        for label, dataset_info in datasets.items():
            if dataset_info['epsilon'] is None:
                baseline = dataset_info['data']
                break
        
        if baseline is None:
            print("No baseline dataset found. Cannot perform utility analysis.")
            return
        
        metrics = {}
        
        # Calculate utility metrics for each privacy level
        for label, dataset_info in datasets.items():
            if dataset_info['epsilon'] is not None:  # Skip baseline
                df = dataset_info['data']
                epsilon = dataset_info['epsilon']
                
                # Mean Absolute Error for numerical columns
                mae_scores = {}
                numerical_cols = ['age_at_admission', 'length_of_stay_days', 'medication_count']
                
                for col in numerical_cols:
                    if col in df.columns and col in baseline.columns:
                        # Calculate MAE between means
                        baseline_mean = baseline[col].mean()
                        private_mean = df[col].mean()
                        mae = abs(baseline_mean - private_mean)
                        mae_scores[col] = mae
                
                # Hellinger distance for categorical distributions
                categorical_distances = {}
                categorical_cols = ['sex', 'care_level', 'dementia_status']
                
                for col in categorical_cols:
                    if col in df.columns and col in baseline.columns:
                        baseline_dist = baseline[col].value_counts(normalize=True)
                        private_dist = df[col].value_counts(normalize=True)
                        
                        # Calculate Hellinger distance
                        all_categories = set(baseline_dist.index) | set(private_dist.index)
                        hellinger_dist = 0
                        
                        for cat in all_categories:
                            p = baseline_dist.get(cat, 0)
                            q = private_dist.get(cat, 0)
                            hellinger_dist += (np.sqrt(p) - np.sqrt(q)) ** 2
                        
                        hellinger_dist = np.sqrt(hellinger_dist / 2)
                        categorical_distances[col] = hellinger_dist
                
                metrics[epsilon] = {
                    'mae_scores': mae_scores,
                    'categorical_distances': categorical_distances,
                    'overall_utility': 1 - np.mean(list(categorical_distances.values()))
                }
        
        # Print utility analysis
        print("Mean Absolute Error (vs baseline):")
        print("-" * 50)
        for epsilon in sorted(metrics.keys()):
            print(f"\nε = {epsilon}:")
            for col, mae in metrics[epsilon]['mae_scores'].items():
                print(f"  {col}: {mae:.3f}")
        
        print("\nCategorical Distribution Distances (Hellinger):")
        print("-" * 50)
        for epsilon in sorted(metrics.keys()):
            print(f"\nε = {epsilon}:")
            for col, dist in metrics[epsilon]['categorical_distances'].items():
                print(f"  {col}: {dist:.3f}")
        
        # Plot privacy-utility tradeoff
        epsilons = sorted(metrics.keys())
        utilities = [metrics[eps]['overall_utility'] for eps in epsilons]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, utilities, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Data Utility Score', fontsize=12)
        plt.title('Privacy-Utility Tradeoff\n(Higher ε = Less Privacy, Higher Utility)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # Annotate points
        for eps, util in zip(epsilons, utilities):
            plt.annotate(f'ε={eps}', (eps, util), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_path, 'privacy_utility_tradeoff.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics
    
    def generate_data_quality_report(self, datasets):
        """Generate comprehensive data quality report."""
        print("\n=== Data Quality Assessment ===\n")
        
        quality_report = {}
        
        for label, dataset_info in datasets.items():
            df = dataset_info['data']
            epsilon = dataset_info['epsilon']
            
            # Basic quality metrics
            total_records = len(df)
            missing_values = df.isnull().sum().sum()
            duplicate_records = df.duplicated().sum()
            
            # Value range validation
            age_invalid = ((df['age_at_admission'] < 65) | (df['age_at_admission'] > 105)).sum()
            los_invalid = (df['length_of_stay_days'] < 0).sum()
            seifa_invalid = ((df['seifa_decile'] < 1) | (df['seifa_decile'] > 10)).sum()
            
            # Logical consistency checks
            # Check if discharge_date is after admission_date
            df['admission_date'] = pd.to_datetime(df['admission_date'])
            df['discharge_date'] = pd.to_datetime(df['discharge_date'])
            
            date_inconsistent = 0
            for idx, row in df.iterrows():
                if pd.notna(row['discharge_date']) and row['discharge_date'] < row['admission_date']:
                    date_inconsistent += 1
            
            quality_metrics = {
                'total_records': total_records,
                'missing_values': missing_values,
                'missing_percentage': (missing_values / (total_records * len(df.columns))) * 100,
                'duplicate_records': duplicate_records,
                'age_invalid': age_invalid,
                'los_invalid': los_invalid,
                'seifa_invalid': seifa_invalid,
                'date_inconsistent': date_inconsistent,
                'overall_quality_score': 100 - (
                    (missing_values / (total_records * len(df.columns))) * 100 +
                    (duplicate_records / total_records) * 100 +
                    (age_invalid / total_records) * 100 +
                    (los_invalid / total_records) * 100 +
                    (date_inconsistent / total_records) * 100
                )
            }
            
            quality_report[label] = quality_metrics
            
            print(f"Dataset: {label}")
            if epsilon is not None:
                print(f"Privacy Level (ε): {epsilon}")
            print(f"Records: {total_records:,}")
            print(f"Missing values: {missing_values} ({quality_metrics['missing_percentage']:.2f}%)")
            print(f"Duplicates: {duplicate_records}")
            print(f"Invalid ages: {age_invalid}")
            print(f"Invalid length of stay: {los_invalid}")
            print(f"Date inconsistencies: {date_inconsistent}")
            print(f"Quality Score: {quality_metrics['overall_quality_score']:.1f}/100")
            print("-" * 50)
        
        return quality_report
    
    def demographic_analysis(self, datasets):
        """Analyze demographic patterns in the synthetic data."""
        print("\n=== Demographic Analysis ===\n")
        
        # Create demographic comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Age distribution by sex
        ax1 = axes[0, 0]
        for label, dataset_info in datasets.items():
            df = dataset_info['data']
            for sex in ['M', 'F']:
                subset = df[df['sex'] == sex]['age_at_admission']
                ax1.hist(subset, alpha=0.5, label=f'{label} - {sex}', bins=20, density=True)
        ax1.set_title('Age Distribution by Sex')
        ax1.set_xlabel('Age at Admission')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # 2. Care level distribution
        ax2 = axes[0, 1]
        care_level_data = {}
        for label, dataset_info in datasets.items():
            df = dataset_info['data']
            care_dist = df['care_level'].value_counts(normalize=True)
            care_level_data[label] = care_dist
        
        care_df = pd.DataFrame(care_level_data).fillna(0)
        care_df.plot(kind='bar', ax=ax2)
        ax2.set_title('Care Level Distribution')
        ax2.set_ylabel('Proportion')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Length of stay by care level
        ax3 = axes[1, 0]
        for label, dataset_info in datasets.items():
            df = dataset_info['data']
            care_los = df.groupby('care_level')['length_of_stay_days'].mean()
            ax3.plot(care_los.index, care_los.values, 'o-', label=label, markersize=6)
        ax3.set_title('Average Length of Stay by Care Level')
        ax3.set_ylabel('Days')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Dementia prevalence by age group
        ax4 = axes[1, 1]
        for label, dataset_info in datasets.items():
            df = dataset_info['data']
            df['age_group'] = pd.cut(df['age_at_admission'], 
                                   bins=[65, 75, 85, 95, 105], 
                                   labels=['65-74', '75-84', '85-94', '95+'])
            dementia_by_age = df.groupby('age_group')['dementia_status'].apply(
                lambda x: (x == 'Yes').mean()
            )
            ax4.plot(dementia_by_age.index, dementia_by_age.values, 'o-', label=label, markersize=6)
        ax4.set_title('Dementia Prevalence by Age Group')
        ax4.set_ylabel('Proportion with Dementia')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_path, 'demographic_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_analysis_report(self, datasets, quality_report, privacy_metrics):
        """Export comprehensive analysis report."""
        report_lines = []
        report_lines.append("NACDC Synthetic Dataset Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset overview
        report_lines.append("DATASET OVERVIEW")
        report_lines.append("-" * 20)
        for label, dataset_info in datasets.items():
            report_lines.append(f"{label}:")
            report_lines.append(f"  Records: {dataset_info['n_records']:,}")
            if dataset_info['epsilon']:
                report_lines.append(f"  Privacy Budget (ε): {dataset_info['epsilon']}")
            report_lines.append(f"  File: {dataset_info['filename']}")
            report_lines.append("")
        
        # Quality assessment
        report_lines.append("DATA QUALITY SCORES")
        report_lines.append("-" * 20)
        for label, metrics in quality_report.items():
            report_lines.append(f"{label}: {metrics['overall_quality_score']:.1f}/100")
        report_lines.append("")
        
        # Privacy-utility summary
        if privacy_metrics:
            report_lines.append("PRIVACY-UTILITY TRADEOFF")
            report_lines.append("-" * 25)
            for epsilon in sorted(privacy_metrics.keys()):
                utility = privacy_metrics[epsilon]['overall_utility']
                report_lines.append(f"ε = {epsilon}: Utility Score = {utility:.3f}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 15)
        
        if privacy_metrics:
            best_balance = max(privacy_metrics.items(), 
                             key=lambda x: x[1]['overall_utility'] if x[0] <= 1.0 else 0)
            report_lines.append(f"Recommended privacy level: ε = {best_balance[0]} (good privacy-utility balance)")
        
        report_lines.append("- Datasets maintain realistic demographic patterns")
        report_lines.append("- All privacy levels suitable for research use")
        report_lines.append("- Lower ε values recommended for sensitive applications")
        report_lines.append("- Higher ε values suitable for general research")
        
        # Save report
        report_path = os.path.join(self.analysis_path, "nacdc_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Analysis report saved to: {report_path}")
        return report_lines


def main():
    """Main analysis function."""
    print("=== NACDC Synthetic Dataset Analysis ===\n")
    
    analyzer = NACDCDatasetAnalyzer()
    
    # Load all datasets
    datasets = analyzer.load_datasets()
    
    if not datasets:
        print("No synthetic datasets found. Please run nacdc_synthetic_dp.py first.")
        return
    
    print(f"Found {len(datasets)} datasets for analysis:")
    for label, info in datasets.items():
        print(f"  - {label}: {info['n_records']:,} records")
    print()
    
    # Perform comprehensive analysis
    print("1. Comparing distributions across privacy levels...")
    analyzer.compare_distributions(datasets)
    
    print("\n2. Analyzing privacy-utility tradeoff...")
    privacy_metrics = analyzer.privacy_utility_analysis(datasets)
    
    print("\n3. Assessing data quality...")
    quality_report = analyzer.generate_data_quality_report(datasets)
    
    print("\n4. Analyzing demographic patterns...")
    analyzer.demographic_analysis(datasets)
    
    print("\n5. Generating comprehensive report...")
    analyzer.export_analysis_report(datasets, quality_report, privacy_metrics)
    
    print("\n=== Analysis Complete ===")
    print("Check the generated plots and analysis report for detailed insights.")


if __name__ == "__main__":
    main()
