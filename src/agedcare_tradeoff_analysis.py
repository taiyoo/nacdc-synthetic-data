"""
Privacy-Utility Trade-off Analysis for Aged Care Synthetic Data
Analyzes utility metrics comparing DP datasets against baseline data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import os
import warnings
warnings.filterwarnings('ignore')

class AgedCareTradeoffAnalyzer:
    """Analyze privacy-utility trade-offs for aged care synthetic datasets."""
    
    def __init__(self, data_dir='/Users/samsu/Engineering/usyd/csec5614/data'):
        self.data_dir = data_dir
        self.baseline_data = None
        self.dp_datasets = {}
        
    def load_datasets(self):
        """Load baseline and DP datasets."""
        print("Loading datasets...")
        
        # Load baseline
        baseline_file = os.path.join(self.data_dir, "agedcare_baseline_3000.csv")
        self.baseline_data = pd.read_csv(baseline_file)
        print(f"Loaded baseline: {len(self.baseline_data)} records")
        
        # Load DP datasets
        epsilon_values = [0.5, 1.0, 2.0]
        for eps in epsilon_values:
            dp_file = os.path.join(self.data_dir, f"agedcare_synthetic_dp_{eps}_3000.csv")
            self.dp_datasets[eps] = pd.read_csv(dp_file)
            print(f"Loaded DP epsilon={eps}: {len(self.dp_datasets[eps])} records")
    
    def calculate_utility(self):
        """Calculate utility metrics."""
        results = {}
        
        key_vars = ['age_at_admission', 'length_of_stay_days', 'medication_count', 'acfi_care_domain']
        
        for eps in sorted(self.dp_datasets.keys()):
            dp_data = self.dp_datasets[eps]
            
            # Method 1: MAE-based utility (normalized by range)
            mae_utilities = []
            
            # Method 2: Correlation-based utility  
            correlations = []
            
            # Method 3: Relative error utility
            relative_error_utilities = []
            
            for var in key_vars:
                if var in self.baseline_data.columns and var in dp_data.columns:
                    baseline_vals = self.baseline_data[var].values
                    dp_vals = dp_data[var].values
                    
                    # MAE utility (1 - normalized MAE)
                    mae = np.mean(np.abs(baseline_vals - dp_vals))
                    var_range = np.max(baseline_vals) - np.min(baseline_vals)
                    mae_utility = max(0, 1 - (mae / var_range)) if var_range > 0 else 0
                    mae_utilities.append(mae_utility)
                    
                    # Correlation utility
                    correlation = max(0, np.corrcoef(baseline_vals, dp_vals)[0, 1])
                    correlations.append(correlation)
                    
                    # Relative error utility
                    baseline_mean = np.mean(baseline_vals)
                    relative_error = mae / baseline_mean if baseline_mean > 0 else float('inf')
                    relative_error_utility = max(0, 1 - min(1, relative_error))
                    relative_error_utilities.append(relative_error_utility)
            
            results[eps] = {
                'mae_utility': np.mean(mae_utilities),
                'correlation_utility': np.mean(correlations), 
                'relative_error_utility': np.mean(relative_error_utilities),
                'individual_mae_utilities': dict(zip(key_vars, mae_utilities)),
                'individual_correlations': dict(zip(key_vars, correlations))
            }
        
        return results
    
    def create_chart(self, results):
        """Create privacy vs utility chart."""
        plt.style.use('default')
        fig, (ax1) = plt.subplots(1, 1, figsize=(14, 6))
        fig.suptitle('Privacy vs Utility Trade-off: Aged Care Synthetic Data', 
                    fontsize=14, fontweight='bold')
        
        epsilon_values = sorted(results.keys())
        
        # Chart 1: Multiple utility metrics
        mae_utilities = [results[eps]['mae_utility'] for eps in epsilon_values]
        corr_utilities = [results[eps]['correlation_utility'] for eps in epsilon_values]  
        rel_error_utilities = [results[eps]['relative_error_utility'] for eps in epsilon_values]
        
        ax1.plot(epsilon_values, mae_utilities, 'o-', linewidth=3, markersize=8, 
                color='#2E86C1', label='MAE-based Utility')
        ax1.plot(epsilon_values, corr_utilities, 's-', linewidth=3, markersize=8,
                color='#E74C3C', label='Correlation Utility') 
        ax1.plot(epsilon_values, rel_error_utilities, '^-', linewidth=3, markersize=8,
                color='#27AE60', label='Relative Error Utility')
        
        ax1.set_xlabel('Privacy Budget (Epsilon)', fontsize=12)
        ax1.set_ylabel('Utility Score', fontsize=12)
        ax1.set_title('Privacy-Utility Trade-off (Multiple Metrics)', fontsize=13)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value annotations
        for i, eps in enumerate(epsilon_values):
            ax1.annotate(f'{mae_utilities[i]:.3f}', (eps, mae_utilities[i]), 
                        textcoords="offset points", xytext=(0,15), ha='center', 
                        fontsize=9, color='#2E86C1', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        output_path = os.path.join(self.data_dir, '../analysis/privacy_utility_chart.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nChart saved: {output_path}")
        
        plt.show()
    
    def run_analysis(self):
        """Run privacy-utility analysis."""
        print("=" * 60)
        print("PRIVACY vs UTILITY ANALYSIS")
        print("=" * 60)
        
        self.load_datasets()
        
        # Calculate utilities
        print(f"\nCalculating utility metrics...")
        results = self.calculate_utility()
        
        # Print summary
        print(f"\nUTILITY SUMMARY:")
        print("-" * 20)
        for eps in sorted(results.keys()):
            data = results[eps]
            print(f"Epsilon {eps}: MAE={data['mae_utility']:.3f}, "
                  f"Corr={data['correlation_utility']:.3f}, "
                  f"RelErr={data['relative_error_utility']:.3f}")
        
        # Create chart
        print(f"\nCreating privacy-utility chart...")
        self.create_chart(results)
        

def main():
    """Main execution."""
    analyzer = AgedCareTradeoffAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
