import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


class Re_identificationRiskScorer:
    """
    Risk-Aware Re-identification Scoring System (RSS)
    --------------------------------------------------
    Evaluates privacy risks in synthetic aged-care datasets using:
    - k-Anonymity
    - l-Diversity
    - t-Closeness

    Attributes:
        quasi_identifiers (list): Attributes used to form equivalence classes.
        sensitive_attributes (list): Attributes considered sensitive.
        k_threshold (int): Minimum k value for compliance.
        l_threshold (int): Minimum average diversity.
        t_threshold (float): Maximum acceptable t-closeness.
    """

    def __init__(
        self,
        quasi_identifiers=None,
        sensitive_attributes=None,
        k_threshold=5,
        l_threshold=2,
        t_threshold=0.2
    ):
        self.quasi_identifiers = quasi_identifiers or ['age_at_admission', 'sex']
        self.sensitive_attributes = sensitive_attributes or [
            'dementia_status', 'acfi_care_domain',
            'acfi_accommodation_domain', 'acfi_complex_health_care',
            'falls_risk', 'medication_count', 'chronic_conditions_count',
            'mobility_assistance', 'personal_care_assistance'
        ]
        self.k_threshold = k_threshold
        self.l_threshold = l_threshold
        self.t_threshold = t_threshold

    def compute_metrics(self, df):
        """Compute k-anonymity, l-diversity, and t-closeness metrics."""
        grouped = df.groupby(self.quasi_identifiers)
        k_values, l_values, t_values = [], [], []

        for _, group in grouped:
            # 1) k-Anonymity
            k_values.append(len(group))

            # 2) l-Diversity
            l = np.mean([
                group[attr].nunique()
                for attr in self.sensitive_attributes
                if attr in group.columns
            ])
            l_values.append(l)

            # 3) t-Closeness
            for attr in self.sensitive_attributes:
                if attr not in df.columns:
                    continue
                global_dist = df[attr].value_counts(normalize=True)
                local_dist = group[attr].value_counts(normalize=True)

                all_vals = set(global_dist.index) | set(local_dist.index)
                global_vec = np.array([global_dist.get(v, 0) for v in all_vals])
                local_vec = np.array([local_dist.get(v, 0) for v in all_vals])

                t = jensenshannon(global_vec, local_vec)
                if not np.isnan(t):
                    t_values.append(t)

        metrics = {
            'k_min': np.min(k_values),
            'k_avg': np.mean(k_values),
            'l_avg': np.mean(l_values),
            't_avg': np.mean(t_values),
            't_max': np.max(t_values)
        }
        return metrics

    def assess_compliance(self, metrics):
        """Determine compliance based on computed metrics."""
        compliant = (
            metrics['k_min'] >= self.k_threshold and
            metrics['l_avg'] >= self.l_threshold and
            metrics['t_avg'] <= self.t_threshold
        )
        return "Compliant" if compliant else "Non-Compliant"

    def generate_report(self, metrics, compliance):
        """Generate a formatted privacy compliance report."""
        print("\n📊 Privacy Compliance Report")
        print("----------------------------------------")
        print(f"Minimum k-Anonymity: {metrics['k_min']}")
        print(f"Average l-Diversity: {metrics['l_avg']:.2f}")
        print(f"Average t-Closeness: {metrics['t_avg']:.3f}")
        print(f"Maximum t-Closeness: {metrics['t_max']:.3f}")

        if compliance == "Compliant":
            print("\n✅Dataset meets privacy compliance requirements.")
        else:
            print("\n⚠️Dataset may not meet compliance thresholds.")

    def evaluate_dataset(self, df):
        """Main pipeline: compute metrics, assess compliance, print report."""
        print("\n🔐 Running Re-identification Risk Scorer (RRS)...")
        metrics = self.compute_metrics(df)
        compliance = self.assess_compliance(metrics)
        self.generate_report(metrics, compliance)
        return metrics, compliance


if __name__ == "__main__":
    datasets = [
        ("High Privacy (ε=0.5)", "data/synthetic/agedcare_synthetic_dp_0.5_3000.csv"),
        ("Moderate Privacy (ε=1.0)", "data/synthetic/agedcare_synthetic_dp_1.0_3000.csv"),
        ("Lower Privacy (ε=2.0)", "data/synthetic/agedcare_synthetic_dp_2.0_3000.csv"),
    ]

    scorer = Re_identificationRiskScorer()

    for label, path in datasets:
        print(f"\n==============================")
        print(f" Evaluating Dataset: {label}")
        print(f"==============================")

        df = pd.read_csv(path)
        metrics, compliance = scorer.evaluate_dataset(df)

        print("\nResults Summary:")
        print(f"Dataset: {label}")
        print(f"Compliance Status: {compliance}")
        print(f"Metrics: {metrics}")
