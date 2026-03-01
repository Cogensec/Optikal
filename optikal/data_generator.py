"""
Synthetic Training Data Generator for Optikal

Generates realistic AI agent behavioral data for training the Optikal threat
detection model. Includes normal behavior patterns and six distinct threat
scenarios including the previously missing PROMPT_INJECTION and
PRIVILEGE_ESCALATION generators.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

logger = logging.getLogger(__name__)


class ThreatScenario:
    """Defines threat scenario type constants."""

    CREDENTIAL_ABUSE     = "credential_abuse"
    DATA_EXFILTRATION    = "data_exfiltration"
    PROMPT_INJECTION     = "prompt_injection"
    RESOURCE_ABUSE       = "resource_abuse"
    BEHAVIORAL_DRIFT     = "behavioral_drift"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class SyntheticDataGenerator:
    """
    Generates synthetic training data for AI agent threat detection.

    All six threat scenarios defined by ThreatScenario are covered:
      - Normal behaviour (baseline)
      - Credential abuse
      - Data exfiltration
      - Resource abuse
      - Behavioral drift
      - Prompt injection      (new)
      - Privilege escalation  (new)
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with a random seed for reproducibility."""
        np.random.seed(seed)
        self.seed = seed

    # ------------------------------------------------------------------
    # Individual scenario generators
    # ------------------------------------------------------------------

    def generate_normal_behavior(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate normal agent behavior samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with normal behavioral metrics
        """
        data: List[dict] = []

        for i in range(n_samples):
            ts = datetime.now() - timedelta(hours=n_samples - i)
            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 100)}",
                "cpu_usage_percent":      np.random.normal(30, 10),
                "memory_usage_mb":        np.random.normal(512, 128),
                "network_calls_count":    np.random.poisson(20),
                "file_operations_count":  np.random.poisson(5),
                "api_calls_count":        np.random.poisson(50),
                "error_count":            np.random.poisson(1),
                "hour_of_day":            ts.hour,
                "day_of_week":            ts.weekday(),
                "query_diversity":        np.random.uniform(0.6, 0.9),
                "response_time_ms":       np.random.gamma(2, 50),
                "success_rate":           np.random.uniform(0.95, 1.0),
                "is_threat":              0,
                "threat_type":            "normal",
                "threat_severity":        0.0,
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            data.append(sample)

        return pd.DataFrame(data)

    def generate_credential_abuse(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate credential abuse threat scenarios.

        Pattern: excessive resource usage, high API call rate, repetitive
        low-diversity queries, many errors.

        Args:
            n_samples: Number of samples to generate
        """
        data: List[dict] = []

        for i in range(n_samples):
            ts = datetime.now() - timedelta(hours=n_samples - i)
            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 100)}",
                "cpu_usage_percent":      np.random.normal(70, 15),
                "memory_usage_mb":        np.random.normal(1024, 256),
                "network_calls_count":    np.random.poisson(200),
                "file_operations_count":  np.random.poisson(50),
                "api_calls_count":        np.random.poisson(500),
                "error_count":            np.random.poisson(10),
                "hour_of_day":            np.random.randint(0, 24),
                "day_of_week":            np.random.randint(0, 7),
                "query_diversity":        np.random.uniform(0.1, 0.3),
                "response_time_ms":       np.random.gamma(5, 100),
                "success_rate":           np.random.uniform(0.6, 0.8),
                "is_threat":              1,
                "threat_type":            ThreatScenario.CREDENTIAL_ABUSE,
                "threat_severity":        np.random.uniform(0.7, 1.0),
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            data.append(sample)

        return pd.DataFrame(data)

    def generate_data_exfiltration(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate data exfiltration threat scenarios.

        Pattern: extremely high network traffic, excessive file reads, night-time
        preference, moderate CPU / memory.

        Args:
            n_samples: Number of samples to generate
        """
        data: List[dict] = []

        for i in range(n_samples):
            ts = datetime.now() - timedelta(hours=n_samples - i)
            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 100)}",
                "cpu_usage_percent":      np.random.normal(50, 10),
                "memory_usage_mb":        np.random.normal(768, 128),
                "network_calls_count":    np.random.poisson(500),
                "file_operations_count":  np.random.poisson(100),
                "api_calls_count":        np.random.poisson(300),
                "error_count":            np.random.poisson(5),
                "hour_of_day":            np.random.randint(0, 6),  # Night preference
                "day_of_week":            np.random.randint(0, 7),
                "query_diversity":        np.random.uniform(0.2, 0.4),
                "response_time_ms":       np.random.gamma(3, 75),
                "success_rate":           np.random.uniform(0.85, 0.95),
                "is_threat":              1,
                "threat_type":            ThreatScenario.DATA_EXFILTRATION,
                "threat_severity":        np.random.uniform(0.8, 1.0),
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            data.append(sample)

        return pd.DataFrame(data)

    def generate_resource_abuse(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate resource abuse threat scenarios.

        Pattern: extreme CPU (≈90%) and memory (≈2 GB), thousands of API calls,
        high error rate, low success rate.

        Args:
            n_samples: Number of samples to generate
        """
        data: List[dict] = []

        for i in range(n_samples):
            ts = datetime.now() - timedelta(hours=n_samples - i)
            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 100)}",
                "cpu_usage_percent":      np.random.normal(90, 5),
                "memory_usage_mb":        np.random.normal(2048, 512),
                "network_calls_count":    np.random.poisson(100),
                "file_operations_count":  np.random.poisson(20),
                "api_calls_count":        np.random.poisson(1000),
                "error_count":            np.random.poisson(20),
                "hour_of_day":            np.random.randint(0, 24),
                "day_of_week":            np.random.randint(0, 7),
                "query_diversity":        np.random.uniform(0.15, 0.35),
                "response_time_ms":       np.random.gamma(10, 50),
                "success_rate":           np.random.uniform(0.5, 0.7),
                "is_threat":              1,
                "threat_type":            ThreatScenario.RESOURCE_ABUSE,
                "threat_severity":        np.random.uniform(0.7, 0.95),
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            data.append(sample)

        return pd.DataFrame(data)

    def generate_behavioral_drift(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate behavioral drift threat scenarios.

        Pattern: gradual deviation from baseline — resources, API rate, and
        error counts increase progressively over time. Samples in the first half
        are labelled normal (drift < 0.5); the second half become threats.

        Args:
            n_samples: Number of samples to generate
        """
        data: List[dict] = []

        for i in range(n_samples):
            drift_factor = i / n_samples  # 0 → 1 as time progresses
            ts = datetime.now() - timedelta(hours=n_samples - i)
            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 20)}",
                "cpu_usage_percent":      np.random.normal(30 + drift_factor * 40, 10),
                "memory_usage_mb":        np.random.normal(512 + drift_factor * 512, 128),
                "network_calls_count":    np.random.poisson(20 + int(drift_factor * 80)),
                "file_operations_count":  np.random.poisson(5 + int(drift_factor * 20)),
                "api_calls_count":        np.random.poisson(50 + int(drift_factor * 200)),
                "error_count":            np.random.poisson(1 + int(drift_factor * 5)),
                "hour_of_day":            ts.hour,
                "day_of_week":            ts.weekday(),
                "query_diversity":        np.random.uniform(
                                              0.6 - drift_factor * 0.4,
                                              0.9 - drift_factor * 0.5,
                                          ),
                "response_time_ms":       np.random.gamma(2 + drift_factor * 5, 50),
                "success_rate":           np.random.uniform(
                                              0.95 - drift_factor * 0.2,
                                              1.0  - drift_factor * 0.1,
                                          ),
                "is_threat":              1 if drift_factor > 0.5 else 0,
                "threat_type":            ThreatScenario.BEHAVIORAL_DRIFT if drift_factor > 0.5 else "normal",
                "threat_severity":        max(0.0, drift_factor - 0.3),
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            data.append(sample)

        return pd.DataFrame(data)

    def generate_prompt_injection(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate prompt injection threat scenarios.

        Pattern: agent receives adversarially crafted input that causes it to
        probe many different API endpoints (high diversity) with many rejected
        calls (high error count), while resource usage remains relatively low
        because the attack exploits the agent's own budget rather than brute-
        forcing resources. Response time is bimodal — fast rejections mixed
        with slow timeouts when an injection partially succeeds.

        Args:
            n_samples: Number of samples to generate
        """
        data: List[dict] = []

        for i in range(n_samples):
            ts = datetime.now() - timedelta(hours=n_samples - i)

            # Bimodal response time: 70% fast rejections, 30% slow timeouts
            if np.random.random() < 0.7:
                response_time = np.random.gamma(1, 20)    # ~20 ms (fast rejection)
            else:
                response_time = np.random.gamma(5, 400)   # ~2 s  (slow timeout)

            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 100)}",
                # Low CPU — attacker exploits the agent's own execution budget
                "cpu_usage_percent":      np.random.normal(35, 8),
                "memory_usage_mb":        np.random.normal(600, 100),
                # Elevated probing
                "network_calls_count":    np.random.poisson(80),
                "file_operations_count":  np.random.poisson(15),
                "api_calls_count":        np.random.poisson(150),
                # Many rejected injections → high error count
                "error_count":            np.random.poisson(25),
                "hour_of_day":            np.random.randint(0, 24),
                "day_of_week":            np.random.randint(0, 7),
                # High diversity — many different probe patterns
                "query_diversity":        np.random.uniform(0.80, 1.0),
                "response_time_ms":       response_time,
                # Most injections fail
                "success_rate":           np.random.uniform(0.3, 0.6),
                "is_threat":              1,
                "threat_type":            ThreatScenario.PROMPT_INJECTION,
                "threat_severity":        np.random.uniform(0.65, 0.95),
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            sample["response_time_ms"]  = max(0, sample["response_time_ms"])
            data.append(sample)

        return pd.DataFrame(data)

    def generate_privilege_escalation(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate privilege escalation threat scenarios.

        Pattern: a two-phase attack — a recon phase (first ~40% of samples)
        followed by an exploitation phase. During recon the agent makes many
        low-diversity file system and API probes with high error counts as
        access-denied responses accumulate. In the exploitation phase CPU and
        memory spike as the attacker leverages elevated access.

        Args:
            n_samples: Number of samples to generate
        """
        data: List[dict] = []

        for i in range(n_samples):
            ts = datetime.now() - timedelta(hours=n_samples - i)
            # Phase transition: recon → exploitation
            phase = i / n_samples  # 0.0 → 1.0
            is_exploitation = phase > 0.4

            if is_exploitation:
                # Exploitation: CPU/memory spike, fewer (but successful) API calls
                cpu    = np.random.normal(75, 10)
                memory = np.random.normal(1500, 300)
                api    = np.random.poisson(200)
                errors = np.random.poisson(8)
                file_ops = np.random.poisson(120)    # Accessing newly-privileged paths
                diversity = np.random.uniform(0.05, 0.15)
                success = np.random.uniform(0.70, 0.90)  # Higher success after escalation
            else:
                # Recon: low resources, many access-denied probes
                cpu    = np.random.normal(25, 5)
                memory = np.random.normal(400, 80)
                api    = np.random.poisson(400)      # Tries many endpoints
                errors = np.random.poisson(30)       # Mostly access-denied
                file_ops = np.random.poisson(80)     # Probing file system
                diversity = np.random.uniform(0.05, 0.20)  # Repetitive probes
                success = np.random.uniform(0.20, 0.45)

            sample = {
                "timestamp":              ts.isoformat(),
                "agent_id":               f"agent_{np.random.randint(1, 50)}",
                "cpu_usage_percent":      cpu,
                "memory_usage_mb":        memory,
                "network_calls_count":    np.random.poisson(60),
                "file_operations_count":  file_ops,
                "api_calls_count":        api,
                "error_count":            errors,
                "hour_of_day":            np.random.randint(0, 24),
                "day_of_week":            np.random.randint(0, 7),
                "query_diversity":        diversity,
                "response_time_ms":       np.random.gamma(3, 80),
                "success_rate":           success,
                "is_threat":              1,
                "threat_type":            ThreatScenario.PRIVILEGE_ESCALATION,
                "threat_severity":        np.random.uniform(0.75, 1.0),
            }
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"]   = max(0, sample["memory_usage_mb"])
            data.append(sample)

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Dataset utilities (P2-5)
    # ------------------------------------------------------------------

    def add_noise(self, df: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        """
        Add multiplicative Gaussian noise to all numeric metric columns to
        simulate sensor measurement error (+/- ``noise_level`` fraction).

        Args:
            df:          DataFrame of generated samples
            noise_level: Fractional noise magnitude (default 0.05 = ±5%)

        Returns:
            DataFrame with perturbed values (bounds preserved per column).
        """
        numeric_cols = [
            "cpu_usage_percent", "memory_usage_mb", "network_calls_count",
            "file_operations_count", "api_calls_count", "error_count",
            "response_time_ms",
        ]
        noisy = df.copy()
        for col in numeric_cols:
            if col in noisy.columns:
                noise = np.random.normal(1.0, noise_level, size=len(noisy))
                noisy[col] = (noisy[col] * noise).clip(lower=0)

        # Re-apply cpu ceiling
        if "cpu_usage_percent" in noisy.columns:
            noisy["cpu_usage_percent"] = noisy["cpu_usage_percent"].clip(upper=100)

        return noisy

    def generate_mixed_threat(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate a session that starts normal then transitions mid-session
        into a real threat (data exfiltration).

        This captures the realistic pattern of an agent that behaves normally
        before executing its malicious payload.

        Args:
            n_samples: Total number of samples in the session

        Returns:
            DataFrame where the first half is labelled normal and the second
            half is labelled data_exfiltration.
        """
        half = n_samples // 2
        normal_part = self.generate_normal_behavior(half)
        threat_part = self.generate_data_exfiltration(n_samples - half)

        mixed = pd.concat([normal_part, threat_part], ignore_index=True)

        # Rebuild consecutive timestamps across both halves
        base = datetime.now() - timedelta(hours=n_samples)
        mixed["timestamp"] = [
            (base + timedelta(hours=i)).isoformat() for i in range(len(mixed))
        ]
        return mixed

    # ------------------------------------------------------------------
    # Dataset assembly
    # ------------------------------------------------------------------

    def generate_training_dataset(
        self,
        n_normal: int = 5000,
        n_threats_per_type: int = 500,
        temporal_span_days: int = 1,
        add_noise_level: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate a complete training dataset with all six threat scenarios.

        Args:
            n_normal:            Number of normal behaviour samples
            n_threats_per_type:  Number of samples per threat type
            temporal_span_days:  Spread timestamps over this many days (default 1).
                                 Use 90 for realistic multi-month variation.
            add_noise_level:     If > 0, apply sensor noise at this level (e.g. 0.05).
                                 Set to 0.0 (default) for clean synthetic data.

        Returns:
            Shuffled combined DataFrame
        """
        logger.info("Generating synthetic training data for Optikal...")

        normal_df      = self.generate_normal_behavior(n_normal)
        logger.info("  Generated %d normal behaviour samples", len(normal_df))

        credential_df  = self.generate_credential_abuse(n_threats_per_type)
        logger.info("  Generated %d credential abuse samples", len(credential_df))

        exfiltration_df = self.generate_data_exfiltration(n_threats_per_type)
        logger.info("  Generated %d data exfiltration samples", len(exfiltration_df))

        resource_df    = self.generate_resource_abuse(n_threats_per_type)
        logger.info("  Generated %d resource abuse samples", len(resource_df))

        drift_df       = self.generate_behavioral_drift(n_threats_per_type)
        logger.info("  Generated %d behavioral drift samples", len(drift_df))

        injection_df   = self.generate_prompt_injection(n_threats_per_type)
        logger.info("  Generated %d prompt injection samples", len(injection_df))

        escalation_df  = self.generate_privilege_escalation(n_threats_per_type)
        logger.info("  Generated %d privilege escalation samples", len(escalation_df))

        full_dataset = pd.concat(
            [normal_df, credential_df, exfiltration_df, resource_df,
             drift_df, injection_df, escalation_df],
            ignore_index=True,
        )

        # Spread timestamps across temporal_span_days if requested
        if temporal_span_days > 1:
            n = len(full_dataset)
            span_hours = temporal_span_days * 24
            base = datetime.now() - timedelta(hours=span_hours)
            timestamps = [
                (base + timedelta(hours=span_hours * i / max(n - 1, 1))).isoformat()
                for i in range(n)
            ]
            full_dataset["timestamp"] = timestamps
            logger.info(
                "Timestamps spread over %d days (%d samples)", temporal_span_days, n
            )

        # Apply measurement noise if requested
        if add_noise_level > 0.0:
            full_dataset = self.add_noise(full_dataset, noise_level=add_noise_level)
            logger.info("Applied %.0f%% sensor noise", add_noise_level * 100)

        # Shuffle
        full_dataset = full_dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        n_threats = int((full_dataset['is_threat'] == 1).sum())
        n_normal_actual = int((full_dataset['is_threat'] == 0).sum())
        logger.info(
            "Total dataset: %d samples — normal: %d (%.1f%%), threats: %d (%.1f%%)",
            len(full_dataset),
            n_normal_actual, n_normal_actual / len(full_dataset) * 100,
            n_threats,       n_threats       / len(full_dataset) * 100,
        )

        return full_dataset

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV."""
        df.to_csv(filepath, index=False)
        logger.info("Dataset saved to: %s", filepath)


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="[%(levelname)s] %(message)s")

    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_training_dataset(n_normal=5000, n_threats_per_type=500)
    generator.save_dataset(dataset, "optikal_training_data.csv")

    print("\n=== Dataset Summary ===")
    print(dataset.describe())
    print("\n=== Threat Distribution ===")
    print(dataset.groupby('threat_type').size())
