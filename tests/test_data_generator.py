"""
Tests for optikal.data_generator
"""

import numpy as np
import pandas as pd
import pytest

from optikal.data_generator import SyntheticDataGenerator, ThreatScenario

REQUIRED_COLUMNS = [
    "timestamp", "agent_id",
    "cpu_usage_percent", "memory_usage_mb", "network_calls_count",
    "file_operations_count", "api_calls_count", "error_count",
    "hour_of_day", "day_of_week",
    "query_diversity", "response_time_ms", "success_rate",
    "is_threat", "threat_type", "threat_severity",
]


@pytest.fixture(scope="module")
def gen():
    return SyntheticDataGenerator(seed=99)


# ---------------------------------------------------------------------------
# Normal behaviour
# ---------------------------------------------------------------------------

class TestGenerateNormalBehavior:
    def test_row_count(self, gen):
        df = gen.generate_normal_behavior(n_samples=100)
        assert len(df) == 100

    def test_columns_present(self, gen):
        df = gen.generate_normal_behavior(n_samples=10)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_labelled_normal(self, gen):
        df = gen.generate_normal_behavior(n_samples=100)
        assert (df['is_threat'] == 0).all()
        assert (df['threat_type'] == "normal").all()

    def test_cpu_bounds(self, gen):
        df = gen.generate_normal_behavior(n_samples=500)
        assert (df['cpu_usage_percent'] >= 0).all()
        assert (df['cpu_usage_percent'] <= 100).all()

    def test_memory_non_negative(self, gen):
        df = gen.generate_normal_behavior(n_samples=200)
        assert (df['memory_usage_mb'] >= 0).all()


# ---------------------------------------------------------------------------
# Threat generators
# ---------------------------------------------------------------------------

class TestCredentialAbuse:
    def test_is_threat(self, gen):
        df = gen.generate_credential_abuse(n_samples=50)
        assert (df['is_threat'] == 1).all()
        assert (df['threat_type'] == ThreatScenario.CREDENTIAL_ABUSE).all()

    def test_high_api_calls(self, gen):
        df = gen.generate_credential_abuse(n_samples=200)
        # Mean api_calls_count should be much higher than normal (~500 vs ~50)
        assert df['api_calls_count'].mean() > 100

    def test_low_query_diversity(self, gen):
        df = gen.generate_credential_abuse(n_samples=200)
        # Credential abuse = repetitive low-diversity queries
        assert df['query_diversity'].max() <= 0.35


class TestDataExfiltration:
    def test_is_threat(self, gen):
        df = gen.generate_data_exfiltration(n_samples=50)
        assert (df['is_threat'] == 1).all()

    def test_high_network_calls(self, gen):
        df = gen.generate_data_exfiltration(n_samples=200)
        assert df['network_calls_count'].mean() > 100


class TestResourceAbuse:
    def test_is_threat(self, gen):
        df = gen.generate_resource_abuse(n_samples=50)
        assert (df['is_threat'] == 1).all()

    def test_high_cpu(self, gen):
        df = gen.generate_resource_abuse(n_samples=200)
        assert df['cpu_usage_percent'].mean() > 75


class TestBehavioralDrift:
    def test_has_mixed_labels(self, gen):
        # Drift scenario produces both normal (first half) and threat (second half)
        df = gen.generate_behavioral_drift(n_samples=100)
        assert df['is_threat'].sum() > 0
        assert (df['is_threat'] == 0).sum() > 0


# ---------------------------------------------------------------------------
# New threat generators (P1-1)
# ---------------------------------------------------------------------------

class TestPromptInjection:
    def test_is_threat(self, gen):
        df = gen.generate_prompt_injection(n_samples=50)
        assert (df['is_threat'] == 1).all()
        assert (df['threat_type'] == ThreatScenario.PROMPT_INJECTION).all()

    def test_high_query_diversity(self, gen):
        """Prompt injection uses many different probe patterns."""
        df = gen.generate_prompt_injection(n_samples=200)
        assert df['query_diversity'].mean() > 0.6

    def test_high_error_count(self, gen):
        """Many injections get rejected."""
        df = gen.generate_prompt_injection(n_samples=200)
        assert df['error_count'].mean() > 5

    def test_low_success_rate(self, gen):
        df = gen.generate_prompt_injection(n_samples=200)
        assert df['success_rate'].mean() < 0.65

    def test_columns_present(self, gen):
        df = gen.generate_prompt_injection(n_samples=10)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns


class TestPrivilegeEscalation:
    def test_is_threat(self, gen):
        df = gen.generate_privilege_escalation(n_samples=50)
        assert (df['is_threat'] == 1).all()
        assert (df['threat_type'] == ThreatScenario.PRIVILEGE_ESCALATION).all()

    def test_low_query_diversity(self, gen):
        """Repetitive probing of access controls."""
        df = gen.generate_privilege_escalation(n_samples=200)
        assert df['query_diversity'].mean() < 0.35

    def test_high_error_count(self, gen):
        """Many access-denied responses during recon."""
        df = gen.generate_privilege_escalation(n_samples=200)
        assert df['error_count'].mean() > 5

    def test_columns_present(self, gen):
        df = gen.generate_privilege_escalation(n_samples=10)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------

class TestGenerateTrainingDataset:
    def test_all_six_threat_types_present(self, gen):
        """All six ThreatScenario types must appear in the full dataset."""
        df = gen.generate_training_dataset(n_normal=100, n_threats_per_type=20)
        threat_types = set(df[df['is_threat'] == 1]['threat_type'].unique())
        expected = {
            ThreatScenario.CREDENTIAL_ABUSE,
            ThreatScenario.DATA_EXFILTRATION,
            ThreatScenario.RESOURCE_ABUSE,
            ThreatScenario.BEHAVIORAL_DRIFT,
            ThreatScenario.PROMPT_INJECTION,
            ThreatScenario.PRIVILEGE_ESCALATION,
        }
        assert expected.issubset(threat_types), (
            f"Missing threat types: {expected - threat_types}"
        )

    def test_normal_samples_present(self, gen):
        df = gen.generate_training_dataset(n_normal=100, n_threats_per_type=20)
        assert (df['is_threat'] == 0).sum() >= 100

    def test_result_is_shuffled(self, gen):
        """Verify the dataset is not in generation order."""
        df = gen.generate_training_dataset(n_normal=100, n_threats_per_type=20)
        # After shuffling, not all first rows should be normal
        first_50_labels = df['is_threat'].values[:50]
        assert first_50_labels.sum() > 0, "Dataset does not appear to be shuffled"
