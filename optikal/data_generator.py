"""
Synthetic Training Data Generator for Optikal

Generates realistic AI agent behavioral data for training the Optikal threat detection model.
Includes both normal behavior patterns and various threat scenarios.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json


class ThreatScenario:
    """Defines a threat scenario for synthetic data generation"""
    
    CREDENTIAL_ABUSE = "credential_abuse"
    DATA_EXFILTRATION = "data_exfiltration"
    PROMPT_INJECTION = "prompt_injection"
    RESOURCE_ABUSE = "resource_abuse"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    PRIVILEGE_ESCALATION = "privilege_escalation"


class SyntheticDataGenerator:
    """
    Generates synthetic training data for AI agent threat detection
    """
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.seed = seed
        
    def generate_normal_behavior(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate normal agent behavior data
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with normal behavioral metrics
        """
        data = []
        
        for i in range(n_samples):
            # Normal operational parameters
            sample = {
                "timestamp": (datetime.now() - timedelta(hours=n_samples-i)).isoformat(),
                "agent_id": f"agent_{np.random.randint(1, 100)}",
                
                # Resource usage (normal ranges)
                "cpu_usage_percent": np.random.normal(30, 10),  # Mean 30%, std 10%
                "memory_usage_mb": np.random.normal(512, 128),  # Mean 512MB
                "network_calls_count": np.random.poisson(20),   # ~20 calls
                
                # Operations (normal ranges)
                "file_operations_count": np.random.poisson(5),   # ~5 file ops
                "api_calls_count": np.random.poisson(50),        # ~50 API calls
                "error_count": np.random.poisson(1),             # ~1 error
                
                # Time patterns
                "hour_of_day": (datetime.now() - timedelta(hours=n_samples-i)).hour,
                "day_of_week": (datetime.now() - timedelta(hours=n_samples-i)).weekday(),
                
                # Normal behavior indicators
                "query_diversity": np.random.uniform(0.6, 0.9),  # High diversity
                "response_time_ms": np.random.gamma(2, 50),      # ~100ms average
                "success_rate": np.random.uniform(0.95, 1.0),    # High success
                
                # Label
                "is_threat": 0,
                "threat_type": "normal",
                "threat_severity": 0.0
            }
            
            # Ensure non-negative values
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"] = max(0, sample["memory_usage_mb"])
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def generate_credential_abuse(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate credential abuse threat scenarios"""
        data = []
        
        for i in range(n_samples):
            sample = {
                "timestamp": (datetime.now() - timedelta(hours=n_samples-i)).isoformat(),
                "agent_id": f"agent_{np.random.randint(1, 100)}",
                
                # Excessive resource usage
                "cpu_usage_percent": np.random.normal(70, 15),   # High CPU
                "memory_usage_mb": np.random.normal(1024, 256),  # High memory
                "network_calls_count": np.random.poisson(200),   # Excessive calls
                
                # Excessive operations
                "file_operations_count": np.random.poisson(50),  # Many file ops
                "api_calls_count": np.random.poisson(500),       # Excessive API calls
                "error_count": np.random.poisson(10),            # Many errors
                
                "hour_of_day": np.random.randint(0, 24),
                "day_of_week": np.random.randint(0, 7),
                
                # Suspicious patterns
                "query_diversity": np.random.uniform(0.1, 0.3),  # Low diversity (repetitive)
                "response_time_ms": np.random.gamma(5, 100),     # Slower responses
                "success_rate": np.random.uniform(0.6, 0.8),     # Lower success
                
                "is_threat": 1,
                "threat_type": ThreatScenario.CREDENTIAL_ABUSE,
                "threat_severity": np.random.uniform(0.7, 1.0)
            }
            
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"] = max(0, sample["memory_usage_mb"])
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def generate_data_exfiltration(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate data exfiltration threat scenarios"""
        data = []
        
        for i in range(n_samples):
            sample = {
                "timestamp": (datetime.now() - timedelta(hours=n_samples-i)).isoformat(),
                "agent_id": f"agent_{np.random.randint(1, 100)}",
                
                # High network activity
                "cpu_usage_percent": np.random.normal(50, 10),
                "memory_usage_mb": np.random.normal(768, 128),
                "network_calls_count": np.random.poisson(500),   # Very high network
                
                # Unusual file access
                "file_operations_count": np.random.poisson(100), # Excessive reads
                "api_calls_count": np.random.poisson(300),
                "error_count": np.random.poisson(5),
                
                "hour_of_day": np.random.randint(0, 6),  # Often at night
                "day_of_week": np.random.randint(0, 7),
                
                "query_diversity": np.random.uniform(0.2, 0.4),
                "response_time_ms": np.random.gamma(3, 75),
                "success_rate": np.random.uniform(0.85, 0.95),
                
                "is_threat": 1,
                "threat_type": ThreatScenario.DATA_EXFILTRATION,
                "threat_severity": np.random.uniform(0.8, 1.0)
            }
            
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"] = max(0, sample["memory_usage_mb"])
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def generate_resource_abuse(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate resource abuse threat scenarios"""
        data = []
        
        for i in range(n_samples):
            sample = {
                "timestamp": (datetime.now() - timedelta(hours=n_samples-i)).isoformat(),
                "agent_id": f"agent_{np.random.randint(1, 100)}",
                
                # Extreme resource usage
                "cpu_usage_percent": np.random.normal(90, 5),    # Very high CPU
                "memory_usage_mb": np.random.normal(2048, 512),  # Very high memory
                "network_calls_count": np.random.poisson(100),
                
                "file_operations_count": np.random.poisson(20),
                "api_calls_count": np.random.poisson(1000),      # Excessive APIs
                "error_count": np.random.poisson(20),            # Many errors
                
                "hour_of_day": np.random.randint(0, 24),
                "day_of_week": np.random.randint(0, 7),
                
                "query_diversity": np.random.uniform(0.15, 0.35),
                "response_time_ms": np.random.gamma(10, 50),     # Very slow
                "success_rate": np.random.uniform(0.5, 0.7),     # Low success
                
                "is_threat": 1,
                "threat_type": ThreatScenario.RESOURCE_ABUSE,
                "threat_severity": np.random.uniform(0.7, 0.95)
            }
            
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"] = max(0, sample["memory_usage_mb"])
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def generate_behavioral_drift(self, n_samples: int = 100) -> pd.DataFrame:
        """Generate behavioral drift threat scenarios"""
        data = []
        
        for i in range(n_samples):
            # Gradual drift from normal behavior
            drift_factor = i / n_samples  # Increases over time
            
            sample = {
                "timestamp": (datetime.now() - timedelta(hours=n_samples-i)).isoformat(),
                "agent_id": f"agent_{np.random.randint(1, 20)}",  # Same agents
                
                # Gradually increasing usage
                "cpu_usage_percent": np.random.normal(30 + drift_factor * 40, 10),
                "memory_usage_mb": np.random.normal(512 + drift_factor * 512, 128),
                "network_calls_count": np.random.poisson(20 + int(drift_factor * 80)),
                
                "file_operations_count": np.random.poisson(5 + int(drift_factor * 20)),
                "api_calls_count": np.random.poisson(50 + int(drift_factor * 200)),
                "error_count": np.random.poisson(1 + int(drift_factor * 5)),
                
                "hour_of_day": (datetime.now() - timedelta(hours=n_samples-i)).hour,
                "day_of_week": (datetime.now() - timedelta(hours=n_samples-i)).weekday(),
                
                "query_diversity": np.random.uniform(0.6 - drift_factor * 0.4, 0.9 - drift_factor * 0.5),
                "response_time_ms": np.random.gamma(2 + drift_factor * 5, 50),
                "success_rate": np.random.uniform(0.95 - drift_factor * 0.2, 1.0 - drift_factor * 0.1),
                
                "is_threat": 1 if drift_factor > 0.5 else 0,
                "threat_type": ThreatScenario.BEHAVIORAL_DRIFT if drift_factor > 0.5 else "normal",
                "threat_severity": max(0, drift_factor - 0.3)
            }
            
            sample["cpu_usage_percent"] = max(0, min(100, sample["cpu_usage_percent"]))
            sample["memory_usage_mb"] = max(0, sample["memory_usage_mb"])
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def generate_training_dataset(
        self,
        n_normal: int = 5000,
        n_threats_per_type: int = 500
    ) -> pd.DataFrame:
        """
        Generate complete training dataset with balanced threat scenarios
        
        Args:
            n_normal: Number of normal samples
            n_threats_per_type: Number of samples per threat type
            
        Returns:
            Complete training dataset
        """
        print("Generating synthetic training data for Optikal...")
        
        # Generate all scenarios
        normal_df = self.generate_normal_behavior(n_normal)
        print(f"  ✓ Generated {len(normal_df)} normal behavior samples")
        
        credential_df = self.generate_credential_abuse(n_threats_per_type)
        print(f"  ✓ Generated {len(credential_df)} credential abuse samples")
        
        exfiltration_df = self.generate_data_exfiltration(n_threats_per_type)
        print(f"  ✓ Generated {len(exfiltration_df)} data exfiltration samples")
        
        resource_df = self.generate_resource_abuse(n_threats_per_type)
        print(f"  ✓ Generated {len(resource_df)} resource abuse samples")
        
        drift_df = self.generate_behavioral_drift(n_threats_per_type)
        print(f"  ✓ Generated {len(drift_df)} behavioral drift samples")
        
        # Combine all data
        full_dataset = pd.concat([
            normal_df,
            credential_df,
            exfiltration_df,
            resource_df,
            drift_df
        ], ignore_index=True)
        
        # Shuffle
        full_dataset = full_dataset.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        print(f"\n✓ Total dataset size: {len(full_dataset)} samples")
        print(f"  - Normal: {(full_dataset['is_threat'] == 0).sum()} ({(full_dataset['is_threat'] == 0).sum() / len(full_dataset) * 100:.1f}%)")
        print(f"  - Threats: {(full_dataset['is_threat'] == 1).sum()} ({(full_dataset['is_threat'] == 1).sum() / len(full_dataset) * 100:.1f}%)")
        
        return full_dataset
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV"""
        df.to_csv(filepath, index=False)
        print(f"\n✓ Dataset saved to: {filepath}")


if __name__ == "__main__":
    # Generate training data
    generator = SyntheticDataGenerator(seed=42)
    
    # Create dataset
    dataset = generator.generate_training_dataset(
        n_normal=5000,
        n_threats_per_type=500
    )
    
    # Save to file
    generator.save_dataset(dataset, "optikal_training_data.csv")
    
    print("\n=== Dataset Summary ===")
    print(dataset.describe())
    print("\n=== Threat Distribution ===")
    print(dataset.groupby('threat_type').size())
