"""
Optikal Inference Example

Demonstrates how to use trained Optikal models for threat detection.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# Add optikal to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from optikal.feature_engineering import OptikalFeatureEngineer


def load_model(model_dir="optikal/optikal_models"):
    """
    Load trained Optikal model and scaler
    
    Returns:
        Tuple of (model, engineer)
    """
    model_path = Path(model_dir) / "optikal_isolation_forest.pkl"
    scaler_path = Path(model_dir) / "optikal_scaler.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}.\\n"
            "Please train the model first: cd optikal && python train_quick.py"
        )
    
    # Load model
    model = joblib.load(model_path)
    
    # Load scaler
    engineer = OptikalFeatureEngineer()
    engineer.load_scaler(str(scaler_path))
    
    print(f"✓ Model loaded from {model_dir}")
    return model, engineer


def create_sample_agent_data():
    """Create sample agent behavioral data for testing"""
    return pd.DataFrame([
        {
            "cpu_usage_percent": 35.2,
            "memory_usage_mb": 512.0,
            "network_calls_count": 25,
            "file_operations_count": 3,
            "api_calls_count": 45,
            "error_count": 1,
            "hour_of_day": 14,
            "day_of_week": 2,
            "query_diversity": 0.75,
            "response_time_ms": 120.0,
            "success_rate": 0.97
        }
    ])


def create_suspicious_agent_data():
    """Create suspicious agent behavioral data for testing"""
    return pd.DataFrame([
        {
            "cpu_usage_percent": 85.0,  # Very high CPU
            "memory_usage_mb": 1800.0,  # High memory
            "network_calls_count": 450,  # Excessive network calls
            "file_operations_count": 150,  # Many file operations
            "api_calls_count": 800,  # Excessive API calls
            "error_count": 25,  # Many errors
            "hour_of_day": 3,  # Middle of night
            "day_of_week": 6,  # Weekend
            "query_diversity": 0.15,  # Low diversity (repetitive)
            "response_time_ms": 450.0,  # Slow responses
            "success_rate": 0.62  # Low success rate
        }
    ])


def predict_threat(model, engineer, agent_data):
    """
    Predict threat score for agent behavioral data
    
    Args:
        model: Trained Isolation Forest model
        engineer: Feature engineer with fitted scaler
        agent_data: DataFrame with agent behavioral metrics
        
    Returns:
        Threat score (0-1, higher = more threatening)
    """
    # Extract features
    features = engineer.extract_features(agent_data)
    
    # Scale features
    X, feature_names = engineer.transform(features)
    
    # Get anomaly score
    scores_raw = model.decision_function(X)
    # Normalize to [0, 1] (higher = more anomalous/threatening)
    threat_scores = 1 / (1 + np.exp(scores_raw))
    
    return threat_scores[0], feature_names, X[0]


def main():
    """Main demonstration"""
    print("=" * 70)
    print("OPTIKAL INFERENCE EXAMPLE")
    print("=" * 70)
    
    # Load model
    print("\n[1/3] Loading model...")
    model, engineer = load_model()
    
    # Test normal behavior
    print("\n[2/3] Testing NORMAL agent behavior...")
    normal_data = create_sample_agent_data()
    normal_threat_score, feature_names, normal_features = predict_threat(
        model, engineer, normal_data
    )
    
    print(f"\n  Agent Metrics:")
    print(f"    CPU: {normal_data['cpu_usage_percent'].values[0]:.1f}%")
    print(f"    Memory: {normal_data['memory_usage_mb'].values[0]:.0f} MB")
    print(f"    API Calls: {normal_data['api_calls_count'].values[0]}")
    print(f"    Query Diversity: {normal_data['query_diversity'].values[0]:.2f}")
    
    print(f"\n  Threat Score: {normal_threat_score:.4f}")
    if normal_threat_score < 0.5:
        print(f"  Status: ✅ NORMAL - No threat detected")
    else:
        print(f"  Status: ⚠️  SUSPICIOUS")
    
    # Test suspicious behavior
    print("\n[3/3] Testing SUSPICIOUS agent behavior...")
    suspicious_data = create_suspicious_agent_data()
    suspicious_threat_score, _, suspicious_features = predict_threat(
        model, engineer, suspicious_data
    )
    
    print(f"\n  Agent Metrics:")
    print(f"    CPU: {suspicious_data['cpu_usage_percent'].values[0]:.1f}%")
    print(f"    Memory: {suspicious_data['memory_usage_mb'].values[0]:.0f} MB")
    print(f"    API Calls: {suspicious_data['api_calls_count'].values[0]}")
    print(f"    Query Diversity: {suspicious_data['query_diversity'].values[0]:.2f}")
    
    print(f"\n  Threat Score: {suspicious_threat_score:.4f}")
    if suspicious_threat_score >= 0.7:
        print(f"  Status: 🚨 HIGH THREAT - Immediate action required!")
    elif suspicious_threat_score >= 0.5:
        print(f"  Status: ⚠️  MODERATE THREAT - Investigation recommended")
    else:
        print(f"  Status: ✅ NORMAL")
    
    print("\n" + "=" * 70)
    print("✓ OPTIKAL INFERENCE COMPLETE")
    print("=" * 70)
    
    # Show feature importance
    print("\nTop 5 Most Important Features:")
    # This is simplified - in practice you'd use SHAP or model-specific importance
    feature_diffs = np.abs(suspicious_features - normal_features)
    top_indices = np.argsort(feature_diffs)[-5:][::-1]
    
    for idx in top_indices:
        print(f"  - {feature_names[idx]}: {feature_diffs[idx]:.4f}")


if __name__ == "__main__":
    main()
