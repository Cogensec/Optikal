"""
ONNX Model Exporter for Optikal

Exports trained Optikal models to ONNX format for deployment with NVIDIA Triton Inference Server.
"""

import numpy as np
import joblib
import json
from pathlib import Path

try:
    import tensorflow as tf
    import tf2onnx
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ERROR: ONNX tools not available. Install with: pip install tf2onnx onnx")


def export_isolation_forest_to_onnx(
    model_path: str,
    output_path: str,
    n_features: int = 18
):
    """
    Export Isolation Forest to ONNX
    
    Note: Scikit-learn models require skl2onnx for ONNX export
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("ERROR: skl2onnx not available. Install with: pip install skl2onnx")
        return False
    
    print(f"\n=== Exporting Isolation Forest to ONNX ===")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    
    # Load model
    model = joblib.load(model_path)
    
    # Define input type
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=13
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    print(f"✓ Isolation Forest exported to ONNX")
    return True


def export_lstm_to_onnx(
    model_path: str,
    output_path: str,
    sequence_length: int = 10,
    n_features: int = 18
):
    """
    Export LSTM model to ONNX
    """
    if not ONNX_AVAILABLE:
        print("ERROR: ONNX tools not available")
        return False
    
    print(f"\n=== Exporting LSTM to ONNX ===")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    
    # Load Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Define input spec
    spec = (tf.TensorSpec((None, sequence_length, n_features), tf.float32, name="input"),)
    
    # Convert to ONNX
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(output_path)
    )
    
    print(f"✓ LSTM exported to ONNX")
    return True


def create_triton_config(
    model_name: str,
    input_name: str,
    input_shape: list,
    output_name: str,
    output_shape: list,
    output_path: str
):
    """
    Create Triton model configuration file
    """
    config = f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 1000
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: {input_shape}
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: {output_shape}
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
dynamic_batching {{
  preferred_batch_size: [ 100, 500, 1000 ]
  max_queue_delay_microseconds: 1000
}}
"""
    
    with open(output_path, 'w') as f:
        f.write(config)
    
    print(f"✓ Created Triton config: {output_path}")


def setup_triton_repository(models_dir: str = "../../models"):
    """
    Set up Triton model repository structure
    """
    print("\n=== Setting Up Triton Model Repository ===")
    
    models_path = Path(models_dir)
    
    # Create directory structure
    if_model_dir = models_path / "optikal_isolation_forest" / "1"
    lstm_model_dir = models_path / "optikal_lstm" / "1"
    
    if_model_dir.mkdir(parents=True, exist_ok=True)
    lstm_model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created model directories")
    print(f"  - {if_model_dir}")
    print(f"  - {lstm_model_dir}")
    
    return if_model_dir, lstm_model_dir


def export_optikal_models(
    models_dir: str = "optikal_models",
    triton_repo: str = "../../models"
):
    """
    Main export pipeline for Optikal models
    """
    print("=" * 70)
    print("OPTIKAL MODEL EXPORT TO ONNX")
    print("=" * 70)
    
    # Load metadata
    metadata_path = f"{models_dir}/optikal_metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    n_features = metadata['n_features']
    sequence_length = metadata['sequence_length']
    
    print(f"\nModel metadata:")
    print(f"  Features: {n_features}")
    print(f"  Sequence length: {sequence_length}")
    
    # Setup Triton repository
    if_model_dir, lstm_model_dir = setup_triton_repository(triton_repo)
    
    # Export Isolation Forest
    if_success = export_isolation_forest_to_onnx(
        model_path=f"{models_dir}/optikal_isolation_forest.pkl",
        output_path=str(if_model_dir / "model.onnx"),
        n_features=n_features
    )
    
    if if_success:
        # Create Triton config
        create_triton_config(
            model_name="optikal_isolation_forest",
            input_name="float_input",
            input_shape=[n_features],
            output_name="output",
            output_shape=[1],
            output_path=str(if_model_dir.parent / "config.pbtxt")
        )
    
    # Export LSTM
    lstm_model_path = f"{models_dir}/optikal_lstm.h5"
    if Path(lstm_model_path).exists():
        lstm_success = export_lstm_to_onnx(
            model_path=lstm_model_path,
            output_path=str(lstm_model_dir / "model.onnx"),
            sequence_length=sequence_length,
            n_features=n_features
        )
        
        if lstm_success:
            # Create Triton config
            create_triton_config(
                model_name="optikal_lstm",
                input_name="input",
                input_shape=[sequence_length, n_features],
                output_name="output",
                output_shape=[1],
                output_path=str(lstm_model_dir.parent / "config.pbtxt")
            )
    else:
        print("\nℹ️  LSTM model not found, skipping LSTM export")
    
    print("\n" + "=" * 70)
    print("✓ OPTIKAL MODEL EXPORT COMPLETE")
    print("=" * 70)
    print(f"\nModels exported to: {triton_repo}/")
    print("\nNext steps:")
    print("1. Start Triton Inference Server:")
    print(f"   docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \\")
    print(f"       -v {Path(triton_repo).absolute()}:/models \\")
    print(f"       nvcr.io/nvidia/tritonserver:24.01-py3 \\")
    print(f"       tritonserver --model-repository=/models")
    print("\n2. Test inference:")
    print("   curl http://localhost:8000/v2/models/optikal_isolation_forest/ready")


if __name__ == "__main__":
    import sys
    
    # Check if models exist
    if not Path("optikal_models").exists():
        print("ERROR: Trained models not found. Run optikal_trainer.py first.")
        sys.exit(1)
    
    # Export models
    export_optikal_models(
        models_dir="optikal_models",
        triton_repo="../../models"
    )
