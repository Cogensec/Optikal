"""
Optikal - GPU-Accelerated AI Threat Detection

A machine learning model for detecting security threats in AI agent behavior.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="optikal",
    version="1.0.0",
    author="Cogensec",
    author_email="info@cogensec.com",
    description="GPU-Accelerated AI Threat Detection for AI Agent Security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cogensec/Optikal",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "lstm": [
            "tensorflow>=2.13.0",
            "keras>=2.13.0",
        ],
        "onnx": [
            "tf2onnx>=1.15.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "skl2onnx>=1.15.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "explain": [
            "shap>=0.42.0",
        ],
        "tracking": [
            "mlflow>=2.7.0",
        ],
        "tuning": [
            "optuna>=3.3.0",
        ],
        "gbm": [
            "lightgbm>=4.0.0",
        ],
        "streaming": [
            "confluent-kafka>=2.2.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "all": [
            "tensorflow>=2.13.0",
            "keras>=2.13.0",
            "tf2onnx>=1.15.0",
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "skl2onnx>=1.15.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "shap>=0.42.0",
            "mlflow>=2.7.0",
            "optuna>=3.3.0",
            "pyyaml>=6.0.0",
            "lightgbm>=4.0.0",
            "confluent-kafka>=2.2.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "optikal-train=optikal.train_quick:main",
            "optikal-export=optikal.export_onnx:main",
        ],
    },
    include_package_data=True,
    keywords="ai security threat-detection machine-learning gpu anomaly-detection lstm",
    project_urls={
        "Bug Reports": "https://github.com/Cogensec/Optikal/issues",
        "Source": "https://github.com/Cogensec/Optikal",
        "Documentation": "https://github.com/Cogensec/Optikal/blob/main/README.md",
    },
)
