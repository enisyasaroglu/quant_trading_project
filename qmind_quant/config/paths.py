# qmind_quant/config/paths.py

from pathlib import Path

# This file is located at: qmind_quant/config/paths.py
# The project root is two levels up from this file's parent directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define other key directories relative to the project root
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "qmind_quant" / "ml_models" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
