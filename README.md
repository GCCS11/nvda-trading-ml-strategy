# NVDA Deep Learning Trading Strategy

A systematic trading strategy using deep learning models (MLP, CNN, LSTM) trained on engineered time series features for NVDA stock.

Project Structure
```
├── data/               # Data storage
│   ├── raw/           # Raw NVDA data (15 years)
│   └── processed/     # Processed features and splits
├── src/               # Source code
│   ├── models/        # Neural network architectures (MLP, CNN, LSTM)
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── training.py
│   ├── backtesting.py
│   └── utils.py
├── notebooks/         # Jupyter notebooks
│   └── 03_data_drift_analysis.ipynb
├── config/           # Configuration files
│   └── config.yaml
├── models/           # Saved model weights
│   └── saved/
├── reports/          # Analysis outputs and visualizations
└── report/           # Executive report (deliverable)
    └── Executive_Report.md
```

## Setup and Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB RAM

### Installation
```bash
# Clone repository
git clone https://github.com/GCCS11/nvda-trading-ml-strategy.git
cd nvda-trading-ml-strategy

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install --no-cache-dir -r requirements.txt

# For GPU support (NVIDIA)
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 1. Download Data
```bash
python src/data_loader.py
```

### 2. Generate Features and Preprocess
```bash
python src/data_preprocessing.py
```

### 3. Train Models
```bash
python src/train_all_models.py
```

### 4. Run Backtesting
```bash
python src/backtesting.py
```

### 5. View Data Drift Analysis
```bash
jupyter notebook notebooks/03_data_drift_analysis.ipynb
```

### 6. View Executive Report
```bash
# The complete analysis and results are in:
report/Executive_Report.md
```

## 👤 Author

Developed as part of Microestructuras y Sistemas de Trading - 003 Advanced Trading Strategies: Deep Learning course project by Gian Carlo Campos Sayavedra.