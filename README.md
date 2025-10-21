# NVDA Deep Learning Trading Strategy

A systematic trading strategy using deep learning models ((MLP, CNN, LSTM) trained on engineered time series features for NVDA stock.

## Project Structure
```
├── data/               # Data storage
│   ├── raw/           # Raw downloaded data
│   └── processed/     # Processed features
├── src/               # Source code
│   ├── models/        # Neural network architectures
│   └── data_loader.py # Data download and validation
├── notebooks/         # Jupyter notebooks for analysis
├── config/           # Configuration files
└── mlruns/          # MLFlow experiment tracking
```

## Setup

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended for training)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/nvda-trading-ml-strategy.git
cd nvda-trading-ml-strategy

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt
```

## Usage

### Download Data
```bash
python src/data_loader.py
```

This downloads 15 years of NVDA historical data from Yahoo Finance.

## Project Status

- [x] Project setup
- [x] Data loader implementation
- [ ] Feature engineering
- [ ] Model implementation (MLP, CNN, LSTM)
- [ ] MLFlow integration
- [ ] Data drift analysis
- [ ] Backtesting engine
- [ ] Executive report

## Author

Developed as part of 003 Advanced Trading Strategies: Deep Learning project by Gian Carlo Campos Sayavedra.

