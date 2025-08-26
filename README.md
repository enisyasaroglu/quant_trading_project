# QMind Quant Platform

A modular, event-driven, and data-driven quantitative trading platform built from scratch in Python. This project provides a complete end-to-end framework for developing, backtesting, optimizing, and deploying algorithmic trading strategies, including those powered by machine learning.

## Key Features

- **Event-Driven Backtesting Engine:** A high-fidelity engine that processes historical market data chronologically to provide realistic strategy simulations and avoid look-ahead bias.
- **Machine Learning Strategy Integration:** Complete workflow for developing and deploying ML-based strategies:
  - **Feature Engineering:** Robust pipeline to create predictive features from raw OHLCV data.
  - **Model Training:** Jupyter Notebook environment for training and evaluating models (Random Forest, XGBoost).
  - **Live Prediction:** Seamless integration of trained models into a live trading strategy class.
- **Live Paper Trading:** Real-time execution of trades against the Alpaca paper trading API, driven by a live WebSocket data stream.
- **Advanced Systems Engineering:**
  - **Portfolio-Level Risk Management:** A max-drawdown "kill-switch" to protect capital by liquidating positions when risk limits are breached.
  - **Hyperparameter Optimization:** An automated system using Optuna to find the most robust parameters for trading strategies.
- **Real-time UI Dashboard:** A Streamlit-based web dashboard for monitoring the live system's status, portfolio value, current positions, and event logs.
- **Automated Testing:** Unit tests using `pytest` to ensure the reliability of core components.

## Architecture

The system is built on a modular, event-driven architecture. Components such as the data handler, strategy, portfolio manager, and execution handler are decoupled and communicate via a central event queue. This design allows for high cohesion, low coupling, and makes it easy to swap components (e.g., switching from a historical data handler to a live one).



## Tech Stack

- **Core:** Python 3.10+
- **Data & ML:** Pandas, NumPy, Scikit-learn, XGBoost, Pandas-TA
- **Optimization:** Optuna
- **Live Integration:** Alpaca Trade API, Redis
- **UI:** Streamlit
- **Development & Testing:** Poetry, Pytest, Docker, Git & GitHub

## Getting Started

### Prerequisites

- Python 3.10+
- Conda or another virtual environment manager
- Docker Desktop (for Redis)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/qmind_quant_platform.git](https://github.com/YOUR_USERNAME/qmind_quant_platform.git)
    cd qmind_quant_platform
    ```
2.  **Install dependencies using Poetry:**
    ```bash
    poetry install
    ```
3.  **Set up API Keys:**
    Create a `.env` file in the project root and add your Alpaca paper trading keys:
    ```ini
    APCA_API_KEY_ID="PK..."
    APCA_API_SECRET_KEY="..."
    ```
4.  **Start Redis:**
    ```bash
    docker run -d --name qmind-redis -p 6379:6379 redis
    ```

## Usage

### 1. Data Ingestion & Feature Engineering
First, download historical data and create features for the ML models.
```bash
python scripts/run_ingestion.py
python scripts/run_feature_engineering.py
```

### 2. Model Training
Train the ML models using the Jupyter Notebooks located in `notebooks/04_ml_research/`.

### 3. Run a Backtest
Execute a backtest using a trained model. The HTML report will be saved in the `reports/` directory.
```bash
python scripts/run_backtest.py
```

### 4. Run Strategy Optimization
Find the best parameters for a strategy (e.g., the Moving Average Crossover).
```bash
python scripts/run_optimization.py
```

### 5. Run the Live Trading System
This requires two separate terminals. Run these commands during US market hours.

**Terminal 1: Start the Trading Engine**
```bash
python scripts/run_live_trading.py
```
**Terminal 2: Start the UI Dashboard**
```bash
streamlit run dashboard.py
```

### 6. Run Tests
To verify the integrity of the components:
```bash
pytest
```