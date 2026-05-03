# 📈 AI-Powered Stock Direction Predictor

This project is a machine learning-based tool designed to predict the **next-day price direction** (Bullish or Bearish) of stocks and commodities using historical data and technical indicators. It compares multiple algorithms to find the most accurate model for a specific asset.

---

## 🚀 Key Features

*   **Automated Data Pipeline:** Fetches real-time historical data using the `yfinance` API.
*   **Feature Engineering:** Calculates 8+ technical indicators, including:
    *   **RSI (Relative Strength Index):** To identify overbought/oversold conditions.
    *   **OBV (On-Balance Volume):** To track momentum and capital flow.
    *   **Bollinger Bands:** To analyze volatility and price purity.
    *   **Moving Averages (MA5):** To detect short-term trends.
*   **Multi-Model Comparison:** Automatically trains and evaluates:
    *   Logistic Regression, Random Forest, SVM, KNN, and XGBoost.
*   **Hyperparameter Tuning:** Uses `GridSearchCV` and `TimeSeriesSplit` for optimized performance.
*   **Time-Series Aware:** Specifically designed for financial data by avoiding shuffling to prevent data leakage.

---

## 🧠 How It Works

The core logic follows a supervised learning approach where the model learns from historical "signals" to predict a future "outcome."

### 1. Data Alignment (The Shift Logic)
To predict tomorrow, we shift the price data by **-1**. This allows the model to map **today's indicators ($X_{today}$)** to **tomorrow's price direction ($Y_{tomorrow}$)**.

> **Target Logic:** If $Price_{t+1} > Price_{t}$ then Target = 1 (Bullish), else 0 (Bearish).

### 2. The Training Pipeline
1.  **Scaling:** All features are normalized using `StandardScaler` within a `Pipeline` to ensure fair competition between indicators like OBV (millions) and Purity (0-1).
2.  **Cross-Validation:** Uses `TimeSeriesSplit` to ensure the model is always tested on "future" data during the tuning phase.
3.  **Optimization:** Automatically finds the best parameters (e.g., C for LR, n_estimators for XGBoost).

---

## 📊 Performance Insight

During testing, the model showed varying success rates depending on the asset's volatility:
*   **Gold (GC=F):** Achieved up to **58% accuracy** using Logistic Regression.
*   **Tech Stocks (NVDA, AAPL):** Often performed best with ensemble methods like XGBoost.

---

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install pandas numpy yfinance scikit-learn xgboost
