import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import yfinance as yf



# 1. HELPER FUNCTIONS
def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_stock_data(symbol):
    """Downloads stock data from Yahoo Finance"""
    data = yf.download(symbol, start="2018-01-01", end="2026-05-02")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    if data.empty:
        print(f"Warning: No data found for {symbol}.")
        return None
        
    data.dropna(inplace=True)
    print(f"--- Data ready for {symbol} ---")
    return data

def add_technical_indicators(df):
    """Adds technical indicators as features to the dataframe"""
    df = df.copy()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(df['Close'], 14)    
    
    # Money Flow & Volatility
    df['Money_Flow'] = df['Close'] * df['Volume']
    df["Volatility"] = (df["High"] - df["Low"]) / df["Close"]
    df["Body_Change"] = (df["Close"] - df["Open"]) / df["Open"]
    df["MA_Deviation"] = (df["Close"] - df["MA5"]) / df["MA5"]
    
    # Bollinger Bands
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['Middle_Band'] + (df['STD'] * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['STD'] * 2)
    df['Bollinger_Purity'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # On-Balance Volume (OBV)
    df['Prev_Close'] = df['Close'].shift(1)
    df['Volume_Direction'] = np.where(df['Close'] > df['Prev_Close'], df['Volume'],
                             np.where(df['Close'] < df['Prev_Close'], -df['Volume'], 0))
    df['OBV'] = df['Volume_Direction'].cumsum()
    
    df.dropna(inplace=True) 
    return df

def add_target_column(df):
    """Adds the target column for classification (1: Price Up, 0: Price Down/Equal)"""
    df = df.copy()
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Day_Close'] > df['Close']).astype(int)
    df.drop(columns=['Next_Day_Close'], inplace=True)
    return df

# 2. DATA PREPARATION
symbol = input("Please enter the stock ticker (e.g., AAPL or EREGL.IS): ")
raw_data = fetch_stock_data(symbol)

if raw_data is not None:
    featured_data = add_technical_indicators(raw_data)
    final_data = add_target_column(featured_data)
    
    features = ['MA5', 'RSI', 'Money_Flow', 'Volatility', 'Body_Change', 'MA_Deviation', 'Bollinger_Purity', 'OBV']
    
    # Separate the latest row for real-time prediction
    latest_data = final_data[features].tail(1)
    
    # Prepare training data (dropping the last row where Target is NaN)
    training_set = final_data.dropna(subset=['Target'])
    
    X = training_set[features]
    y = training_set["Target"]

    # Time Series Split (No shuffle for time-series data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, shuffle=False)

    # 3. MODELS AND HYPERPARAMETERS
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    param_grids = {
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10], "model__solver": ['liblinear']},
        "SVM": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf"], "model__gamma": ["scale"]},
        "Random Forest": {"model__n_estimators": [100, 300], "model__max_depth": [5, 10]},
        "XGBoost": {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__subsample": [0.8]},
        "KNN": {"model__n_neighbors": [10, 20, 30], "model__weights": ["uniform", "distance"]}
    }

    best_model = None
    best_score = 0
    best_model_name = ""

    # 4. TRAINING LOOP
    for name, model in models.items():
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Pipeline for Scaling and Modeling
        pipeline = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        
        if name in param_grids:
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=tscv, n_jobs=1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            clf = grid_search.best_estimator_
        else:
            clf = pipeline.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        current_score = accuracy_score(y_test, y_pred)
        
        print(f"{name} Accuracy Score: {current_score:.4f}")

        if current_score > best_score:
            best_score = current_score
            best_model = clf
            best_model_name = name

    # 5. FINAL RESULTS AND BOT PREDICTION
    print("-" * 30)
    print(f"🏆 Best Performing Model: {best_model_name}")
    print(f"🎯 Highest Accuracy Score: {best_score:.4f}")
    print("-" * 30)

    # Bot Decision for Tomorrow
    prediction = best_model.predict(latest_data)
    if prediction[0] == 1:
        print(f"🚀 Bot Prediction: {symbol} is likely to be BULLISH (UP) tomorrow!")
    else:
        print(f"📉 Bot Prediction: {symbol} is likely to be BEARISH (DOWN) tomorrow!")


