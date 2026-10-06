"""
Predictive Analytics Using Historical Data
Steps: load -> clean -> features -> train/test split -> model -> evaluate -> forecast -> plot

Usage:
    python predictive_analytics.py                 # runs on built-in sample data
    python predictive_analytics.py sales.csv       # your CSV (needs 'date' and 'value' columns)
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- 1. LOAD
def load_data(path=None):
    if path:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        return df[["date", "value"]]
    # Sample data: 5 years monthly sales with trend + seasonality + noise
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    t = np.arange(60)
    value = 200 + 4 * t + 25 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 8, 60)
    df = pd.DataFrame({"date": dates, "value": value})
    df.loc[[10, 25], "value"] = np.nan   # missing values
    df.loc[40, "value"] = 900            # outlier
    return df

# ---------------- 2. CLEAN
def clean(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")
    df = df.set_index("date")
    q1, q3 = df["value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (df["value"] < q1 - 3 * iqr) | (df["value"] > q3 + 3 * iqr)
    print(f"Outliers removed: {int(mask.sum())}")
    df.loc[mask, "value"] = np.nan
    print(f"Missing values filled: {int(df['value'].isna().sum())}")
    df["value"] = df["value"].interpolate(method="time").bfill().ffill()
    return df

# ---------------- 3. FEATURES
def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["t"] = np.arange(len(df))
    f["month"] = df.index.month
    f["sin"] = np.sin(2 * np.pi * f["month"] / 12)
    f["cos"] = np.cos(2 * np.pi * f["month"] / 12)
    return f

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<18} MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:5.2f}%  R2={r2:6.3f}")
    return rmse

# ---------------- 4. MAIN
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = clean(load_data(path))
    X, y = make_features(df), df["value"]

    # time-ordered split (never shuffle time series)
    split = int(len(df) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }
    print("\nModel evaluation on test set:")
    scores, preds = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds[name] = m.predict(X_te)
        scores[name] = evaluate(name, y_te.values, preds[name])

    best = min(scores, key=scores.get)
    print(f"\nBest model: {best}")

    # refit best model on all data, forecast next 12 periods
    model = models[best].fit(X, y)
    freq = pd.infer_freq(df.index) or "MS"
    future_idx = pd.date_range(df.index[-1], periods=13, freq=freq)[1:]
    fut = pd.DataFrame(index=future_idx)
    fut["t"] = np.arange(len(df), len(df) + 12)
    fut["month"] = fut.index.month
    fut["sin"] = np.sin(2 * np.pi * fut["month"] / 12)
    fut["cos"] = np.cos(2 * np.pi * fut["month"] / 12)
    forecast = pd.Series(model.predict(fut[X.columns]), index=future_idx)
    forecast.to_csv("forecast_next_12.csv", header=["forecast"])
    print("\nNext 12 periods forecast:\n", forecast.round(2))

    # ---------------- 5. VISUALIZE
    fig, ax = plt.subplots(2, 1, figsize=(11, 9))
    ax[0].plot(df.index, y, label="Historical", color="steelblue")
    ax[0].plot(y_te.index, preds[best], "--", label=f"Test prediction ({best})", color="orange")
    ax[0].plot(forecast.index, forecast, label="12-month forecast", color="green", lw=2)
    ax[0].axvline(y_te.index[0], color="grey", ls=":", label="Train/Test split")
    ax[0].set_title("Historical Data, Predictions and Forecast")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].scatter(y_te, preds[best], color="purple")
    lims = [min(y_te.min(), preds[best].min()), max(y_te.max(), preds[best].max())]
    ax[1].plot(lims, lims, "r--")
    ax[1].set_xlabel("Actual"); ax[1].set_ylabel("Predicted")
    ax[1].set_title("Actual vs Predicted (closer to red line = better)")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("forecast_plot.png", dpi=150)
    print("\nSaved: forecast_plot.png, forecast_next_12.csv")

if __name__ == "__main__":
    main()
