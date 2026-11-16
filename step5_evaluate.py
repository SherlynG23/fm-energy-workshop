import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# --- Reproduce prepared dataset ---
df = pd.read_csv("Data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week
df["lag_1h"]      = df["electricity_kwh"].shift(1)
df["lag_2h"]      = df["electricity_kwh"].shift(2)
df["lag_3h"]      = df["electricity_kwh"].shift(3)
df["lag_24h"]     = df["electricity_kwh"].shift(24)
df["lag_168h"]    = df["electricity_kwh"].shift(168)

df_model = df.dropna().reset_index(drop=True)

# --- Split ---
cutoff = df_model["timestamp"].max() - pd.Timedelta(weeks=2)
train  = df_model[df_model["timestamp"] <= cutoff].reset_index(drop=True)
test   = df_model[df_model["timestamp"] >  cutoff].reset_index(drop=True)

feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col   = "electricity_kwh"

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

# --- Train ---
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# --- Model predictions on test set ---
y_pred = model.predict(X_test)

# --- Baseline: predict next hour = previous hour (lag_1h) ---
y_baseline = test["lag_1h"]

# --- Metrics ---
def metrics(actual, predicted, name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = root_mean_squared_error(actual, predicted)
    return name, mae, rmse

model_name,    model_mae,    model_rmse    = metrics(y_test, y_pred,       "GBR Model")
baseline_name, baseline_mae, baseline_rmse = metrics(y_test, y_baseline,   "Baseline (lag_1h)")

print("=== Evaluation on test set (16 Feb – 29 Feb 2016, 336 rows) ===")
print()
print(f"{'':22} {'MAE (kWh)':>12} {'RMSE (kWh)':>12}")
print("-" * 48)
print(f"{baseline_name:22} {baseline_mae:12.3f} {baseline_rmse:12.3f}")
print(f"{model_name:22} {model_mae:12.3f} {model_rmse:12.3f}")
print()

mae_improvement  = (1 - model_mae  / baseline_mae)  * 100
rmse_improvement = (1 - model_rmse / baseline_rmse) * 100
print(f"Model improvement over baseline:")
print(f"  MAE  reduced by {mae_improvement:.1f}%")
print(f"  RMSE reduced by {rmse_improvement:.1f}%")

print()
print("=== Context: test set mean consumption ===")
print(f"  Mean kWh : {y_test.mean():.2f}")
print(f"  Model MAE as % of mean : {model_mae / y_test.mean() * 100:.1f}%")
