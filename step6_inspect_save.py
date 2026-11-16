import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
y_pred = model.predict(X_test)

# --- Plot: actual vs predicted over the two-week test window ---
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(test["timestamp"], y_test,  color="#1f77b4", linewidth=1.2, label="Actual")
ax.plot(test["timestamp"], y_pred,  color="#ff7f0e", linewidth=1.0,
        linestyle="--", alpha=0.85, label="Predicted")

ax.set_title("Actual vs Predicted Electricity Consumption\n(Test window: 16 Feb – 29 Feb 2016)",
             fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Electricity (kWh)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.xticks(rotation=30)
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate MAE on the plot
mae  = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
ax.text(0.01, 0.97, f"MAE = {mae:.2f} kWh   RMSE = {rmse:.2f} kWh",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
plot_path = "models/actual_vs_predicted.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot saved → {plot_path}")

# --- Save model ---
model_path = "models/gbr_model.joblib"
joblib.dump(model, model_path)
print(f"Model saved → {model_path}")

# --- Save feature order ---
feature_order_path = "models/feature_order.json"
with open(feature_order_path, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"Feature order saved → {feature_order_path}")

# --- Confirm files exist ---
import os
for path in [model_path, feature_order_path, plot_path]:
    size = os.path.getsize(path)
    print(f"  {path}  ({size:,} bytes)")

# --- Quick error analysis ---
errors = y_pred - y_test.values
print("\n=== Prediction error distribution (kWh) ===")
print(f"  Mean error (bias) : {errors.mean():+.3f}  (near 0 = unbiased)")
print(f"  Std of errors     : {errors.std():.3f}")
print(f"  Worst over-predict: {errors.max():+.3f}")
print(f"  Worst under-predict:{errors.min():+.3f}")
print(f"  % of hours within ±2 kWh: {(np.abs(errors) <= 2).mean()*100:.1f}%")
