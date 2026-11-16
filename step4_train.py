import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

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

print("=== Model trained successfully ===")
print(f"Algorithm : GradientBoostingRegressor")
print(f"random_state : 42  (results are reproducible)")
print(f"Training rows used : {len(X_train)}")
print(f"Features           : {feature_cols}")

print("\n=== Model hyperparameters (defaults) ===")
params = model.get_params()
for k in ["n_estimators", "learning_rate", "max_depth", "min_samples_leaf", "subsample"]:
    print(f"  {k:20s}: {params[k]}")

print("\n=== Training set R² (how well it fits what it was trained on) ===")
train_r2 = model.score(X_train, y_train)
print(f"  R² = {train_r2:.4f}  (1.0 = perfect fit)")

print("\n=== Feature importances (which inputs matter most) ===")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
for feat, imp in importances.items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:12s} {imp:.4f}  {bar}")
