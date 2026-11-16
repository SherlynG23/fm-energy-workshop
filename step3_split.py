import pandas as pd

# --- Reproduce the prepared dataset from Step 2 ---
df = pd.read_csv("Data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

df["hour"]        = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.day_of_week

df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)

df_model = df.dropna().reset_index(drop=True)

# --- Define the cutoff: last two weeks = 14 days = 336 hours ---
cutoff = df_model["timestamp"].max() - pd.Timedelta(weeks=2)
print(f"Test set starts at: {cutoff + pd.Timedelta(hours=1)}")
print(f"Data ends at:       {df_model['timestamp'].max()}")

# --- Split ---
train = df_model[df_model["timestamp"] <= cutoff].reset_index(drop=True)
test  = df_model[df_model["timestamp"] >  cutoff].reset_index(drop=True)

feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
target_col   = "electricity_kwh"

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

print("\n=== Split summary ===")
print(f"Train rows : {len(train)}  ({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
print(f"Test rows  : {len(test)}   ({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

print("\n=== No data leakage check ===")
print(f"Latest train timestamp : {train['timestamp'].max()}")
print(f"Earliest test timestamp: {test['timestamp'].min()}")
gap = test['timestamp'].min() - train['timestamp'].max()
print(f"Gap between them       : {gap}  (should be exactly 1 hour)")

print("\n=== Target (electricity_kwh) stats by split ===")
print(f"{'':12} {'mean':>6} {'std':>6} {'min':>6} {'max':>6}")
print(f"{'Train':12} {y_train.mean():6.2f} {y_train.std():6.2f} {y_train.min():6.2f} {y_train.max():6.2f}")
print(f"{'Test':12} {y_test.mean():6.2f}  {y_test.std():6.2f} {y_test.min():6.2f} {y_test.max():6.2f}")
