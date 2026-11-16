import pandas as pd

# Load and sort by time
df = pd.read_csv("Data/one_meter.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# --- Calendar features ---
df["hour"]        = df["timestamp"].dt.hour          # 0-23
df["day_of_week"] = df["timestamp"].dt.day_of_week   # Monday=0, Sunday=6

# --- Lag features (look back N hours) ---
df["lag_1h"]   = df["electricity_kwh"].shift(1)
df["lag_2h"]   = df["electricity_kwh"].shift(2)
df["lag_3h"]   = df["electricity_kwh"].shift(3)
df["lag_24h"]  = df["electricity_kwh"].shift(24)
df["lag_168h"] = df["electricity_kwh"].shift(168)

# --- Drop first 168 rows (they lack full lag history) ---
df_model = df.dropna().reset_index(drop=True)

print("=== Rows before dropping incomplete history ===")
print(f"Original rows : {len(df)}")
print(f"Rows dropped  : {len(df) - len(df_model)}  (first 168 hours = 7 days)")
print(f"Rows remaining: {len(df_model)}")

print("\n=== Date range after drop ===")
print(f"From: {df_model['timestamp'].min()}")
print(f"To:   {df_model['timestamp'].max()}")

print("\n=== Feature columns ===")
feature_cols = ["hour", "day_of_week", "lag_1h", "lag_2h", "lag_3h", "lag_24h", "lag_168h"]
print(feature_cols)

print("\n=== Sample rows (first 3) ===")
print(df_model[["timestamp", "electricity_kwh"] + feature_cols].head(3).to_string(index=False))

print("\n=== Missing values after drop ===")
print(df_model[feature_cols + ["electricity_kwh"]].isnull().sum())

print("\n=== Feature statistics ===")
print(df_model[feature_cols].describe().round(2))
