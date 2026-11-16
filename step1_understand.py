import pandas as pd

# Load the data
df = pd.read_csv("Data/one_meter.csv", parse_dates=["timestamp"])

print("=== Shape ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== Column types ===")
print(df.dtypes)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Date range ===")
print(f"From: {df['timestamp'].min()}")
print(f"To:   {df['timestamp'].max()}")
total_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600 + 1
print(f"Expected hours: {int(total_hours)}, Actual rows: {len(df)}")

print("\n=== Electricity consumption summary (kWh) ===")
print(df["electricity_kwh"].describe().round(2))

print("\n=== Missing values ===")
print(df.isnull().sum())

print("\n=== Duplicate timestamps ===")
print(f"Duplicate timestamps: {df['timestamp'].duplicated().sum()}")

print("\n=== Is the series evenly spaced (1-hour gaps)? ===")
gaps = df["timestamp"].diff().dropna()
non_hourly = gaps[gaps != pd.Timedelta("1h")]
if non_hourly.empty:
    print("Yes — every row is exactly 1 hour apart.")
else:
    print(f"No — {len(non_hourly)} irregular gap(s) found:")
    print(non_hourly)

print("\n=== Unique meter and building IDs ===")
print(f"Meters:    {df['meter_id'].unique()}")
print(f"Buildings: {df['building_id'].unique()}")
