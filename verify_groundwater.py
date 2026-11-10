
from pathlib import Path
import pandas as pd

DATA_DIR = Path("Data")

files = [
    DATA_DIR / "gwl_manual_quarterly_cgwb_pb_1991_2020 (1).csv",
    DATA_DIR / "gwl_manual_quarterly_cgwb_pb_2021_2025.csv",
]

level_col = "Groundwater Level Quarterly Manual (meter)"

frames = []

for file in files:
    if not file.exists():
        raise FileNotFoundError(f"CSV file not found: {file}")

    df = pd.read_csv(file)
    df["Source File"] = file.name
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

data["District_Original"] = data["District"].astype("string").str.strip().str.upper()
data["District"] = data["District_Original"].replace({
    "MOHALI": "SAS NAGAR (MOHALI)",
    "SAS NAGAR": "SAS NAGAR (MOHALI)"
})
data[level_col] = pd.to_numeric(data[level_col], errors="coerce")

data = data.dropna(subset=["District", level_col])

print("\n===== DATASET CHECK =====")
print("Valid measurement records:", len(data))
print("Districts:", data["District"].nunique())
print("Source files:", data["Source File"].nunique())

print("\n===== LUDHIANA CHECK =====")
ludhiana = data[data["District"] == "LUDHIANA"]

if not ludhiana.empty:
    print("Valid records:", len(ludhiana))
    print("Calculated average:", round(ludhiana[level_col].mean(), 2), "mbgl")
    print("Minimum:", ludhiana[level_col].min(), "mbgl")
    print("Maximum:", ludhiana[level_col].max(), "mbgl")
else:
    print("No Ludhiana records found.")

print("\n===== DISTRICTS WITH HIGHEST AVERAGE =====")
district_stats = (
    data.groupby("District")[level_col]
    .agg(["mean", "min", "max", "count"])
    .sort_values("mean", ascending=False)
)

print(district_stats.head(10).round(2).to_string())

print("\n===== DISTRICT WITH HIGHEST INDIVIDUAL MEASUREMENT =====")
idx = data[level_col].idxmax()
print(data.loc[idx, ["District", level_col, "Source File"]].to_string())
