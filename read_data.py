
import pandas as pd
from pathlib import Path

data_folder = Path("Data")
files = [f for f in data_folder.glob("*.csv") if "searchable_corpus" not in f.name]

if not files:
    print("No CSV files found!")
    raise SystemExit

frames = []

for file in files:
    df = pd.read_csv(file)
    df["Source File"] = file.name
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

level_col = "Groundwater Level Quarterly Manual (meter)"

data[level_col] = pd.to_numeric(
    data[level_col], errors="coerce"
)

# Remove rows without district or groundwater level
clean = data.dropna(
    subset=["District", level_col]
).copy()

# Canonical District Normalization: Unify MOHALI and SAS NAGAR
clean["District_Original"] = clean["District"].astype(str).str.strip().str.upper()
clean["District"] = clean["District_Original"].replace({
    "MOHALI": "SAS NAGAR (MOHALI)",
    "SAS NAGAR": "SAS NAGAR (MOHALI)"
})

# District-wise summary
summary = (
    clean.groupby("District")[level_col]
    .agg(["count", "mean", "min", "max"])
    .round(2)
    .sort_values("mean")
)

print("\n===== DISTRICT-WISE GROUNDWATER SUMMARY =====")
print(summary.to_string())

# Save results
output_file = Path("district_groundwater_summary.csv")
summary.to_csv(output_file)

print("\nAnalysis completed!")
print("Saved to:", output_file.resolve())
