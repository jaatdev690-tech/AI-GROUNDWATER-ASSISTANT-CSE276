
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Read both datasets
file1 = "Data/gwl_manual_quarterly_cgwb_pb_1991_2020 (1).csv"
file2 = "Data/gwl_manual_quarterly_cgwb_pb_2021_2025.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)

# Clean column names
df.columns = df.columns.str.strip()

# Convert date and groundwater level to proper formats
date_col = "Data Acquisition Time"
level_col = "Groundwater Level Quarterly Manual (meter)"

df[date_col] = pd.to_datetime(
    df[date_col], errors="coerce", dayfirst=True
)

df[level_col] = pd.to_numeric(df[level_col], errors="coerce")

# Remove rows with missing required values
df = df.dropna(subset=["District", date_col, level_col])

# Canonical District Normalization: Unify MOHALI and SAS NAGAR
df["District_Original"] = df["District"].astype(str).str.strip().str.upper()
df["District"] = df["District_Original"].replace({
    "MOHALI": "SAS NAGAR (MOHALI)",
    "SAS NAGAR": "SAS NAGAR (MOHALI)"
})

# Extract year
df["Year"] = df[date_col].dt.year

# Calculate yearly average for each district
yearly = df.groupby(
    ["Year", "District"]
)[level_col].mean().reset_index()

# Save yearly statistical results
yearly.to_csv("yearly_groundwater_summary.csv", index=False)

# Calculate Punjab-wide average for each year
state_yearly = df.groupby("Year")[level_col].mean()

# Plot historical trend
plt.figure(figsize=(12, 6))
plt.plot(
    state_yearly.index,
    state_yearly.values,
    marker="o"
)

plt.title("Year-wise Average Groundwater Level in Punjab")
plt.xlabel("Year")
plt.ylabel("Average Groundwater Level (meters)")
plt.grid(True)
plt.tight_layout()
plt.savefig("groundwater_yearly_trend.png", dpi=300)
plt.show()

print("Yearly analysis completed!")
print("Saved: yearly_groundwater_summary.csv")
print("Saved: groundwater_yearly_trend.png")
