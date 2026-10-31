import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
possible_paths = [
    "Dataset/jewelry_cleaned.csv",
    "../Dataset/jewelry_cleaned.csv"
]

dataset_path = None

for path in possible_paths:
    if os.path.exists(path):
        dataset_path = path
        break

if dataset_path is None:
    raise FileNotFoundError(
        "Dataset not found. Please check the location of "
        "jewelry_cleaned.csv."
    )

df = pd.read_csv(dataset_path)

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Check required columns
required_columns = [
    "Order.datetime",
    "Price",
    "Category.alias",
    "Main.metal",
    "Main.gem"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Convert date and price columns
df["Order.datetime"] = pd.to_datetime(
    df["Order.datetime"], errors="coerce", utc=True
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

# Remove invalid dates and prices
df = df.dropna(subset=["Order.datetime", "Price"])

# Reports folder of the project (next to the Dataset folder)
reports_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(dataset_path))), "Reports"
)
os.makedirs(reports_dir, exist_ok=True)

# -----------------------------------
# 1. Monthly Average Jewellery Price
# -----------------------------------

df["Month"] = (
    df["Order.datetime"]
    .dt.tz_convert(None)
    .dt.to_period("M")
)

monthly_price = df.groupby("Month")["Price"].mean()

plt.figure(figsize=(10, 5))
monthly_price.plot(marker="o")

plt.title("Monthly Average Jewellery Price")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(reports_dir, "monthly_average_price.png"))
plt.show()

# -----------------------------------
# 2. Monthly Number of Records
# -----------------------------------

monthly_records = df.groupby("Month").size()

plt.figure(figsize=(10, 5))
monthly_records.plot(kind="bar")

plt.title("Monthly Jewellery Records")
plt.xlabel("Month")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(reports_dir, "monthly_records.png"))
plt.show()

# -----------------------------------
# 3. Most Common Jewellery Categories
# -----------------------------------

category_counts = df["Category.alias"].value_counts()

plt.figure(figsize=(10, 5))
category_counts.plot(kind="bar")

plt.title("Jewellery Category Analysis")
plt.xlabel("Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(reports_dir, "category_analysis.png"))
plt.show()

# -----------------------------------
# 4. Jewellery Metal Analysis
# -----------------------------------

metal_counts = df["Main.metal"].value_counts()

plt.figure(figsize=(8, 5))
metal_counts.plot(kind="bar")

plt.title("Jewellery Metal Analysis")
plt.xlabel("Metal")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(reports_dir, "metal_analysis.png"))
plt.show()

# -----------------------------------
# 5. Print Summary
# -----------------------------------

print("\n====== JEWELLERY TREND ANALYSIS ======")

print("\nMonthly Average Price:")
print(monthly_price.round(2))

print("\nMonthly Number of Records:")
print(monthly_records)

print("\nMost Common Jewellery Categories:")
print(category_counts)

print("\nMetal-wise Record Counts:")
print(metal_counts)

print("\nTrend analysis completed!")
print("Check the Reports folder for graphs.")