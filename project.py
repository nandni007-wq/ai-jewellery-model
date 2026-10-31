# =============================================================
#  JEWELLERY PRICE PREDICTION AND TREND ANALYSIS
#  SYBCA-AI (Sem-III) | 303 Machine Learning | Mini Project
#  Type of problem : Supervised Learning - Regression
# =============================================================
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import sweetviz as sv
except ImportError:          # project still runs if Sweetviz is not installed
    sv = None

warnings.filterwarnings("ignore")
print("imported")

# -------------------------------------------------------------
# Project folders (works from any working directory)
#   Project_Folder/Dataset/dataset.csv
#   Project_Folder/Python_Code/project.py
#   Project_Folder/Reports/
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = None
for folder in [os.path.dirname(SCRIPT_DIR), SCRIPT_DIR, os.getcwd(), os.path.dirname(os.getcwd())]:
    if os.path.exists(os.path.join(folder, "Dataset", "dataset.csv")):
        BASE_DIR = folder
        break
if BASE_DIR is None:
    raise FileNotFoundError(
        "Dataset not found. Please keep 'Dataset/dataset.csv' next to the 'Python_Code' folder."
    )

DATASET_PATH = os.path.join(BASE_DIR, "Dataset", "dataset.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "Dataset", "jewelry_cleaned.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "Reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def save_and_show(file_name):
    """Save the current graph in the Reports folder and show it."""
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, file_name), dpi=100)
    plt.show()
    plt.close()


# =============================================================
# STEP 2 & 3 : DATASET COLLECTION AND UNDERSTANDING THE DATA
# =============================================================
print("=" * 60)
print("JEWELLERY PRICE PREDICTION AND TREND ANALYSIS")
print("=" * 60)

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()          # remove extra spaces from column names

print("\n1. Dataset Shape")
print("Total Rows   :", df.shape[0])
print("Total Columns:", df.shape[1])

print("\n2. First Five Records")
print(df.head())

print("\n3. Dataset Information")
df.info()

print("\nColumn names:", df.columns.tolist())
print("\nData Types")
print(df.dtypes)

quantitative = df.select_dtypes(include=["int64", "float64"])
print("\nQuantitative (Numerical) Columns:", quantitative.columns.tolist())

qualitative = df.select_dtypes(include=["object"])
print("Qualitative (Categorical) Columns:", qualitative.columns.tolist())

print("\nTotal missing values  :", int(df.isnull().sum().sum()))
print("Total duplicate rows  :", int(df.duplicated().sum()))

print("\nStatistical Summary")
print(df.describe(include="all").to_string())
print("\nDataset Inspection Completed!")

# keep the figures of the dataset BEFORE cleaning (for Before vs After comparison)
before_rows = df.shape[0]
before_missing = int(df.isnull().sum().sum())
before_duplicates = int(df.duplicated().sum())
before_cols_with_missing = int((df.isnull().sum() > 0).sum())

# =============================================================
# STEP 5 : HANDLE MISSING VALUES, DUPLICATES AND OUTLIERS
# =============================================================
print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

numerical_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(include=["object", "category"]).columns

# numerical columns -> median ; Price is the target, so rows with missing Price are removed
for col in numerical_columns:
    if df[col].isnull().any():
        if col == "Price":
            print(f"\nMissing values found in target column: {col}")
            print("Rows with missing Price will be removed.")
            df = df.dropna(subset=["Price"])
        else:
            df[col] = df[col].fillna(df[col].median())

# categorical columns -> mode (most frequent value)
for col in categorical_columns:
    if df[col].isnull().any():
        mode_values = df[col].mode()
        if not mode_values.empty:
            df[col] = df[col].fillna(mode_values.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows Before Removal:", int(df.duplicated().sum()))
df = df.drop_duplicates()
print("Duplicate Rows After Removal :", int(df.duplicated().sum()))

# Outliers: the IQR method was applied on the FULL raw data (see dataset_correction.py).
# Here we only check the Price range; rows are NOT removed again so that 9000 records remain.
Q1 = df["Price"].quantile(0.25)
Q3 = df["Price"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
n_outside = int(((df["Price"] < lower) | (df["Price"] > upper)).sum())
print(f"\nPrice range after outlier treatment: {df['Price'].min():.2f} to {df['Price'].max():.2f}")
print(f"Records outside the IQR limits of this sample ({lower:.1f} to {upper:.1f}): {n_outside} (kept)")

df.to_csv(CLEANED_PATH, index=False)
print("\nCleaned dataset saved successfully ->", CLEANED_PATH)

# ---- Before vs After comparison ----
comparison = pd.DataFrame({
    "Metric": ["Total Rows", "Total Columns", "Missing Values", "Columns with Missing Values", "Duplicate Rows"],
    "Before Cleaning": [before_rows, 10, before_missing, before_cols_with_missing, before_duplicates],
    "After Cleaning": [df.shape[0], df.shape[1], int(df.isnull().sum().sum()),
                       int((df.isnull().sum() > 0).sum()), int(df.duplicated().sum())],
})
print("\n=== Before vs After Comparison ===")
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(REPORTS_DIR, "Before_After_Comparison.csv"), index=False)

# =============================================================
# STEP 4 : EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv(CLEANED_PATH)
numerical_columns = df.select_dtypes(include="number").columns.tolist()

# 1. Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"].dropna(), kde=True, edgecolor="black")
plt.title("Histogram of Price", fontsize=16)
plt.xlabel("Price")
plt.ylabel("Frequency")
save_and_show("Histogram.png")

# 2. Box Plot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Price"])
plt.title("Box Plot of Price")
save_and_show("Box_Plot.png")

# 3. Pie Chart
counts = df["Category.alias"].fillna("Unknown").value_counts().head(10)
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%")
plt.title("Pie Chart of Category.alias")
save_and_show("Pie_Chart.png")

# 4. Bar Chart - category wise average price
plt.figure(figsize=(9, 5))
sns.barplot(x="Category.alias", y="Price", data=df, color="skyblue", edgecolor="black")
plt.xticks(rotation=45)
plt.title("Average Price by Category")
save_and_show("Bar_Chart.png")

# 5. Correlation Matrix (columns having a single value, e.g. Quantity, cannot be correlated)
corr_df = df[numerical_columns].copy()
dt = pd.to_datetime(df["Order.datetime"], errors="coerce", utc=True)
corr_df["Year"] = dt.dt.year
corr_df["Month"] = dt.dt.month
corr_df = corr_df.loc[:, corr_df.nunique() > 1]
corr_df = corr_df.drop(columns=["Category.ID", "Purchased.product.ID"], errors="ignore")   # ID columns are not real features
plt.figure(figsize=(7, 5))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
save_and_show("Correlation_Matrix.png")

print("EDA graphs saved in the Reports folder.")

# =============================================================
# TREND ANALYSIS (price and records over time)
# =============================================================
print("\n" + "=" * 60)
print("TREND ANALYSIS")
print("=" * 60)

trend = df.copy()
trend["Order.datetime"] = pd.to_datetime(trend["Order.datetime"], errors="coerce", utc=True)
trend = trend.dropna(subset=["Order.datetime", "Price"])
trend["Month"] = trend["Order.datetime"].dt.tz_convert(None).dt.to_period("M")
trend["Year"] = trend["Order.datetime"].dt.year

monthly_price = trend.groupby("Month")["Price"].mean()
monthly_records = trend.groupby("Month").size()

plt.figure(figsize=(10, 5))
monthly_price.plot(marker="o")
plt.title("Monthly Average Jewellery Price")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.xticks(rotation=45)
save_and_show("monthly_average_price.png")

plt.figure(figsize=(10, 5))
monthly_records.plot(kind="bar")
plt.title("Monthly Jewellery Records")
plt.xlabel("Month")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
save_and_show("monthly_records.png")

category_counts = trend["Category.alias"].value_counts()
plt.figure(figsize=(10, 5))
category_counts.plot(kind="bar")
plt.title("Jewellery Category Analysis")
plt.xlabel("Category")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
save_and_show("category_analysis.png")

metal_counts = trend["Main.metal"].value_counts()
plt.figure(figsize=(8, 5))
metal_counts.plot(kind="bar")
plt.title("Jewellery Metal Analysis")
plt.xlabel("Metal")
plt.ylabel("Number of Records")
plt.xticks(rotation=45)
save_and_show("metal_analysis.png")

yearly_price = trend.groupby("Year")["Price"].agg(["mean", "count"]).round(2)
print("\nYear-wise average price:")
print(yearly_price)

# trend line using only months having at least 30 records (first and last month have very few records)
reliable = monthly_price[monthly_records >= 30]
slope = np.polyfit(np.arange(len(reliable)), reliable.values, 1)[0]
print(f"\nTrend: average price changes by about {slope:+.2f} per month (months with 30+ records)")
print("Months with very few records (average price not reliable):",
      [str(m) for m in monthly_records[monthly_records < 30].index])

print("\nMost Common Jewellery Categories:")
print(category_counts)
print("\nMetal-wise Record Counts:")
print(metal_counts)
print("\nTrend analysis completed!")

# =============================================================
# STEP 6 : AUTOMATED EDA (Sweetviz)
# =============================================================
print("\n" + "=" * 60)
print("AUTOMATED EDA (SWEETVIZ)")
print("=" * 60)
if sv is not None:
    eda_report = sv.analyze(pd.read_csv(CLEANED_PATH))
    eda_report.show_html(filepath=os.path.join(REPORTS_DIR, "EDA_Report.html"), open_browser=False)
    print("Sweetviz report saved -> Reports/EDA_Report.html")
else:
    print("Sweetviz is not installed (pip install sweetviz). Existing Reports/EDA_Report.html is kept.")

# =============================================================
# STEP 7 : FEATURE SELECTION AND TRAIN-TEST SPLIT
# =============================================================
print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

df = pd.read_csv(CLEANED_PATH)

target = "Price"                                    # Dependent variable (label)
features = [                                        # Independent variables
    "Quantity.of.SKU.in.the.order",
    "Category.alias",
    "Product.gender..for.male.female.",
    "Main.color",
    "Main.metal",
    "Main.gem",
]

feature_report = pd.DataFrame({
    "Column": ["Price", "Category.alias", "Product.gender..for.male.female.", "Main.color",
               "Main.metal", "Main.gem", "Quantity.of.SKU.in.the.order", "Order.datetime",
               "Purchased.product.ID", "Category.ID"],
    "Role": ["Dependent Variable (Label)", "Independent", "Independent", "Independent",
             "Independent", "Independent", "Independent", "Not used in model",
             "Not used in model", "Not used in model"],
    "Reason": ["Value to be predicted", "Type of jewellery affects price", "Gender wise pricing",
               "Property of the item", "Metal affects price", "Gem affects price",
               "Order detail (same value for all rows)", "Used for Trend Analysis",
               "Identifier, not a property of the item", "Same information as Category.alias"],
})
print(feature_report.to_string(index=False))
feature_report.to_csv(os.path.join(REPORTS_DIR, "Feature_Selection_Report.csv"), index=False)

df = df[features + [target]].copy()
df = df.dropna(subset=[target])

X = df[features]
y = df[target]

# 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nDependent Variable   :", target)
print("Independent Variables:", features)
print("Training Records     :", len(X_train))
print("Testing Records      :", len(X_test))

# =============================================================
# STEP 8 : BUILD ML MODEL - LINEAR REGRESSION
# =============================================================
print("\n" + "=" * 60)
print("MODEL BUILDING - LINEAR REGRESSION")
print("=" * 60)

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# numerical data -> median imputation
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# categorical data -> most frequent imputation + One Hot Encoding (text -> 0/1)
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# =============================================================
# STEP 9 : MODEL EVALUATION (R2, MSE, RMSE, MAE)
# =============================================================
y_pred = model.predict(X_test)

train_r2 = model.score(X_train, y_train)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n========== LINEAR REGRESSION RESULTS ==========")
print("Features Used   :", features)
print("Training Records:", len(X_train))
print("Testing Records :", len(X_test))
print("Train R2 Score  :", round(train_r2, 4))
print("Test R2 Score   :", round(r2, 4))
print("MSE             :", round(mse, 2))
print("RMSE            :", round(rmse, 2))
print("MAE             :", round(mae, 2))

print("\nActual vs Predicted (first 10 test records):")
print(pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10].round(2)}).to_string(index=False))

# =============================================================
# STEP 10 : IMPROVE MODEL - CROSS VALIDATION, REGULARIZATION, COMPARE RESULTS
# =============================================================
print("\n" + "=" * 60)
print("IMPROVE MODEL")
print("=" * 60)

# 5-fold cross validation of Linear Regression (training data only)
lr_cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
print("Linear Regression 5-fold CV R2 :", lr_cv.round(4), "| average =", round(lr_cv.mean(), 4))

# Ridge (L2) regularization - best alpha is chosen by cross validation
ridge_search = GridSearchCV(
    Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Ridge())]),
    {"regressor__alpha": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="r2")
ridge_search.fit(X_train, y_train)
ridge_pred = ridge_search.predict(X_test)

# Lasso (L1) regularization - best alpha is chosen by cross validation
lasso_search = GridSearchCV(
    Pipeline(steps=[("preprocessor", preprocessor), ("regressor", Lasso(max_iter=10000))]),
    {"regressor__alpha": [0.01, 0.1, 1, 10]}, cv=5, scoring="r2")
lasso_search.fit(X_train, y_train)
lasso_pred = lasso_search.predict(X_test)

results = pd.DataFrame([
    ["Linear Regression", "-", lr_cv.mean(), r2_score(y_test, y_pred),
     mean_squared_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred)), mean_absolute_error(y_test, y_pred)],
    ["Ridge (L2)", ridge_search.best_params_["regressor__alpha"], ridge_search.best_score_, r2_score(y_test, ridge_pred),
     mean_squared_error(y_test, ridge_pred), np.sqrt(mean_squared_error(y_test, ridge_pred)), mean_absolute_error(y_test, ridge_pred)],
    ["Lasso (L1)", lasso_search.best_params_["regressor__alpha"], lasso_search.best_score_, r2_score(y_test, lasso_pred),
     mean_squared_error(y_test, lasso_pred), np.sqrt(mean_squared_error(y_test, lasso_pred)), mean_absolute_error(y_test, lasso_pred)],
], columns=["Model", "Best Alpha", "CV R2 (train)", "Test R2", "MSE", "RMSE", "MAE"])

print("\n========== MODEL COMPARISON ==========")
print(results.round(4).to_string(index=False))
results.round(4).to_csv(os.path.join(REPORTS_DIR, "Model_Comparison.csv"), index=False)

best_row = results.loc[results["Test R2"].idxmax()]
print("\nBest model (highest Test R2):", best_row["Model"], "| Test R2 =", round(best_row["Test R2"], 4))
best_model = {"Linear Regression": model,
              "Ridge (L2)": ridge_search.best_estimator_,
              "Lasso (L1)": lasso_search.best_estimator_}[best_row["Model"]]

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Jewellery Price")
plt.ylabel("Predicted Jewellery Price")
plt.title("Actual vs Predicted Jewellery Price")
save_and_show("Actual_vs_Predicted.png")

# =============================================================
# STEP 12 : PREDICT PRICE OF NEW JEWELLERY
# =============================================================
print("\n" + "=" * 60)
print("JEWELLERY PRICE PREDICTION")
print("=" * 60)

if sys.stdin is not None and sys.stdin.isatty():
    print("Available categories:", sorted(df["Category.alias"].unique()))
    print("Available colors    :", sorted(df["Main.color"].unique()))
    print("Available metals    :", sorted(df["Main.metal"].unique()))
    try:
        quantity = int(input("Enter Quantity: "))
        category = input("Enter Category (example jewelry.ring): ").strip()
        gender = input("Enter Gender (f / m): ").strip()
        color = input("Enter Main Color: ").strip()
        metal = input("Enter Main Metal: ").strip()
        gem = input("Enter Main Gem: ").strip()
    except ValueError:
        print("Invalid input. Quantity must be a number.")
        sys.exit(0)
else:
    print("(Non-interactive run - sample jewellery is used)")
    quantity, category, gender, color, metal, gem = 1, "jewelry.ring", "f", "red", "gold", "diamond"

new_jewellery = pd.DataFrame([{
    "Quantity.of.SKU.in.the.order": quantity,
    "Category.alias": category,
    "Product.gender..for.male.female.": gender,
    "Main.color": color,
    "Main.metal": metal,
    "Main.gem": gem,
}])
predicted_price = best_model.predict(new_jewellery)
print("\nJewellery details:", new_jewellery.iloc[0].to_dict())
print("Predicted Jewellery Price:", round(float(predicted_price[0]), 2))
print("\nNote: This is an estimated price based on historical dataset records.")
print("\nPROJECT COMPLETED SUCCESSFULLY!")
