import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# Step 1: Find the cleaned dataset
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

# Step 2: Load dataset
df = pd.read_csv(dataset_path)

# Step 3: Select features and target
features = [
    "Quantity.of.SKU.in.the.order",
    "Category.alias",
    "Product.gender..for.male.female.",
    "Main.color",
    "Main.metal",
    "Main.gem"
]

target = "Price"

# Check whether required columns exist
required_columns = features + [target]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

# Step 4: Clean required columns
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df["Quantity.of.SKU.in.the.order"] = pd.to_numeric(
    df["Quantity.of.SKU.in.the.order"],
    errors="coerce"
)

df = df.dropna(subset=[target])

df = df[df[target] > 0]

# Step 5: Separate input and output
X = df[features]
y = df[target]

# Step 6: Preprocess categorical features
categorical_features = [
    "Category.alias",
    "Product.gender..for.male.female.",
    "Main.color",
    "Main.metal",
    "Main.gem"
]

numerical_features = [
    "Quantity.of.SKU.in.the.order"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# Step 7: Create and train the model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

model.fit(X, y)

print("\n===================================")
print(" JEWELLERY PRICE PREDICTION SYSTEM")
print("===================================")

# Step 8: Take input from the user
try:
    quantity = float(input("Enter quantity: "))

    category = input("Enter category alias: ").strip()
    gender = input("Enter gender (f / m): ").strip()
    color = input("Enter main color: ").strip()
    metal = input("Enter main metal: ").strip()
    gem = input("Enter main gem: ").strip()

    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")

    # Step 9: Create input data
    new_jewellery = pd.DataFrame(
        [{
            "Quantity.of.SKU.in.the.order": quantity,
            "Category.alias": category,
            "Product.gender..for.male.female.": gender,
            "Main.color": color,
            "Main.metal": metal,
            "Main.gem": gem
        }]
    )

    # Step 10: Predict price
    predicted_price = model.predict(new_jewellery)[0]

    print("\n===================================")
    print(f"Predicted Price: {predicted_price:.2f}")
    print("===================================")

    print(
        "\nNote: This is an estimated price "
        "based on historical dataset records."
    )

except ValueError as error:
    print("\nInvalid input:", error)