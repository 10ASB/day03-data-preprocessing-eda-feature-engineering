import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# LOAD DATASET
df = pd.read_csv("internB_delivery_delays.csv")

# BASIC EDA
print(df.head())
df.info()
print(df.describe(include="all"))
print("Shape:", df.shape)

print("\nMissing Values")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Clean order value before numerical imputation
df["order_value"] = (
    df["order_value"].astype(str)
    .str.replace("INR", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")

# Standardize text columns
text_cols = [
    "product_category", "customer_city", "courier_partner",
    "warehouse_zone", "payment_mode"
]
for col in text_cols:
    df[col] = df[col].astype("string").str.strip().str.title()

# Missing values: numeric median and categorical mode
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=["object", "string"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    modes = df[col].mode(dropna=True)
    if not modes.empty:
        df[col] = df[col].fillna(modes.iloc[0])

# UNIVARIATE ANALYSIS
plt.figure(figsize=(8, 6))
df["product_category"].value_counts().plot(kind="bar")
plt.title("Product Category Distribution")
plt.xlabel("Product category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["distance_km"], bins=25, kde=True)
plt.title("Distance Distribution")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(y="payment_mode", data=df)
plt.title("Payment Mode")
plt.tight_layout()
plt.show()

# BIVARIATE ANALYSIS
plt.figure(figsize=(8, 5))
sns.boxplot(x="is_delayed", y="distance_km", data=df)
plt.title("Distance vs Delay")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="warehouse_zone", hue="is_delayed", data=df)
plt.title("Warehouse Zone vs Delay")
plt.tight_layout()
plt.show()

# Correlation for numeric columns
numeric = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Numeric Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(x="is_delayed", hue="courier_partner", data=df)
plt.title("Courier Partner vs Delay")
plt.tight_layout()
plt.show()

# PREPROCESSING PIPELINE DEMO
target = "is_delayed"
X = df.drop(columns=[target])
y = df[target]

# NOTE: This demonstrates preprocessing mechanics, not a final prediction-ready
# feature set. Review post-outcome fields such as actual_delivery_days,
# customer_rating, and delay_complaint_raised for target leakage first.
num_features = X.select_dtypes(include=np.number).columns
cat_features = X.select_dtypes(include=["object", "string"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nPipeline completed successfully")
print("Train shape after preprocessing:", X_train_processed.shape)
print("Test shape after preprocessing:", X_test_processed.shape)
print("Reminder: fit preprocessing on train only; transform test with same fitted transformer.")
