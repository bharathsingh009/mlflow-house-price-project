# Databricks notebook source
display(spark.sql("SHOW TABLES IN default"))

# COMMAND ----------

df = spark.table("default.housing").toPandas()

# COMMAND ----------

# ============================================
# STEP 0: Imports
# ============================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

mlflow.set_registry_uri("databricks-uc")

# ============================================
# STEP 1: Load the data
# ============================================
df = spark.table("default.housing").toPandas()
print("Total rows loaded:", len(df))
df.head()

# ============================================
# STEP 2: Clean the data
# ============================================
df = df.dropna()
df = pd.get_dummies(df, columns=["ocean_proximity"])

X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

print("Rows after cleaning:", len(df))

# ============================================
# STEP 3: Split into Old (70%) / New (10%) / Test (20%)
# ============================================
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_old, X_new, y_old, y_new = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42)

X_combined = pd.concat([X_old, X_new])
y_combined = pd.concat([y_old, y_new])

print("Old data rows (70%):", len(X_old))
print("New data rows (10%):", len(X_new))
print("Combined old+new rows (80%):", len(X_combined))
print("Test data rows (20%):", len(X_test))

# ============================================
# STEP 4: Train Model 1 (old data only, 70%)
# ============================================
with mlflow.start_run(run_name="Model_1_old_data") as run1:
    model1 = RandomForestRegressor(n_estimators=100, random_state=42)
    model1.fit(X_old, y_old)
    preds1 = model1.predict(X_test)
    error1 = mean_absolute_error(y_test, preds1)

    signature1 = infer_signature(X_old, model1.predict(X_old))

    mlflow.log_param("data_used", "70% (old only)")
    mlflow.log_metric("MAE", error1)
    mlflow.sklearn.log_model(
        model1, "model",
        signature=signature1,
        input_example=X_old.iloc[:5]
    )
    run_id_model1 = run1.info.run_id

print("Model 1 average error (MAE):", error1)

# ============================================
# STEP 5: Train Model 2 (old + new combined, 80%)
# ============================================
with mlflow.start_run(run_name="Model_2_combined_data") as run2:
    model2 = RandomForestRegressor(n_estimators=100, random_state=42)
    model2.fit(X_combined, y_combined)
    preds2 = model2.predict(X_test)
    error2 = mean_absolute_error(y_test, preds2)

    signature2 = infer_signature(X_combined, model2.predict(X_combined))

    mlflow.log_param("data_used", "80% (old + new combined)")
    mlflow.log_metric("MAE", error2)
    mlflow.sklearn.log_model(
        model2, "model",
        signature=signature2,
        input_example=X_combined.iloc[:5]
    )
    run_id_model2 = run2.info.run_id

print("Model 2 average error (MAE):", error2)

# ============================================
# STEP 6: Compare, register the better model, promote using ALIAS
# ============================================
client = MlflowClient()
model_name = "workspace.default.house_price_model"   # <-- change "workspace.default" if your catalog/schema differ

print("\n--- COMPARISON ---")
print("Model 1 error:", error1)
print("Model 2 error:", error2)

if error2 < error1:
    print("Model 2 is BETTER -> registering and promoting Model 2")
    best_run_id = run_id_model2
    best_model_label = "Model 2 (old+new combined)"
else:
    print("Model 1 is BETTER -> registering and promoting Model 1")
    best_run_id = run_id_model1
    best_model_label = "Model 1 (old only)"

result = mlflow.register_model(f"runs:/{best_run_id}/model", model_name)

client.set_registered_model_alias(
    name=model_name,
    alias="Production",
    version=result.version
)

print(f"\n✅ {model_name} version {result.version} ({best_model_label}) is now aliased as Production")

# ============================================
# STEP 7: Test a live prediction
# ============================================
sample_house = X_test.iloc[[0]]
real_price = y_test.iloc[0]

loaded_model = mlflow.sklearn.load_model(f"models:/{model_name}@Production")
predicted_price = loaded_model.predict(sample_house)[0]

print("\n--- LIVE PREDICTION TEST ---")
print("Sample house features:\n", sample_house)
print("Predicted price:", predicted_price)
print("Actual real price:", real_price)
print("Error on this house:", abs(predicted_price - real_price))

# COMMAND ----------

# ============================================
# STEP 2.5: EDA (Exploratory Data Analysis) Graphs
# ============================================
import matplotlib.pyplot as plt
import seaborn as sns

# Graph 1: Distribution of house prices
plt.figure(figsize=(8,5))
sns.histplot(df["median_house_value"], bins=50, kde=True)
plt.title("Distribution of Median House Values")
plt.xlabel("Median House Value")
plt.ylabel("Count")
plt.show()

# Graph 2: Median Income vs House Price (correlation)
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["median_income"], y=df["median_house_value"], alpha=0.3)
plt.title("Median Income vs Median House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()

# Graph 3: Correlation heatmap (which features matter most)
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=["float64", "int64"])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Graph 4: Average house price by ocean proximity (before one-hot encoding)
# NOTE: run this BEFORE Step 2's pd.get_dummies() line, using the original df