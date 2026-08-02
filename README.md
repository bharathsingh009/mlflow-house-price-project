# MLflow House Price Prediction Pipeline on Databricks

An end-to-end Machine Learning pipeline built for **Databricks**, leveraging **MLflow**, **Unity Catalog**, and **Scikit-Learn** to train, compare, register, and deploy house price regression models.

---

## 🚀 Project Overview

This project demonstrates a production-grade machine learning lifecycle workflow within a Databricks environment:
1. **Data Ingestion & Cleaning:** Loads raw housing data from Unity Catalog, handles missing values, and performs one-hot encoding on categorical features (`ocean_proximity`).
2. **Data Splitting Strategy:** Splits data into training sets (Old data vs. Combined Old + New data) and a held-out Test set to evaluate performance robustness.
3. **Model Tracking & Experimentation:** Trains multiple `RandomForestRegressor` iterations, tracking parameters, metrics (MAE), and input signatures automatically using **MLflow**.
4. **Automated Model Registry & Aliasing:** Compares model performance dynamically, registers the superior model to **Unity Catalog**, and promotes it using MLflow Aliases (`@Production`).
5. **Inference Verification:** Loads the live Production model from Unity Catalog to execute single-row validation checks.

---

## 📊 Pipeline Architecture & Steps

* **Step 0 & 1:** Connects to the Databricks Spark session, inspects available tables, and loads `default.housing` into a Pandas DataFrame.
* **Step 2 (Data Preprocessing):** Drops `NaN` values and transforms categorical properties into numerical indicator columns.
* **Step 3 (Splitting):** 
  * **Test Set:** 20%
  * **Old Data (Model 1 training):** 70% 
  * **New Data Increment:** 10% (Combined with old for Model 2 = 80%)
* **Step 4 & 5 (Training & Logging):** 
  * *Model 1:* Trained solely on historical/old data.
  * *Model 2:* Trained on combined historical + newly added data.
* **Step 6 (Evaluation & Promotion):** Evaluates both runs using **Mean Absolute Error (MAE)** against the test set. Automatically registers the best-performing model to `workspace.default.house_price_model` and sets the `@Production` alias via `MlflowClient`.
* **Step 7 (Inference Test):** Validates the production endpoint by running a live prediction on an unseen sample house.

---

## 📈 Exploratory Data Analysis (EDA)

The notebook also includes visualization blocks using `matplotlib` and `seaborn` covering:
1. **Distribution of Median House Values** (Target variable spread)
2. **Median Income vs. House Price** (Core feature correlation scatter plot)
3. **Correlation Heatmap** (Numerical feature collinearity check)
4. **Ocean Proximity Breakdown** (Geographic pricing trends)

---

## 🛠️ Tech Stack & Requirements

* **Cloud Platform:** Databricks (Unity Catalog Enabled)
* **Orchestration / Tracking:** MLflow (`mlflow.sklearn`, `MlflowClient`)
* **Core Libraries:** 
  * `pandas`, `scikit-learn`
  * `matplotlib`, `seaborn`
* **Model Artifacts:** Random Forest Regressor

---

## ⚙️ How to Run

1. Import the notebook into your **Databricks Workspace**.
2. Attach the notebook to a cluster with MLflow and standard Python libraries installed.
3. Ensure the source table (`default.housing`) exists in your Unity Catalog / Hive Metastore.
4. Run all cells sequentially to execute training, model evaluation, and deployment promotion.
