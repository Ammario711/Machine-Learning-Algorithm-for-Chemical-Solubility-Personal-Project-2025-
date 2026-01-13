# chemical_solubility_regression.py - Machine Learning pipeline for chemical solubility prediction
# Author: Ammar Arshad
# Purpose: Self-directed ML project using Scikit-learn to predict solubility
# Key contributions: Full end-to-end pipeline, feature engineering prep, model tuning, evaluation & visualization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess dataset (example: ESOL solubility data - replace with your actual CSV)
# Expected columns: mol_weight, logP, etc. + 'solubility' as target
df = pd.read_csv("solubility_data.csv")  # <-- Replace with your dataset path
X = df.drop("solubility", axis=1)
y = df["solubility"]

# Handle missing values and scale features
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Model: Random Forest Regressor (robust to non-linearity)
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R²: {r2:.3f}")

# Cross-validation for robustness
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="neg_root_mean_squared_error")
print(f"CV RMSE: {-cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Feature importance visualization
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).plot(kind="bar")
plt.title("Feature Importances for Solubility Prediction")
plt.tight_layout()
plt.savefig("feature_importances.png")  # Saves plot to file
plt.show()