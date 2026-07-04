# =========================================================
# IMPORT LIBRARY
# =========================================================
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv('AgroFertilizerLoss.csv')

# =========================================================
# TARGET REGRESI
# =========================================================
# Karena regresi, target menggunakan nilai asli
# Fertilizer_Loss_Percentage

y = df['Fertilizer_Loss_Percentage']

# =========================================================
# HAPUS KOLOM DATA LEAKAGE
# =========================================================
columns_to_drop = [
    'Total_Fertilizer_Loss_kg_ha',
    'Nitrogen_Loss',
    'Phosphorus_Loss',
    'Potassium_Loss'
]

X = df.drop(columns=columns_to_drop + ['Fertilizer_Loss_Percentage'])

# =========================================================
# HAPUS DATA KOSONG
# =========================================================
X = X.dropna()
y = y.loc[X.index]

# =========================================================
# ENCODING DATA KATEGORIK
# =========================================================
label_encoders = {}

for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# =========================================================
# SPLIT DATA
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# TRAIN MODEL
# =========================================================
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# =========================================================
# EVALUASI MODEL
# =========================================================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# =========================================================
# SIMPAN MODEL
# =========================================================
model_data = {
    'model': model,
    'features': X.columns.tolist(),
    'encoders': label_encoders
}

joblib.dump(model_data, 'xgboost_regression_model.pkl')
print("\nModel berhasil disimpan!")