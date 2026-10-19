
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# 1. Load dataset
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

# 2. Clean missing and non-numeric values
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric, errors="coerce")


# 3. Select input features
features = [
    "Age",
    "Number of sexual partners",
    "First sexual intercourse",
    "Smokes",
    "Smokes (years)",
    "Hormonal Contraceptives",
    "IUD",
    "STDs",
    "STDs (number)",
    "Dx:Cancer",
    "Dx:CIN",
    "Dx:HPV"
]


# 4. Select inputs and target
X = df[features].copy()
y = df["Biopsy"]

# Remove rows with missing target values
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

# Keep only valid binary target labels (0 or 1)
valid_labels = y.isin([0, 1])
X = X.loc[valid_labels].copy()
y = y.loc[valid_labels].copy()

print("Dataset size:", len(X))
print("\nTarget class counts:")
print(y.value_counts().sort_index())


# 5. Split data BEFORE fitting the imputer
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 6. Handle missing feature values
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_pos_weight = negative_count / positive_count

print("Scale positive weight:", round(scale_pos_weight, 2))

model = XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

# 8. Train model
model.fit(X_train, y_train)


# 9. Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Biopsy Negative (0)", "Biopsy Positive (1)"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )
)


# 10. Save model and imputer
joblib.dump(model, "model.pkl")
joblib.dump(imputer, "imputer.pkl")

print("\nModel saved as model.pkl")
print("Imputer saved as imputer.pkl")
