"""
p0f OS Fingerprint Classification
---------------------------------

(Decision Tree, Random Forest, Logistic Regression, KNN, Naive Bayes).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import set_config

set_config(display='diagram')

# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv("p0f_signatures.csv") 
print("Initial shape:", data.shape)
print("Columns:", data.columns.tolist())

# Target and features
y = data["os_name"]
X = data.drop(columns=["os_name"])

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print(f"Numeric columns: {num_cols}")
print(f"Categorical columns: {cat_cols}")

# -----------------------------
# 2. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. Preprocessing pipelines
# -----------------------------
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), // fills missing
    ("scaler", StandardScaler()) // scales to center around 0
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preproc = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# -----------------------------
# 4. Define models and parameter grids
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_and_grids = {
    "DecisionTree": {
        "pipeline": Pipeline([
            ("preproc", preproc),
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        "param_grid": {
            "clf__max_depth": [5, 10, 20, None],
            "clf__min_samples_split": [2, 5, 10],
        }
    },
    "RandomForest": {
        "pipeline": Pipeline([
            ("preproc", preproc),
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),
        "param_grid": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [10, None],
            "clf__max_features": ["sqrt", "log2"]
        }
    },
    "KNN": {
        "pipeline": Pipeline([
            ("preproc", preproc),
            ("clf", KNeighborsClassifier())
        ]),
        "param_grid": {
            "clf__n_neighbors": [3, 5, 7],
            "clf__weights": ["uniform", "distance"]
        }
    },
    "LogisticRegression": {
        "pipeline": Pipeline([
            ("preproc", preproc),
            ("clf", LogisticRegression(max_iter=2000, multi_class="auto"))
        ]),
        "param_grid": {
            "clf__C": [0.1, 1, 10]
        }
    },
    "NaiveBayes": {
        "pipeline": Pipeline([
            ("preproc", preproc),
            ("clf", GaussianNB())
        ]),
        "param_grid": {}
    }
}

# -----------------------------
# 5. Train and evaluate
# -----------------------------
results = []
best_estimators = {}

for name, cfg in models_and_grids.items():
    print(f"\n=== Training {name} ===")
    grid = GridSearchCV(cfg["pipeline"], cfg["param_grid"], cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} best params: {grid.best_params_}")
    print(f"{name} test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    results.append({"Model": name, "Accuracy": acc})
    best_estimators[name] = grid.best_estimator_

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\nSummary of results:")
print(results_df)

# -----------------------------
# 7. Save best model
# -----------------------------
joblib.dump(best_model, f"best_os_model_{best_name}.joblib")
print(f"Saved model as best_os_model_{best_name}.joblib")