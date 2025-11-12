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
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import set_config

set_config(display='diagram')

cols = ["version","tos","length","id","flags","ttl","protocol","header","source","destination","options","vmid"]

def show_help():
    print("""Usage: train -[h|f]

Trains the classifing model

Available options:
-h, --help      Print this help and exit
-f, --file      Reads the input from a file
""")
    sys.exit()

def prepare_data(data):
    y = data["vmid"]
    X = data.drop(columns=["vmid", "source", "destination", "options", "header"])

    cols_to_convert = ['tos', 'length', 'id', 'ttl']
    for col in cols_to_convert:
        try:
            X[col] = X[col].astype(str).apply(lambda x: int(x, 16))
        except ValueError:
            try:
                X[col] = X[col].astype(str).apply(lambda x: int(x))
            except ValueError:
                pass

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    print(f"Numeric columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, num_cols, cat_cols

def make_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    transformers_list = [("num", num_pipe, num_cols)]

    if cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers_list.append(("cat", cat_pipe, cat_cols))

    preproc = ColumnTransformer(transformers_list, remainder="passthrough")
    return preproc

def prepare_models(preproc):
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
        },
        "MLPClassifier": { 
            "pipeline": Pipeline([
                ("preproc", preproc),
                ("clf", MLPClassifier(max_iter=500, random_state=42))
            ]),
            "param_grid": {
                "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "clf__activation": ["relu", "tanh"],
                "clf__alpha": [0.0001, 0.001],
            }
        },
    }
    return models_and_grids

def train(models_and_grids, X_train, X_test, y_train, y_test, cv):
    results = []
    best_estimators = {}

    for name, cfg in models_and_grids.items():
        print(f"\n=== Running GridSearchCV for {name} ===")
        grid = GridSearchCV(cfg["pipeline"], cfg["param_grid"], cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        try:
            test_proba = best.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, test_proba)
        except Exception:
            test_auc = np.nan
        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} CV best score: {grid.best_score_:.6f}")
        print(f"{name} Test accuracy: {test_acc:.6f}, Test ROC-AUC: {test_auc}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        
        results.append({
            "Model": name,
            "Best Params": grid.best_params_,
            "CV Score": grid.best_score_,
            "Test Accuracy": test_acc,
            "Test ROC-AUC": test_auc
        })
        best_estimators[name] = best

    return results, best_estimators

def evaluate(results, best_estimators, X_test, y_test):
    results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False).reset_index(drop=True)
    print("\nSummary of results")
    print(results_df) 

    best_idx = results_df["Test Accuracy"].idxmax()
    p0f_classifier = results_df.loc[best_idx, "Model"]
    best_model = best_estimators[p0f_classifier]
    filename = f"best_model_{p0f_classifier}.joblib"
    joblib.dump(best_model, filename)
    print(f"Saved best model: {p0f_classifier}")
    y_pred_best = best_model.predict(X_test)

    print("\nClassification report for best model:\n")
    print(classification_report(y_test, y_pred_best))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix ({p0f_classifier})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def main():
    if len(sys.argv) <= 1:
        show_help()
        sys.exit(1)

    flag = sys.argv[1]

    match flag:
        case "-f" | "--file":
            input_arg = sys.argv[2]
            print(f"Data from file: {input_arg}")
            data = pd.read_csv(input_arg, header=None, names=cols)

            print("Initial shape:", data.shape)
            print("Columns:", data.columns.tolist())

            X_train, X_test, y_train, y_test, num_cols, cat_cols = prepare_data(data)
            preproc = make_pipeline(num_cols, cat_cols)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            models_and_grids = prepare_models(preproc)
            results, best_estimators = train(models_and_grids, X_train, X_test, y_train, y_test, cv)
            evaluate(results, best_estimators, X_test, y_test)

        case "-h" | "--help":
            show_help()

        case _:
            print("Unknown flag. Try -h or --help for help.")
            sys.exit(1)


if __name__ == "__main__":
    main()
