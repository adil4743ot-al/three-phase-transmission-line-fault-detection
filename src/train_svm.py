
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.svm import SVC
from load_data import load_raw


project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"
models_dir = results_dir / "models"
tables_dir = results_dir / "tables"

models_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


df = load_raw()
df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

X = df[features].to_numpy(dtype=float)
y = df["fault"].to_numpy(dtype=int)
rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_train_raw = X[train_idx]
X_val_raw = X[val_idx]
y_train = y[train_idx]
y_val = y[val_idx]
mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0)

if np.any(std == 0):
    raise ValueError("A feature has zero standard deviation.")

X_train = (X_train_raw - mean) / std
X_val = (X_val_raw - mean) / std

C_values = [0.1, 1, 10, 100]
gamma_values = [0.1, 1]

results = []
best_linear = None
best_rbf = None
print("\n========== LINEAR SVM ==========")

for C in C_values:
    model = SVC(kernel="linear", C=C)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy(y_train, train_pred)
    val_acc = accuracy(y_val, val_pred)
    support_count = len(model.support_)

    print(
        f"C={C}, Train Accuracy={train_acc:.4f}, "
        f"Validation Accuracy={val_acc:.4f}, "
        f"Support Vectors={support_count}"
    )

    results.append({
        "Kernel": "Linear",
        "C": C,
        "Gamma": "-",
        "Train Accuracy": train_acc,
        "Validation Accuracy": val_acc,
        "Support Vectors": support_count
    })

    if best_linear is None or val_acc > best_linear["accuracy"]:
        best_linear = {
            "model": model,
            "C": C,
            "accuracy": val_acc
        }
print("\n========== RBF SVM ==========")

for C in C_values:
    for gamma in gamma_values:
        model = SVC(kernel="rbf", C=C, gamma=gamma)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_acc = accuracy(y_train, train_pred)
        val_acc = accuracy(y_val, val_pred)
        support_count = len(model.support_)

        print(
            f"C={C}, Gamma={gamma}, "
            f"Train Accuracy={train_acc:.4f}, "
            f"Validation Accuracy={val_acc:.4f}, "
            f"Support Vectors={support_count}"
        )

        results.append({
            "Kernel": "RBF",
            "C": C,
            "Gamma": gamma,
            "Train Accuracy": train_acc,
            "Validation Accuracy": val_acc,
            "Support Vectors": support_count
        })

        if best_rbf is None or val_acc > best_rbf["accuracy"]:
            best_rbf = {
                "model": model,
                "C": C,
                "gamma": gamma,
                "accuracy": val_acc
            }

results_df = pd.DataFrame(results)
print("\n========== ALL RESULTS ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    tables_dir / "svm_results.csv",
    index=False
)
joblib.dump({
    "model": best_linear["model"],
    "mean": mean,
    "std": std,
    "C": best_linear["C"]
}, models_dir / "best_linear_svm.joblib")

joblib.dump({
    "model": best_rbf["model"],
    "mean": mean,
    "std": std,
    "C": best_rbf["C"],
    "gamma": best_rbf["gamma"]
}, models_dir / "best_rbf_svm.joblib")

print("\nBest Linear C:", best_linear["C"])
print("Best Linear validation accuracy:", best_linear["accuracy"])

print("\nBest RBF C:", best_rbf["C"])
print("Best RBF gamma:", best_rbf["gamma"])
print("Best RBF validation accuracy:", best_rbf["accuracy"])

print("\nSVM training completed.")
