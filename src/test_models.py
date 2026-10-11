
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from load_data import load_raw
from metrics import accuracy, confusion_matrix


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)



df = load_raw()

df = df.drop(
    columns=["Unnamed: 7", "Unnamed: 8"],
    errors="ignore"
)

df = df.rename(columns={"Output (S)": "fault"})

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

X_raw = df[features].to_numpy(dtype=float)
y = df["fault"].to_numpy(dtype=int)

rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

X_train_raw = X_raw[train_idx]
X_test_raw = X_raw[test_idx]
y_test = y[test_idx]

logistic_data = np.load(
    RESULTS_DIR / "logistic_models.npz"
)

theta_R = logistic_data["theta_R"]
theta_Q = logistic_data["theta_Q"]

mean_R = logistic_data["mean_R"]
std_R = logistic_data["std_R"]

mean_Q = logistic_data["mean_Q"]
std_Q = logistic_data["std_Q"]


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def predict_logistic(X, theta):
    X = np.column_stack((np.ones(X.shape[0]), X))
    probabilities = sigmoid(X @ theta)
    return (probabilities >= 0.5).astype(int)
X_test_R = (X_test_raw - mean_R) / std_R
pred_R = predict_logistic(X_test_R, theta_R)

X_test_Q_raw = np.column_stack(
    (X_test_raw, X_test_raw ** 2)
)

X_test_Q = (X_test_Q_raw - mean_Q) / std_Q
pred_Q = predict_logistic(X_test_Q, theta_Q)

linear_package = joblib.load(
    MODELS_DIR / "best_linear_svm.joblib"
)

rbf_package = joblib.load(
    MODELS_DIR / "best_rbf_svm.joblib"
)

X_test_linear = (
    X_test_raw - linear_package["mean"]
) / linear_package["std"]

X_test_rbf = (
    X_test_raw - rbf_package["mean"]
) / rbf_package["std"]


pred_linear = linear_package["model"].predict(
    X_test_linear
)

pred_rbf = rbf_package["model"].predict(
    X_test_rbf
)



def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    acc = accuracy(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0
    )

    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "TN": tn,
        "False Alarms (FP)": fp,
        "Missed Faults (FN)": fn,
        "TP": tp
    }

models = {
    "Logistic Set R": pred_R,
    "Logistic Set Q": pred_Q,
    "Linear SVM": pred_linear,
    "RBF SVM": pred_rbf
}

results = []

for name, predictions in models.items():
    metrics_result = calculate_metrics(y_test, predictions)
    metrics_result["Model"] = name
    results.append(metrics_result)

    print("\nModel:", name)
    print("Accuracy:", round(metrics_result["Accuracy"], 6))
    print("Precision:", round(metrics_result["Precision"], 6))
    print("Recall:", round(metrics_result["Recall"], 6))
    print("F1-score:", round(metrics_result["F1-score"], 6))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("Missed faults:", metrics_result["Missed Faults (FN)"])
    print("False alarms:", metrics_result["False Alarms (FP)"])
    print("-" * 40)
results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "TN",
        "False Alarms (FP)",
        "Missed Faults (FN)",
        "TP"
    ]
]

results_df.to_csv(
    TABLES_DIR / "test_results.csv",
    index=False
)

print("\nTest results saved to results/tables/test_results.csv")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, (name, predictions) in zip(
    axes.flatten(), models.items()
):
    cm = confusion_matrix(y_test, predictions)

    ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center"
            )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "confusion_matrices.png",
    dpi=300
)

plt.show()

print("Confusion matrices saved to results/figures/")

