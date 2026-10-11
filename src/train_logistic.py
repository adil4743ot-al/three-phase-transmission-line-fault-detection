
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from load_data import load_raw
from logistic import add_intercept, fit_logistic, predict, compute_cost


# ==========================================
# 1. Project and results folders
# ==========================================

project_dir = Path(__file__).resolve().parent.parent
figures_dir = project_dir / "results" / "figures"
tables_dir = project_dir / "results" / "tables"

figures_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. Load and clean data
# ==========================================

df = load_raw()

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

X_raw = df[features].to_numpy(dtype=float)
y = df["fault"].to_numpy(dtype=int)

print("Dataset shape:", df.shape)


# ==========================================
# 3. Recreate the exact split from Task 1
# ==========================================

rng = np.random.default_rng(42)
idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]
val_idx = idx[n_train:n_train + n_val]
test_idx = idx[n_train + n_val:]

y_train = y[train_idx]
y_val = y[val_idx]

print("\nSplit sizes:")
print("Training:", len(train_idx))
print("Validation:", len(val_idx))
print("Testing:", len(test_idx))


# ==========================================
# 4. Build Set R and Set Q
# ==========================================

# Set R: six original features
X_R_raw = X_raw

# Set Q: six original features followed by their squares
X_Q_raw = np.column_stack((X_raw, X_raw ** 2))

print("\nSet R feature count:", X_R_raw.shape[1])
print("Set Q feature count:", X_Q_raw.shape[1])


# ==========================================
# 5. Standardize using training statistics only
# ==========================================

def prepare_features(X):
    X_train = X[train_idx]
    X_val = X[val_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    if np.any(std == 0):
        raise ValueError("A feature has zero training standard deviation.")

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std

    X_train_std = add_intercept(X_train_std)
    X_val_std = add_intercept(X_val_std)

    return X_train_std, X_val_std, mean, std


X_R_train, X_R_val, mean_R, std_R = prepare_features(X_R_raw)
X_Q_train, X_Q_val, mean_Q, std_Q = prepare_features(X_Q_raw)


# ==========================================
# 6. Check initial cost
# ==========================================

theta_R_zero = np.zeros(X_R_train.shape[1])
theta_Q_zero = np.zeros(X_Q_train.shape[1])

initial_cost_R = compute_cost(X_R_train, y_train, theta_R_zero)
initial_cost_Q = compute_cost(X_Q_train, y_train, theta_Q_zero)

print("\nInitial cost:")
print("Set R:", initial_cost_R)
print("Set Q:", initial_cost_Q)
print("Expected value is approximately 0.693.")


# ==========================================
# 7. Test the four learning rates
# ==========================================

alphas = [0.01, 0.1, 1, 10]
iterations = 2000

results = {}
chosen_alphas = {}


for set_name, X_train, X_val in [
    ("Set R", X_R_train, X_R_val),
    ("Set Q", X_Q_train, X_Q_val)
]:

    print("\n" + "=" * 45)
    print(set_name)
    print("=" * 45)

    plt.figure(figsize=(10, 6))

    stable_alphas = []

    for alpha in alphas:

        theta, costs = fit_logistic(
            X_train,
            y_train,
            alpha=alpha,
            iterations=iterations
        )

        # Check whether cost decreases at every iteration
        decreases_every_time = np.all(np.diff(costs) < 0)

        finite_costs = np.all(np.isfinite(costs))

        is_stable = (
            decreases_every_time
            and finite_costs
            and np.all(np.isfinite(theta))
        )

        print(
            f"Alpha={alpha}: "
            f"Initial cost={costs[0]:.6f}, "
            f"Final cost={costs[-1]:.6f}, "
            f"Decreases every iteration={is_stable}"
        )

        plt.plot(
            np.arange(1, iterations + 1),
            costs,
            label=f"alpha = {alpha}"
        )

        results[(set_name, alpha)] = (theta, costs)

        if is_stable:
            stable_alphas.append(alpha)

    plt.title(f"Cost vs Iteration - {set_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost J(theta)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = "cost_set_R.png" if set_name == "Set R" else "cost_set_Q.png"
    plt.savefig(figures_dir / filename, dpi=300)
    plt.show()

    if not stable_alphas:
        raise RuntimeError(
            f"No tested learning rate decreased cost at every iteration for {set_name}."
        )

    chosen_alphas[set_name] = max(stable_alphas)

    print("Chosen alpha:", chosen_alphas[set_name])


# ==========================================
# 8. Train final models with chosen alpha
# ==========================================

final_models = {}

for set_name, X_train, X_val in [
    ("Set R", X_R_train, X_R_val),
    ("Set Q", X_Q_train, X_Q_val)
]:

    alpha = chosen_alphas[set_name]

    theta, costs = fit_logistic(
        X_train,
        y_train,
        alpha=alpha,
        iterations=iterations
    )

    train_predictions = predict(X_train, theta)
    val_predictions = predict(X_val, theta)

    train_accuracy = np.mean(train_predictions == y_train)
    val_accuracy = np.mean(val_predictions == y_val)

    final_models[set_name] = {
        "theta": theta,
        "costs": costs,
        "alpha": alpha,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy
    }

    print("\n" + "=" * 45)
    print("Final model:", set_name)
    print("Chosen alpha:", alpha)
    print("Final cost:", costs[-1])
    print("Training accuracy:", train_accuracy)
    print("Validation accuracy:", val_accuracy)
    print("Theta:")
    print(theta)


# ==========================================
# 9. Save final model parameters
# ==========================================

np.savez(
    project_dir / "results" / "logistic_models.npz",
    theta_R=final_models["Set R"]["theta"],
    theta_Q=final_models["Set Q"]["theta"],
    alpha_R=final_models["Set R"]["alpha"],
    alpha_Q=final_models["Set Q"]["alpha"],
    mean_R=mean_R,
    std_R=std_R,
    mean_Q=mean_Q,
    std_Q=std_Q
)


# ==========================================
# 10. Print the results table
# ==========================================

print("\nFINAL LOGISTIC REGRESSION RESULTS")
print("-" * 75)
print(
    f"{'Feature Set':<15}"
    f"{'Alpha':<12}"
    f"{'Final Cost':<18}"
    f"{'Validation Accuracy':<20}"
)

for set_name in ["Set R", "Set Q"]:
    model = final_models[set_name]

    print(
        f"{set_name:<15}"
        f"{model['alpha']:<12}"
        f"{model['costs'][-1]:<18.6f}"
        f"{model['val_accuracy']:<20.4f}"
    )

print("\nTask 3 training completed.")
print("Cost plots saved in:", figures_dir)
print("Model parameters saved in results/logistic_models.npz")
