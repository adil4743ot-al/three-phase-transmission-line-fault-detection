import numpy as np
import pandas as pd

from load_data import load_raw


# =========================
# 1. Load the dataset
# =========================

df = load_raw()

print("Raw shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# =========================
# 2. Remove empty columns
# =========================

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])


# =========================
# 3. Rename label column
# =========================

df = df.rename(columns={"Output (S)": "fault"})


# =========================
# 4. Define features
# =========================

features = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]


# =========================
# 5. Check numeric features
# =========================

print("\nNumeric check:")

for col in features:
    print(col, pd.api.types.is_numeric_dtype(df[col]))


# =========================
# 6. Check missing values
# =========================

print("\nMissing values:")
print(df.isnull().sum())


# =========================
# 7. Final columns
# =========================

print("\nFinal columns:")
print(df.columns.tolist())


# =========================
# 8. Count fault classes
# =========================

print("\nFault counts:")
print(df["fault"].value_counts())


# =========================
# 9. Train / Validation / Test split
# =========================

rng = np.random.default_rng(42)

idx = rng.permutation(len(df))

n_train = int(0.6 * len(df))
n_val = int(0.2 * len(df))

train_idx = idx[:n_train]

val_idx = idx[n_train:n_train + n_val]

test_idx = idx[n_train + n_val:]


# =========================
# 10. Split sizes
# =========================

print("\nSplit sizes:")

print("Training:", len(train_idx))
print("Validation:", len(val_idx))
print("Testing:", len(test_idx))


# =========================
# 11. Fault fraction
# =========================

print("\nFault fraction in each split:")

train_fault_fraction = df.iloc[train_idx]["fault"].mean()
val_fault_fraction = df.iloc[val_idx]["fault"].mean()
test_fault_fraction = df.iloc[test_idx]["fault"].mean()

print("Training:", train_fault_fraction)
print("Validation:", val_fault_fraction)
print("Testing:", test_fault_fraction)


# =========================
# 12. Prepare X and y
# =========================

X = df[features].to_numpy(dtype=float)
y = df["fault"].to_numpy()


X_train = X[train_idx]
X_val = X[val_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]


# =========================
# 13. Calculate training statistics
# =========================

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)


# =========================
# 14. Standardize data
# =========================

X_train_std = (X_train - train_mean) / train_std

X_val_std = (X_val - train_mean) / train_std

X_test_std = (X_test - train_mean) / train_std


# =========================
# 15. Display standardization results
# =========================

print("\nTraining mean:")
print(train_mean)

print("\nTraining standard deviation:")
print(train_std)

print("\nFirst 5 standardized training rows:")
print(X_train_std[:5])
