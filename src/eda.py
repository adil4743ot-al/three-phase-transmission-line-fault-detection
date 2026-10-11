import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_raw


# ==========================================
# 1. Load and prepare raw data
# ==========================================

df = load_raw()

# Remove empty columns
df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])

# Rename label
df = df.rename(columns={"Output (S)": "fault"})


# ==========================================
# 2. Create results folder
# ==========================================

os.makedirs("../results/figures", exist_ok=True)


# ==========================================
# 3. Plot 1
# Ia, Ib, Ic vs row number
# First 2000 rows
# Fault regions shaded
# ==========================================

data = df.iloc[:2000]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(data.index, data["Ia"], label="Ia")
ax.plot(data.index, data["Ib"], label="Ib")
ax.plot(data.index, data["Ic"], label="Ic")

# Find continuous fault regions
fault = data["fault"].to_numpy()

start = None

for i in range(len(fault)):

    if fault[i] == 1 and start is None:
        start = i

    if (fault[i] == 0 or i == len(fault) - 1) and start is not None:

        end = i if fault[i] == 0 else i + 1

        ax.axvspan(
            data.index[start],
            data.index[end - 1],
            alpha=0.2
        )

        start = None


ax.set_title("Ia, Ib, Ic for First 2000 Rows")
ax.set_xlabel("Row Number")
ax.set_ylabel("Current")
ax.legend()
ax.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/plot1_currents.png", dpi=300)

plt.show()


# ==========================================
# 4. Plot 2
# Ia vs Ib
# ==========================================

plt.figure(figsize=(8, 6))

no_fault = df["fault"] == 0
fault = df["fault"] == 1

plt.scatter(
    df.loc[no_fault, "Ia"],
    df.loc[no_fault, "Ib"],
    alpha=0.5,
    label="No Fault"
)

plt.scatter(
    df.loc[fault, "Ia"],
    df.loc[fault, "Ib"],
    alpha=0.5,
    label="Fault"
)

plt.title("Ia vs Ib")
plt.xlabel("Ia")
plt.ylabel("Ib")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/plot2_ia_vs_ib.png", dpi=300)

plt.show()


# ==========================================
# 5. Plot 3
# Histogram of Ia
# ==========================================

plt.figure(figsize=(8, 6))

plt.hist(
    df.loc[no_fault, "Ia"],
    bins=50,
    alpha=0.6,
    label="No Fault"
)

plt.hist(
    df.loc[fault, "Ia"],
    bins=50,
    alpha=0.6,
    label="Fault"
)

plt.title("Distribution of Ia")
plt.xlabel("Ia")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("../results/figures/plot3_ia_histogram.png", dpi=300)

plt.show()


print("\nTask 2 completed successfully.")

print("\nFigures saved in:")
print("../results/figures/")
