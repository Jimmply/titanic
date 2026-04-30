"""
Titanic Dataset - Visualizations
Step 2: Generate and save 6 exploratory plots to graphs/
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import seaborn as sns

# ── Setup ──────────────────────────────────────────────────────────────────────
GRAPHS_DIR = "/Users/jimmy/Documents/Claude/titanic_project/graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (9, 6)

# Load data
df = sns.load_dataset("titanic")

# Map numeric survived to readable labels for select plots
df["survived_label"] = df["survived"].map({0: "Did Not Survive", 1: "Survived"})

saved = []

# ── 1. Survival count bar chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
counts = df["survived"].value_counts().rename({0: "Did Not Survive", 1: "Survived"})
colors = ["#d9534f", "#5cb85c"]
bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_title("Titanic — Survival Count", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Outcome", fontsize=12)
ax.set_ylabel("Number of Passengers", fontsize=12)
ax.set_ylim(0, counts.values.max() * 1.15)
plt.tight_layout()
path = os.path.join(GRAPHS_DIR, "01_survival_count.png")
fig.savefig(path, dpi=150)
plt.close(fig)
saved.append(path)
print(f"Saved: {path}")

# ── 2. Survival rate by sex ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
sex_survival = (
    df.groupby("sex")["survived"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
)
sex_survival["outcome"] = sex_survival["survived"].map({0: "Did Not Survive", 1: "Survived"})
sns.barplot(
    data=sex_survival, x="sex", y="pct", hue="outcome",
    palette={"Did Not Survive": "#d9534f", "Survived": "#5cb85c"},
    edgecolor="white", ax=ax
)
ax.set_title("Survival Rate by Sex", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Sex", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(["Female", "Male"])
ax.legend(title="Outcome", fontsize=10)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2, p.get_height() + 0.8),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
path = os.path.join(GRAPHS_DIR, "02_survival_by_sex.png")
fig.savefig(path, dpi=150)
plt.close(fig)
saved.append(path)
print(f"Saved: {path}")

# ── 3. Survival rate by passenger class ───────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
class_survival = (
    df.groupby("pclass")["survived"]
    .value_counts(normalize=True)
    .mul(100)
    .rename("pct")
    .reset_index()
)
class_survival["outcome"] = class_survival["survived"].map({0: "Did Not Survive", 1: "Survived"})
sns.barplot(
    data=class_survival, x="pclass", y="pct", hue="outcome",
    palette={"Did Not Survive": "#d9534f", "Survived": "#5cb85c"},
    edgecolor="white", ax=ax
)
ax.set_title("Survival Rate by Passenger Class", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Passenger Class", fontsize=12)
ax.set_ylabel("Percentage (%)", fontsize=12)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
ax.legend(title="Outcome", fontsize=10)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2, p.get_height() + 0.8),
                    ha="center", va="bottom", fontsize=9)
plt.tight_layout()
path = os.path.join(GRAPHS_DIR, "03_survival_by_class.png")
fig.savefig(path, dpi=150)
plt.close(fig)
saved.append(path)
print(f"Saved: {path}")

# ── 4. Age distribution by survival status (overlapping histogram) ─────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
survived_ages    = df.loc[df["survived"] == 1, "age"].dropna()
not_survived_ages = df.loc[df["survived"] == 0, "age"].dropna()
ax.hist(not_survived_ages, bins=30, alpha=0.55, color="#d9534f",
        label="Did Not Survive", edgecolor="white", linewidth=0.6)
ax.hist(survived_ages, bins=30, alpha=0.55, color="#5cb85c",
        label="Survived", edgecolor="white", linewidth=0.6)
ax.axvline(survived_ages.median(), color="#2e7d32", linestyle="--",
           linewidth=1.5, label=f"Survived median ({survived_ages.median():.1f})")
ax.axvline(not_survived_ages.median(), color="#b71c1c", linestyle="--",
           linewidth=1.5, label=f"Not Survived median ({not_survived_ages.median():.1f})")
ax.set_title("Age Distribution by Survival Status", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Age", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
path = os.path.join(GRAPHS_DIR, "04_age_distribution.png")
fig.savefig(path, dpi=150)
plt.close(fig)
saved.append(path)
print(f"Saved: {path}")

# ── 5. Fare distribution by class (boxplot) ────────────────────────────────────
fig, ax = plt.subplots(figsize=FIGSIZE)
class_labels = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
df["class_label"] = df["pclass"].map(class_labels)
order = ["1st Class", "2nd Class", "3rd Class"]
sns.boxplot(
    data=df, x="class_label", y="fare", hue="class_label", order=order,
    palette={"1st Class": "#5b9bd5", "2nd Class": "#70ad47", "3rd Class": "#ed7d31"},
    flierprops=dict(marker="o", markerfacecolor="grey", markersize=4, alpha=0.5),
    legend=False, ax=ax
)
ax.set_title("Fare Distribution by Passenger Class", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Passenger Class", fontsize=12)
ax.set_ylabel("Fare (£)", fontsize=12)
# Annotate median fares
for i, cls in enumerate(order):
    median_val = df.loc[df["class_label"] == cls, "fare"].median()
    ax.text(i, median_val + 2, f"£{median_val:.1f}", ha="center",
            va="bottom", fontsize=9, fontweight="bold", color="black")
plt.tight_layout()
path = os.path.join(GRAPHS_DIR, "05_fare_by_class.png")
fig.savefig(path, dpi=150)
plt.close(fig)
saved.append(path)
print(f"Saved: {path}")

# ── 6. Correlation heatmap of numeric features ────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle mask
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, vmin=-1, vmax=1,
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 11, "weight": "bold"},
    ax=ax
)
ax.set_title("Correlation Heatmap — Numeric Features", fontsize=15, fontweight="bold", pad=14)
plt.tight_layout()
path = os.path.join(GRAPHS_DIR, "06_correlation_heatmap.png")
fig.savefig(path, dpi=150)
plt.close(fig)
saved.append(path)
print(f"Saved: {path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"All {len(saved)} plots saved to: {GRAPHS_DIR}")
print("=" * 60)
for p in saved:
    size_kb = os.path.getsize(p) / 1024
    print(f"  {os.path.basename(p):40s}  {size_kb:6.1f} KB")
