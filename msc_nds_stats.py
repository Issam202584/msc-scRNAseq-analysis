# msc_nds_stats.py
# ------------------------------------------------------
# Statistical analysis & violin plot of Neural Differentiation Scores (NDS)
# ------------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

sns.set(context="talk", style="whitegrid")

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
df = pd.read_csv("NDS_results.csv")
print(f"Loaded {df.shape[0]} samples from NDS_results.csv")

# Extract only needed columns
if "NDS" not in df.columns:
    raise ValueError("NDS column not found in file. Please run msc_nds_analysis.py first.")

# Optional: assign experimental groups if available
# Here, we’ll just simulate two groups (early vs late differentiation)
df["Group"] = ["Early" if i < len(df)/2 else "Late" for i in range(len(df))]

# ------------------------------------------------------
# Basic statistics
# ------------------------------------------------------
group_stats = df.groupby("Group")["NDS"].describe()
print("\n📊 Group summary:")
print(group_stats)

# t-test
t_stat, p_val = stats.ttest_ind(
    df[df["Group"] == "Early"]["NDS"],
    df[df["Group"] == "Late"]["NDS"],
    equal_var=False
)
print(f"\n🧮 Independent t-test (Early vs Late): t = {t_stat:.3f}, p = {p_val:.4f}")

# ------------------------------------------------------
# Visualization (violin + swarm plot)
# ------------------------------------------------------
plt.figure(figsize=(8, 6))
sns.violinplot(x="Group", y="NDS", data=df, inner="box", palette="viridis", cut=0)
sns.swarmplot(x="Group", y="NDS", data=df, color="black", size=6, alpha=0.7)

plt.title("Neural Differentiation Score by Group")
plt.ylabel("Neural Differentiation Score (0–1)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("NDS_violin_plot.png", dpi=300)
plt.show()

# ------------------------------------------------------
# Save summary
# ------------------------------------------------------
group_stats.to_csv("NDS_group_stats.csv")
print("\n✅ Saved NDS_group_stats.csv and NDS_violin_plot.png")
