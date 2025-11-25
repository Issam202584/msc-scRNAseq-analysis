# msc_nds_analysis.py
# ------------------------------------------------------
# Compute Neural Differentiation Score (NDS) per sample
# using the processed expression data from GSE66554
# (requires GSE_markers_long.csv created in step 1)
# ------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="talk", style="whitegrid")

# ------------------------------------------------------
# Load data
# ------------------------------------------------------
data = pd.read_csv("GSE_markers_long.csv")
print(f"Loaded {len(data)} expression entries")

# Define marker groups
neural_markers = ["TUBB3", "MAP2", "SOX2", "NES"]
msc_markers = ["CD73", "CD90", "CD105"]

# ------------------------------------------------------
# Compute NDS per sample
# ------------------------------------------------------
# Mean per marker group per sample
mean_expr = data.groupby(["GENE", "Sample"])["Expression"].mean().unstack("GENE")

# Fill missing columns (in case any marker absent)
for col in neural_markers + msc_markers:
    if col not in mean_expr.columns:
        mean_expr[col] = 0

# Calculate Neural Differentiation Score (0–1)
mean_expr["Neural_mean"] = mean_expr[neural_markers].mean(axis=1)
mean_expr["MSC_mean"] = mean_expr[msc_markers].mean(axis=1)
mean_expr["NDS"] = mean_expr["Neural_mean"] / (mean_expr["Neural_mean"] + mean_expr["MSC_mean"])

# ------------------------------------------------------
# Save results
# ------------------------------------------------------
mean_expr.to_csv("NDS_results.csv")
print("✅ Saved NDS_results.csv")

# ------------------------------------------------------
# Visualization
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x=mean_expr.index, y="NDS", data=mean_expr, palette="coolwarm")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Neural Differentiation Score (0–1)")
plt.title("Neural Differentiation Score per sample (GSE66554)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("NDS_barplot.png", dpi=300)
plt.show()

# ------------------------------------------------------
# Summary statistics
# ------------------------------------------------------
print("\n📊 Summary:")
print(mean_expr["NDS"].describe())
print("\nHigher NDS → more neural-like expression profile.")
