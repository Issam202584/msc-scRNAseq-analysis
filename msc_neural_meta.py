# msc_neural_meta.py
# Meta-analysis of MSC → Neural differentiation across multiple GEO datasets.
# Outputs:
#   - combined_markers_long.csv
#   - combined_NDS.csv
#   - plot_markers_by_dataset.png
#   - plot_nds_violin_by_dataset.png
#   - stats_summary.txt

import os, re, sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import GEOparse

sns.set(context="talk", style="whitegrid")

# -----------------------------
# CONFIG — edit the list below to add/remove datasets
# -----------------------------
GSE_LIST = [
    "GSE66554",  # Rat MSC → neural (Agilent)
    # "GSE16592",  # Human MSC neural induction (Affymetrix)  ← uncomment to include
    # "GSE118882", # Rat MSC neural differentiation (RNA-seq)  ← uncomment to include (may require special handling)
    # "GSE76381",  # Human MSC → neuron (RNA-seq)             ← uncomment to include (may require special handling)
    # "GSE104410", # Human MSC neurogenic vs control (Affy)    ← uncomment to include
]

# Canonical markers & synonyms (UPPERCASE for matching)
SYNS = {
    "TUBB3": ["TUBB3", "TUBB-3", "TUBBIII", "BETA-III-TUBULIN", "BETA III TUBULIN"],
    "MAP2" : ["MAP2", "MICROTUBULE-ASSOCIATED PROTEIN 2"],
    "SOX2" : ["SOX2"],
    "NES"  : ["NES", "NESTIN"],
    "CD73" : ["CD73", "NT5E"],
    "CD90" : ["CD90", "THY1"],
    "CD105": ["CD105", "ENG"],
}
NEURAL = ["TUBB3", "MAP2", "SOX2", "NES"]
MSC    = ["CD73", "CD90", "CD105"]

# -----------------------------
# Helpers
# -----------------------------
def norm_symbol(x: str | None) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)): return None
    s = str(x).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s if s else None

def pick_symbol_col(df: pd.DataFrame) -> str:
    prefs = ["Gene Symbol","GENE_SYMBOL","SYMBOL","gene_symbol","Gene symbol","Symbol",
             "GeneSymbol","GENE_SYMBOLS","gene_assignment"]
    for c in prefs:
        if c in df.columns: return c
    for c in df.columns:
        if "symbol" in c.lower(): return c
    if "gene_assignment" in df.columns: return "gene_assignment"
    raise RuntimeError("No gene symbol column found in GPL.")

def extract_symbol_from_assignment(cell: str | None) -> str | None:
    # Many platforms: "ACC // SYMBOL // description // ..."
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return None
    parts = str(cell).split("//")
    if len(parts) > 1:
        symbol = parts[1].strip().split()[0]
        return symbol.upper()
    # fallback: first token
    first = str(cell).split("//")[0].strip()
    return first.upper() if first else None

def load_gse_with_platform(gse_id: str):
    # Prefer FAMILY SOFT (contains GPL); if only TXT exists, reload from family SOFT.
    fam = os.path.join(".", f"{gse_id}_family.soft.gz")
    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir="./")  # family.soft.gz or txt
    except Exception as e:
        # Try direct family soft path if already downloaded
        if os.path.exists(fam):
            gse = GEOparse.get_GEO(filepath=fam)
        else:
            raise e
    # ensure GPL present; if missing and fam exists, reload
    if not gse.gpls and os.path.exists(fam):
        gse = GEOparse.get_GEO(filepath=fam)
    if not gse.gpls:
        raise RuntimeError(f"{gse_id}: platform (GPL) not available. Delete {gse_id}.txt and retry.")
    return gse

def gse_to_markers_long(gse_id: str) -> pd.DataFrame:
    """Return long table: columns = [Dataset, Sample, SYMBOL, GENE, Expression]"""
    print(f"\n📥 Processing {gse_id} …")
    gse = load_gse_with_platform(gse_id)
    gpl = list(gse.gpls.values())[0]
    anno = gpl.table.copy()
    anno.columns = anno.columns.str.strip()
    sym_col = pick_symbol_col(anno)
    id_col = "ID" if "ID" in anno.columns else anno.columns[0]
    anno = anno[[id_col, sym_col]].rename(columns={id_col:"ID_REF", sym_col:"SYMBOL_RAW"})
    # normalize symbols
    if sym_col.lower() == "gene_assignment":
        anno["SYMBOL"] = anno["SYMBOL_RAW"].map(extract_symbol_from_assignment)
    else:
        anno["SYMBOL"] = anno["SYMBOL_RAW"].map(norm_symbol)
    anno = anno.dropna(subset=["SYMBOL"]).drop_duplicates(subset=["ID_REF"])

    # Build expression matrix (probes × samples)
    expr = gse.pivot_samples("VALUE")
    if expr.index.name != "ID_REF":
        expr.index.name = "ID_REF"
    expr = expr.reset_index()

    merged = expr.merge(anno[["ID_REF","SYMBOL"]], on="ID_REF", how="left").dropna(subset=["SYMBOL"])
    value_cols = [c for c in merged.columns if c not in ("ID_REF","SYMBOL")]
    if not value_cols:
        raise RuntimeError(f"{gse_id}: no numeric sample columns found.")
    # aggregate probes → gene mean per sample
    gene_mean = (merged
                 .groupby("SYMBOL", as_index=False)[value_cols]
                 .mean())
    gene_mean["SYMBOL"] = gene_mean["SYMBOL"].map(norm_symbol)

    # Keep canonical markers (with synonyms)
    rows = []
    for canon, syns in SYNS.items():
        syns_up = [s.upper() for s in syns]
        sub = gene_mean[gene_mean["SYMBOL"].isin(syns_up)].copy()
        if sub.empty:
            continue
        sub["GENE"] = canon
        # melt to long samples
        long = sub.melt(id_vars=["SYMBOL","GENE"], var_name="Sample", value_name="Expression")
        long = long[long["Sample"].astype(str).str.startswith("GSM")]
        long["Dataset"] = gse_id
        rows.append(long)
    if not rows:
        print(f"⚠️ {gse_id}: no marker symbols matched; skipping.")
        return pd.DataFrame(columns=["Dataset","Sample","SYMBOL","GENE","Expression"])
    out = pd.concat(rows, ignore_index=True)
    return out

def compute_nds(markers_long: pd.DataFrame) -> pd.DataFrame:
    """Return per-sample NDS table with columns: Dataset, Sample, Neural_mean, MSC_mean, NDS"""
    if markers_long.empty:
        return pd.DataFrame(columns=["Dataset","Sample","Neural_mean","MSC_mean","NDS"])
    # mean per sample per canonical gene
    wide = markers_long.groupby(["Dataset","Sample","GENE"])["Expression"].mean().unstack("GENE")
    # ensure columns exist
    for g in NEURAL + MSC:
        if g not in wide.columns:
            wide[g] = np.nan
    # compute NDS
    wide["Neural_mean"] = wide[NEURAL].mean(axis=1, skipna=True)
    wide["MSC_mean"]    = wide[MSC].mean(axis=1, skipna=True)
    wide["NDS"] = wide["Neural_mean"] / (wide["Neural_mean"] + wide["MSC_mean"])
    wide = wide.reset_index()
    return wide

# -----------------------------
# Main workflow
# -----------------------------
all_markers = []
for gse_id in GSE_LIST:
    try:
        df = gse_to_markers_long(gse_id)
    except Exception as e:
        print(f"❌ Skipping {gse_id}: {e}")
        continue
    if not df.empty:
        all_markers.append(df)

if not all_markers:
    raise SystemExit("No datasets produced markers. Aborting.")

markers_long = pd.concat(all_markers, ignore_index=True)
markers_long.to_csv("combined_markers_long.csv", index=False)
print(f"\n✅ Saved combined_markers_long.csv  (rows: {len(markers_long)})")

nds = compute_nds(markers_long)
nds.to_csv("combined_NDS.csv", index=False)
print(f"✅ Saved combined_NDS.csv  (samples: {nds.shape[0]})")

# -----------------------------
# Plots
# -----------------------------
# 1) Marker distributions per dataset
plt.figure(figsize=(11,6))
sns.boxplot(data=markers_long, x="GENE", y="Expression", hue="Dataset")
sns.stripplot(data=markers_long, x="GENE", y="Expression", hue=None, color="k", alpha=0.35)
plt.title("Marker expression across datasets")
plt.xlabel("Canonical marker")
plt.ylabel("Expression (array units)")
plt.tight_layout()
plt.savefig("plot_markers_by_dataset.png", dpi=300)
plt.close()
print("📊 Saved plot_markers_by_dataset.png")

# 2) NDS per dataset
plt.figure(figsize=(8,6))
sns.violinplot(data=nds, x="Dataset", y="NDS", inner="box", cut=0)
sns.swarmplot(data=nds, x="Dataset", y="NDS", color="k", alpha=0.7)
plt.title("Neural Differentiation Score by dataset")
plt.ylim(0,1)
plt.tight_layout()
plt.savefig("plot_nds_violin_by_dataset.png", dpi=300)
plt.close()
print("📊 Saved plot_nds_violin_by_dataset.png")

# -----------------------------
# Basic statistics across datasets
# -----------------------------
summary_lines = []
summary_lines.append(f"Datasets included: {', '.join(nds['Dataset'].unique())}")
summary_lines.append(f"Total samples (rows in combined_NDS.csv): {nds.shape[0]}")
summary_lines.append("\nGroup means (NDS) by dataset:")
means = nds.groupby("Dataset")["NDS"].mean().sort_values()
summary_lines.append(str(means))

# One-way ANOVA (parametric)
groups = [g["NDS"].dropna().values for _, g in nds.groupby("Dataset")]
anova_line = "ANOVA not applicable (need ≥2 datasets with samples)."
if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
    F, p = stats.f_oneway(*groups)
    anova_line = f"One-way ANOVA across datasets: F={F:.3f}, p={p:.4g}"
summary_lines.append("\n" + anova_line)

# Kruskal–Wallis (nonparametric)
kruskal_line = "Kruskal–Wallis not applicable (need ≥2 datasets with samples)."
if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
    H, pkw = stats.kruskal(*groups)
    kruskal_line = f"Kruskal–Wallis across datasets: H={H:.3f}, p={pkw:.4g}"
summary_lines.append(kruskal_line)

with open("stats_summary.txt","w",encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("\n🧮 Stats summary written to stats_summary.txt")
print("\nDone.")
