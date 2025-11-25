# msc_neural_download.py  — robust version (loads FAMILY SOFT with GPL)
# It will:
# - download & parse the full GSE (family.soft.gz) with platform
# - map probe IDs to gene symbols (parsing gene_assignment if needed)
# - aggregate probes per gene
# - extract canonical MSC↔neural markers (with synonyms)
# - save a tidy CSV + a plot

import sys, re, os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import GEOparse

sns.set(context="talk", style="whitegrid")

# ----------------------------
# Config (you can change these)
# ----------------------------
GSE_ID = "GSE66554"
MARKERS = ["TUBB3", "MAP2", "SOX2", "NES", "CD73", "CD90", "CD105"]

# Canonical → synonyms (UPPERCASE)
SYNS = {
    "TUBB3": ["TUBB3", "TUBB-3", "TUBBIII", "BETA-III-TUBULIN", "BETA III TUBULIN"],
    "MAP2" : ["MAP2", "MICROTUBULE-ASSOCIATED PROTEIN 2"],
    "SOX2" : ["SOX2"],
    "NES"  : ["NES", "NESTIN"],
    "CD73" : ["CD73", "NT5E"],
    "CD90" : ["CD90", "THY1"],
    "CD105": ["CD105", "ENG"],
}

print(f"🔎 GSE: {GSE_ID}")
print(f"🧬 Markers (canonical): {', '.join(MARKERS)}")

# ----------------------------
# Load the FAMILY SOFT (with GPL)
# ----------------------------
print("📥 Downloading + parsing GEO FAMILY SOFT (with platform)…")
# IMPORTANT: do NOT use how="quick"
gse = GEOparse.get_GEO(geo=GSE_ID, destdir="./")  # family.soft.gz (has GPL)

if not gse.gpls:
    # Fallback: if for any reason we only got the TXT (quick), force family soft file
    fam_path = os.path.join(".", f"{GSE_ID}_family.soft.gz")
    if os.path.exists(fam_path):
        print("ℹ️ GPL missing in object; reloading from family SOFT file directly…")
        gse = GEOparse.get_GEO(filepath=fam_path)
    else:
        raise RuntimeError("No GPL present. Delete GSE66554.txt if created earlier and rerun.")

# ----------------------------
# Get platform annotation table
# ----------------------------
gpl = list(gse.gpls.values())[0]
anno = gpl.table.copy()
anno.columns = anno.columns.str.strip()

# Pick a symbol column
def pick_symbol_col(df):
    prefs = ["Gene Symbol","GENE_SYMBOL","SYMBOL","gene_symbol","Gene symbol","Symbol",
             "GeneSymbol","GENE_SYMBOLS","gene_assignment"]
    for c in prefs:
        if c in df.columns: 
            return c
    # fallback: any column containing 'symbol'
    for c in df.columns:
        if "symbol" in c.lower():
            return c
    # last chance: gene_assignment will be parsed specially later
    if "gene_assignment" in df.columns:
        return "gene_assignment"
    raise RuntimeError("Could not find any gene symbol column in GPL.")

sym_col = pick_symbol_col(anno)
id_col = "ID" if "ID" in anno.columns else anno.columns[0]
anno = anno[[id_col, sym_col]].rename(columns={id_col: "ID_REF", sym_col: "SYMBOL_RAW"})

# Normalize or parse gene symbols
def norm_symbol(x):
    if pd.isna(x): 
        return None
    s = str(x).strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s if s else None

if sym_col.lower() == "gene_assignment":
    # many platforms put "ACC // SYMBOL // description // ..."
    def extract_symbol(cell):
        if pd.isna(cell):
            return None
        parts = str(cell).split("//")
        if len(parts) > 1:
            symbol = parts[1].strip().split()[0]
            return symbol.upper()
        # sometimes the first token is the symbol
        first = str(cell).split("//")[0].strip()
        return first.upper() if first else None
    anno["SYMBOL"] = anno["SYMBOL_RAW"].map(extract_symbol)
else:
    anno["SYMBOL"] = anno["SYMBOL_RAW"].map(norm_symbol)

anno = anno.dropna(subset=["SYMBOL"]).drop_duplicates(subset=["ID_REF"])

# ----------------------------
# Build expression matrix (probes × samples)
# ----------------------------
expr = gse.pivot_samples("VALUE")  # rows: probe IDs; cols: GSM samples
if expr.index.name != "ID_REF":
    # GEOparse usually sets row index to ID_REF
    expr.index.name = "ID_REF"
expr = expr.reset_index()

# Merge annotation → map probes to gene symbols
merged = expr.merge(anno[["ID_REF","SYMBOL"]], on="ID_REF", how="left")
merged = merged.dropna(subset=["SYMBOL"])

# Aggregate multiple probes → mean per gene, per sample
value_cols = [c for c in merged.columns if c not in ("ID_REF","SYMBOL")]
gene_mean = (merged
             .groupby("SYMBOL", as_index=False)[value_cols]
             .mean())

# ----------------------------
# Filter to requested markers (include synonyms)
# ----------------------------
keep_rows = []
matched = {}
gene_mean["SYMBOL"] = gene_mean["SYMBOL"].map(norm_symbol)

for canon, syns in SYNS.items():
    syns_up = [s.upper() for s in syns]
    sub = gene_mean[gene_mean["SYMBOL"].isin(syns_up)].copy()
    if not sub.empty:
        # rename to canonical gene for plotting consistency
        sub["GENE"] = canon
        keep_rows.append(sub)
        matched[canon] = sorted(sub["SYMBOL"].unique())
    else:
        matched[canon] = []

if not keep_rows:
    # help debugging
    gene_mean.head(50).to_csv("DEBUG_gene_mean_head.csv", index=False)
    anno.head(50).to_csv("DEBUG_gpl_head.csv", index=False)
    print("⚠️ No markers matched. Wrote DEBUG_gene_mean_head.csv and DEBUG_gpl_head.csv")
    raise SystemExit

filtered = pd.concat(keep_rows, ignore_index=True)

# Melt to long for plotting
long_df = filtered.melt(id_vars=["GENE","SYMBOL"], var_name="Sample", value_name="Expression")
# Only keep sample columns (drop ID_REF/GENE/SYMBOL artefacts if any)
long_df = long_df[long_df["Sample"].str.startswith("GSM")]

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(10,6))
sns.boxplot(data=long_df, x="GENE", y="Expression", palette="viridis")
sns.stripplot(data=long_df, x="GENE", y="Expression", color="k", alpha=0.45)
plt.title(f"Marker expression across samples ({GSE_ID})")
plt.xlabel("Canonical marker")
plt.ylabel("Expression (array units)")
plt.tight_layout()
plt.savefig("GSE_markers_plot.png", dpi=300)
plt.show()

# ----------------------------
# Save outputs
# ----------------------------
long_df.sort_values(["GENE","Sample","SYMBOL"]).to_csv("GSE_markers_long.csv", index=False)

print("\n✅ Matched symbols per canonical marker:")
for k, v in matched.items():
    print(f"  {k}: {', '.join(v) if v else '(none)'}")

print("\n📊 Saved files:")
print("  - GSE_markers_plot.png")
print("  - GSE_markers_long.csv")
