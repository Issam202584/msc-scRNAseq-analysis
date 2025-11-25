# MSC scRNA-seq Analysis

This repository contains Python scripts used to analyse a public single-cell RNA-seq dataset of human mesenchymal stem/stromal cells (MSCs) and identify a neural-primed subpopulation.

## Dataset

All data analysed in this project are publicly available from the Gene Expression Omnibus (GEO):

- **GSE143753** – human MSC single-cell RNA-seq dataset

The scripts in this repository operate on processed data downloaded from GEO.

## Contents

- `msc_neural_download.py` – download / preparation of GEO data for analysis  
- `msc_neural_meta.py` – generation and handling of cell and sample metadata  
- `msc_nds_analysis.py` – main single-cell workflow (filtering, normalisation, clustering, dimensionality reduction, lineage scoring, pseudotime)  
- `msc_nds_stats.py` – statistical summaries and export of results

## Requirements

The analysis was run in Python using standard scientific libraries, including for example:

- `scanpy`
- `anndata`
- `numpy`
- `pandas`
- `matplotlib`

Exact versions correspond to those used in the manuscript.

## Reproducibility

These scripts are provided to document the computational steps used in the manuscript and to help other researchers reproduce or extend the analysis of GSE143753.
