# PPI_OC_GP

Implementation of the benchmark in **"Benchmarking one-class learning methods for disease gene prioritization using protein–protein interaction networks"** — a systematic comparison of five families of one-class learning method for ranking candidate disease genes on a protein–protein interaction network, evaluated under a temporal split.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Data](#5-data)
6. [Quick Start](#6-quick-start)
7. [Running the Benchmark](#7-running-the-benchmark)
8. [Models](#8-models)
9. [Input Features](#9-input-features)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Output Format](#11-output-format)
12. [Reproducing Results](#12-reproducing-results)
13. [Citation](#13-citation)

---

## 1. Overview

### Task

Given the set of genes confirmed as disease-associated for a disease up to a temporal cutoff (before 2017), rank all candidate genes so that genuinely novel associations — those first reported from 2017 onward — appear at the top of the list.

### Approach

Disease gene prioritization is not a conventional supervised task. A gene with no recorded association to a disease cannot be treated as a true negative: the absence of evidence usually reflects incomplete knowledge or uneven annotation rather than genuine non-involvement. Labelling every unannotated gene as negative therefore introduces label uncertainty.

This motivates a **positive–unlabeled** formulation. The reliable signal is the set of confirmed disease genes; every remaining gene forms a single unlabeled candidate pool which the model ranks by its anomaly score. Most methods here are strictly one-class on the confirmed positives; the DROCC variants additionally draw a small sample of unlabeled candidates into training to sharpen the decision boundary. No disease gene that arrived after the cutoff is ever seen during training.

Ten configurations are compared across five methodological families:

| Script | Family | Configurations | Reference |
|---|---|---|---|
| `src/baselinemodels.py` | One-Class Support Vector Machine | `OCSVM` | Schölkopf et al., 2001 |
| `src/deep_svdd_gene.py` | Deep Support Vector Data Description | `DeepSVDD` | Ruff et al., 2018 |
| `src/drocc_deepsvdd.py` | Deep Robust One-Class Classification | `DROCC-deepsvdd` | Goyal et al., 2020 |
| `src/drocc_lf.py` | Deep Robust One-Class Classification, local fine-tuning | `DROCC-LF1`, `DROCC-LF2` | Goyal et al., 2020 |
| `src/ocgnn_gene.py` | One-Class Graph Neural Network | `OCGNN-GCN`, `OCGNN-GAT`, `OCGNN-GraphSAGE` | Wang et al., 2019 |
| `src/olga_gene.py` | One-Class Graph Autoencoder | `OLGA/graph`, `OLGA/knn` | Golò et al., 2024 |

`DROCC-LF` extends DROCC with a limited set of close-negative examples and replaces Euclidean distance with a Mahalanobis-based distance to better characterise local geometry in high-dimensional feature spaces.

`OLGA/knn` builds a nearest-neighbour graph in embedding space rather than using the real interaction network.

### Benchmark design

| | |
|---|---|
| Diseases | 47 evaluated, spanning 12 ICD-10 chapters |
| Network | STRING v12 protein links, filtered at `combined_score > 700` |
| Features | 128-dimensional Node2Vec node embeddings of the PPI network |
| Candidate pool | 15,554 genes |
| Train | gene–disease associations known **before 2017** |
| Test | associations reported **from 2017 onward**, ranked against the full unlabeled pool |

### Temporal validation

The temporal split is central to the design. A random split lets later discoveries leak into training and model selection, which overstates prospective performance. Training on pre-2017 associations and testing only on later ones approximates how the method would actually be used.

`disease_summary_2017.csv` lists 49 diseases. 47 are evaluated, after excluding `ICD10_L20` and `ICD10_F90`.

---

## 2. Repository Structure

```
PPI_OC_GP/
├── src/
│   ├── baselinemodels.py     OCSVM baseline + shared metric helpers (AUAC, recall@k)
│   ├── deep_svdd_gene.py     DeepSVDD: MLP encoder trained to collapse positives onto a hypersphere
│   ├── drocc_deepsvdd.py     DROCC: adversarial compactness boundary, one-class on positives
│   ├── drocc_lf.py           DROCC-LF: Mahalanobis projection plus close-negative fine-tuning
│   ├── ocgnn_gene.py         OCGNN: GCN / GAT / GraphSAGE encoders with a deviation-net one-class loss
│   └── olga_gene.py          OLGA: GAT graph autoencoder + logistic one-class loss, on real or kNN graph
├── data/                     benchmark inputs (see §5)
│   ├── disease_summary_2017.csv
│   ├── ppi_2017_700_emb.csv
│   ├── edge_list_2017_{500,700,800}.edg
│   ├── train_2017/           49 files
│   ├── test_2017/            49 files
│   └── close_neg/            close negatives, required by DROCC-LF
└── README.md
```

---

## 3. Requirements

Python ≥ 3.9 is recommended. Developed and tested on Python 3.9.

### Core dependencies

```
numpy  pandas  scikit-learn  networkx  torch        # all models
```

### Model-specific dependencies

```
dgl                                                 # ocgnn_gene.py
torch-geometric                                     # olga_gene.py
```

Verified against torch 2.2.0, dgl 1.1.2, torch-geometric 2.6.1, scikit-learn 0.24.2, networkx 2.5, numpy 1.26.4, pandas 2.2.3.

A GPU is optional. `deep_svdd_gene.py`, `drocc_deepsvdd.py` and `ocgnn_gene.py` detect CUDA automatically; `olga_gene.py` runs on CPU.

---

## 4. Installation

```bash
git clone https://github.com/TehranUni/PPI_OC_GP.git
cd PPI_OC_GP

python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn networkx torch
pip install dgl torch-geometric
```

---

## 5. Data

The benchmark inputs are included in this repository under `data/` (68 MB).

```
data/
├── disease_summary_2017.csv        disease registry, one row per disease
├── ppi_2017_700_emb.csv            node features, 15,554 x 129
├── edge_list_2017_500.edg          PPI edges, combined_score > 500
├── edge_list_2017_700.edg          PPI edges, combined_score > 700   <- used
├── edge_list_2017_800.edg          PPI edges, combined_score > 800
├── train_2017/                     genes known before the cutoff, 49 files
├── test_2017/                      genes first reported from 2017 onward, 49 files
└── close_neg/                      close negatives per disease, used by DROCC-LF only
```

### Required files

| Path (relative to repo root) | Description | Used by |
|---|---|---|
| `data/disease_summary_2017.csv` | Disease registry, 49 rows | all models |
| `data/ppi_2017_700_emb.csv` | Node2Vec node embeddings, 15,554 × 129 | all models |
| `data/edge_list_2017_700.edg` | PPI edge list, `combined_score > 700` | `ocgnn_gene.py`, `olga_gene.py` |
| `data/train_2017/<disease_id>.csv` | Training positives, 49 files | all models |
| `data/test_2017/<disease_id>.csv` | Test positives, 49 files | all models |
| `data/close_neg/<disease_id>.csv` | Close negatives, per disease | `drocc_lf.py` only |


### File formats

**`ppi_2017_700_emb.csv`** — the feature matrix, one row per gene.

```
string_id, feature_1, feature_2, ..., feature_128
```

`string_id` holds STRING protein identifiers and is renamed to `ensembl` at load time, forming the join key against the disease gene lists. The remaining 128 columns are the Node2Vec embedding dimensions.

**`edge_list_2017_*.edg`** — tab-separated, no header, exactly two columns holding the endpoint protein identifiers of each interaction.

**`train_2017/<disease_id>.csv`, `test_2017/<disease_id>.csv`, `close_neg/<disease_id>.csv`** — one row per gene.

```
disease_id, disease_name, gene_symbol, ensembl
```

### Provenance

Derived from STRING v12 protein links for *Homo sapiens* (filtered at `combined_score > 700`), node2vec embeddings computed over the resulting PPI graph, and disease–gene associations from DisGeNET indexed by ICD-10.

---

## 6. Quick Start

### Verify the data is present

Read-only, about a second:

```bash
cd PPI_OC_GP
python -c "
from pathlib import Path
required = [
    'data/disease_summary_2017.csv',
    'data/ppi_2017_700_emb.csv',
    'data/edge_list_2017_700.edg',
    'data/train_2017',
    'data/test_2017',
    'data/close_neg',
]
for f in required:
    p = Path(f)
    print('OK     ' if p.exists() else 'MISSING', f)
"
```

### Run a single disease

To confirm a model runs end to end before launching the full sweep, restrict the disease loop to one ID. `year` and the disease list are set near the bottom of each script:

```bash
python src/drocc_deepsvdd.py          # DROCC-deepsvdd
python src/ocgnn_gene.py --module GraphSAGE
```

---

## 7. Running the Benchmark

Each script is a standalone driver. It loops over every disease in `disease_summary_2017.csv` and writes per-disease scores plus an aggregate results file.

```bash
python src/baselinemodels.py
python src/deep_svdd_gene.py
python src/drocc_deepsvdd.py
python src/drocc_lf.py
python src/ocgnn_gene.py --module GraphSAGE
python src/olga_gene.py
```

`ocgnn_gene.py` takes `--module` from `GCN`, `GAT`, `GraphSAGE`; run it once per module to reproduce all three configurations.

The two DROCC scripts resolve `data/` relative to the **working directory**, so launch them from the repository root rather than from `src/`. This is inherited from running the original notebooks inside their own working folders. The remaining scripts carry a legacy `path` constant naming a Google Drive mount from the original Colab development, which is not used to read the shipped data.

Output directories are **not** created automatically. Create them before running:

```
models/OCSVM/{scores}/
models/DeepSVDD/{scores,results,model}/
models/OCGNN/{scores,checkpoints}/
models/OLGA/{scores,olga_knn_2017/<disease_id>/k=<k>}/
data/DROCC/{scores,models,results}/
results/{OCSVM,DeepSVDD,DROCC_LF1,OCGNN,OLGA}/
log/
```

---

## 8. Models

### 8.1 One-Class SVM (`baselinemodels.py`)

`sklearn.svm.OneClassSVM` with an RBF kernel, fitted on the standardized confirmed positives only. The decision function supplies the anomaly score, where a larger value means more normal.

Features are standardized with `StandardScaler` before fitting. Defaults: `kernel='rbf'`, `nu=0.05`, `gamma='scale'`.

### 8.2 Deep SVDD (`deep_svdd_gene.py`)

An MLP encoder φ maps each gene embedding onto a hypersphere centred at `c`, the mean representation of the training positives. Two objectives are available:

- `one-class` (default) — minimise the mean squared distance to the centre, `mean(‖φ(x) − c‖²)`
- `soft-boundary` — `radius² + (1/ν)·mean(max(0, ‖φ(x) − c‖² − radius²))`, with `radius` re-estimated as the `(1−ν)` quantile of the distance after a 10-epoch warm-up

Encoder `NNet` is `Linear(size, 150) → ReLU → Linear(150, 100) → Sigmoid`, with `rep_dim = 100`. An optional reconstruction autoencoder pretraining phase runs before the one-class phase.


### 8.3 DROCC (`drocc_deepsvdd.py`)

Deep Robust One-Class Classification. After an initial cross-entropy phase, an adversarial compactness term is added: for each positive, gradient ascent finds the point that most resembles the positive class, and that adversarial point is projected onto the annulus of radii `[r, γ·r]` around it. Tightening this annulus forces the learned boundary to stay compact around the inliers rather than spreading across the candidate pool.

Scoring network `NNet` is `size → 512 → 256 → 128 → 1`, with LayerNorm, ELU and dropout 0.3 at each stage.

### 8.4 DROCC-LF (`drocc_lf.py`)

Local fine-tuning on top of DROCC. Two changes separate it from §8.3:

1. **Mahalanobis projection.** The adversarial point is projected using a Mahalanobis-based distance rather than a Euclidean clamp, which better captures local geometry in high-dimensional embedding space. The projection is obtained by numerically solving a constrained optimisation problem over the step length.
2. **Close negatives.** A limited set of close negatives is used during validation, so the model is selected on its ability to separate confirmed positives from nearby non-disease genes rather than from arbitrary background.

The validation split uses positive and close-negative scores; the checkpoint with the best validation AUC is retained.

### 8.5 OCGNN (`ocgnn_gene.py`)

A one-class graph neural network following the deviation-net formulation. A GCN, GAT or GraphSAGE encoder produces node representations; the model learns a centre `c` and a radius, and the anomaly score is `‖φ(x) − c‖² − radius²`. The training objective is

```
radius² + (1/ν) · mean(max(0, score))
```

The centre is initialised from the mean of the first forward pass, and the radius is set from the `(1 − ν)` quantile of the distances.

### 8.6 OLGA (`olga_gene.py`)

A One-Class Graph Autoencoder. The encoder is `GATConv(128 → 48) → Tanh → GATConv(48 → 2)` inside a PyTorch Geometric graph autoencoder with an inner-product decoder, so the representation is two-dimensional and directly inspectable.

Training runs in two phases: an initial reconstruction phase, then a joint objective combining the unsupervised reconstruction loss with the one-class logistic loss

```
score = ‖φ(x) − c‖² − radius²
loss  = mean(where(score > 0, score + 1, exp(score)))
```

Points inside the hypersphere pay an exponential term, points outside pay a linear one — the logistic formulation that distinguishes OLGA from plain reconstruction error. The radius is fixed at `0.35` in the shipped configuration.

---

## 9. Input Features

All models consume the same single feature representation: 128-dimensional Node2Vec embeddings of the PPI network, one row per gene, with `string_id` as the join key.

| Column | Type | Meaning |
|---|---|---|
| `string_id` | string | STRING protein identifier, renamed to `ensembl` at load time |
| `feature_1` … `feature_128` | float | Node2Vec embedding dimensions |

Features are selected by dropping `ensembl`, `label` and `test` from the labelled frame, so **column order in the input file matters** — the gene identifier must come first and the derived label columns last.

Unlike the fusion-based benchmarks that concatenate sequence, literature and network views, every configuration here runs on this single PPI-derived representation, which keeps the comparison between methods clean.

---

## 10. Evaluation Protocol

### Temporal split

```
label = 1   gene is a confirmed disease gene
label = 0   gene is not confirmed
test  = 0   training split
test  = 1   held-out positives and the unlabeled candidate pool
```

Training uses `label == 1 & test == 0`. Evaluation ranks every row with `test == 1` — the held-out positives plus the full unlabeled pool of 15,554 genes. The DROCC variants additionally sample unlabeled genes into training.

### Metrics

| Metric | Description |
|---|---|
| `AUAC` | Area under the accumulation curve, `1 − mean_rank/N` over the positives |
| `R@k` | Fraction of confirmed test positives among the top `k` percent of the ranking, computed at k = 5, 10, 30, 100 |
| `BEDROC@k` | Early-recognition score, aggregated downstream from the per-gene score files |
| `BioSim@k` | Gene-set enrichment similarity to known disease biology, aggregated downstream |

AUAC is computed from normalized ranks alone, so it requires no assumption about which unlabeled genes are negative — the standard choice when confirmed positives are a tiny fraction of the candidate pool.

---

## 11. Output Format

### Per-gene scores

One CSV per disease, sorted so the highest-priority candidates come first. Lower score means more inlier-like.

```
gene_id, y_true, score
```

The score scale differs between methods — DROCC emits a classifier logit, DeepSVDD a distance to the hypersphere center, and OCGNN and OLGA a squared distance to the learned center offset by the radius — so compare rankings, not raw values across methods.

### Per-disease results

Aggregated over the disease loop:

```
Dataset, Train Size, Test Size, Positive Test, Negative Test, AUAC, R@5, R@10, R@30, R@100
```

### Model checkpoints

Where the method keeps one: `model_<disease_id>_<year>.tar` for DeepSVDD, `{disease_id}+OC-<module>+bestcheckpoint.pt` for OCGNN, `model_deep_<disease_id>_<year>.pt` for DROCC-deepsvdd, `model_LF1_<year>_<disease_id>.pt` for DROCC-LF. OLGA caches its graphs as `.gpickle` and saves no weights.

---

## 12. Reproducing Results

### Step 1 — Clone and install

```bash
git clone https://github.com/TehranUni/PPI_OC_GP.git
cd PPI_OC_GP
# Install dependencies (see §4)
```

### Step 2 — Check the data

Confirm every file listed in §5 is present.

```bash
python -c "
from pathlib import Path
for f in ['data/disease_summary_2017.csv', 'data/ppi_2017_700_emb.csv', 'data/close_neg']:
    print('OK' if Path(f).exists() else 'MISSING', f)
"
```

### Step 3 — Create the output directories

```bash
mkdir -p models/{OCSVM/scores,DeepSVDD/{scores,results,model},OCGNN/{scores,checkpoints},OLGA/scores} \
         data/DROCC/{scores,models,results} \
         results/{OCSVM,DeepSVDD,DROCC_LF1,OCGNN,OLGA} log
```

### Step 4 — Run

Each script sweeps all diseases and writes an aggregate results file. For the graph models, run `ocgnn_gene.py` once per module and `olga_gene.py` for both the real and nearest-neighbour graphs.

### Step 5 — Compare

Aggregate the per-gene score files into the reported table. `mean` across the 47 evaluated diseases:

| Model | n | AUAC | R@150 | R@300 | BEDROC@150 | BEDROC@300 |
|---|---|---|---|---|---|---|
| DROCC-deepsvdd | 47 | **0.853** | **0.258** | **0.297** | **0.165** | **0.223** |
| DeepSVDD | 47 | 0.823 | 0.131 | 0.184 | 0.084 | 0.128 |
| DROCC-LF1 | 47 | 0.748 | 0.104 | 0.191 | 0.049 | 0.091 |
| DROCC-LF2 | 47 | 0.732 | 0.045 | 0.139 | 0.040 | 0.066 |
| OCGNN-GAT | 47 | 0.724 | 0.116 | 0.202 | 0.027 | 0.058 |
| OCSVM | 47 | 0.723 | 0.056 | 0.121 | 0.038 | 0.069 |
| OLGA/knn | 47 | 0.698 | 0.042 | 0.140 | 0.015 | 0.031 |
| OCGNN-GCN | 46 | 0.683 | 0.034 | 0.085 | 0.010 | 0.025 |
| OCGNN-GraphSAGE | 47 | 0.682 | 0.086 | 0.177 | 0.025 | 0.050 |
| OLGA/graph | 46 | 0.662 | 0.060 | 0.120 | 0.012 | 0.029 |

R@150 and R@300 count confirmed positives recovered within the top 150 and 300 ranks; BEDROC columns are early-recognition scores on the same cutoffs.

A DROCC-style robust boundary trained on a deep one-class encoder ranks confirmed future disease genes highest overall and at the top of the list, followed by DeepSVDD. Graph-native models trail despite having direct access to the same PPI topology. Performance varies by disease and declines as the number of held-out positives grows, reflecting both the behaviour of rank-based metrics under small samples and the heterogeneous genetic architecture of polygenic disease.

### Randomness

The data splits are fixed with `random_state=42`. GPU nondeterminism and unseeded sampling inside the DROCC adversarial solver mean exact scores may vary slightly between runs and hardware.

---

## 13. Citation

```bibtex
@article{benchmarking_one_class_gene_prioritization,
  title   = {Benchmarking one-class learning methods for disease gene
             prioritization using protein--protein interaction networks},
  journal = {To be specified},
  year    = {2026},
  note    = {Code available at https://github.com/TehranUni/PPI_OC_GP}
}
```

---

## License

To be specified.