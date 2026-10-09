# PPI_OC_GP

Implementation of the benchmark in **"Benchmarking one-class learning methods for disease gene prioritization using protein–protein interaction networks"** — a comparison of five families of one-class learning method for ranking candidate disease genes on a protein–protein interaction network, evaluated under a temporal split.

---

## The problem

Disease gene prioritization is not a conventional supervised task. A gene with no recorded association to a disease cannot be treated as a true negative: the absence of evidence usually reflects incomplete knowledge or uneven annotation rather than genuine non-involvement. Labelling every unannotated gene as negative therefore introduces label uncertainty.

This motivates a **positive–unlabeled** formulation. For each disease the reliable signal is the set of genes confirmed as disease-associated before a temporal cutoff. Every remaining gene forms a single unlabeled candidate pool, which the model ranks by its anomaly score. Most methods here are strictly one-class on the confirmed positives; the DROCC variants additionally draw a small sample of unlabeled candidates into training to sharpen the decision boundary. No disease gene that arrived after the cutoff is ever seen during training.

## Benchmark design

| | |
|---|---|
| Diseases | 47 evaluated, spanning 12 ICD-10 chapters |
| Network | STRING v12 protein links, filtered at `combined_score > 700` |
| Features | 128-dimensional Node2Vec node embeddings of the PPI network |
| Candidate pool | 15,554 genes |
| Train | gene–disease associations known **before 2017** |
| Test | associations reported **from 2017 onward**, ranked against the full unlabeled pool |

The temporal split is central to the design. A random split lets later discoveries leak into training and model selection, which overstates prospective performance. Training on pre-2017 associations and testing only on later ones approximates how the method would actually be used.

## Models

Ten configurations are compared across five methodological families, spanning classical kernel methods to deep and graph-based one-class models.

| Script | Family | Configurations                              | Reference |
|---|---|---------------------------------------------|---|
| `src/baselinemodels.py` | One-Class Support Vector Machine | `OCSVM`                                     | Schölkopf et al., 2001 |
| `src/deep_svdd_gene.py` | Deep Support Vector Data Description | `DeepSVDD`                                  | Ruff et al., 2018 |
| `src/drocc_deepsvdd.py` | Deep Robust One-Class Classification | `DROCC-deepsvdd`                            | Goyal et al., 2020 |
| `src/drocc_lf.py` | Deep Robust One-Class Classification, local fine-tuning | `DROCC-LF1`,  `DROCC-LF2`                   | Goyal et al., 2020 |
| `src/ocgnn_gene.py` | One-Class Graph Neural Network | `OCGNN-GCN`, `OCGNN-GAT`, `OCGNN-GraphSAGE` | Wang et al., 2019 |
| `src/olga_gene.py` | One-Class Graph Autoencoder | `OLGA/graph`, `OLGA/knn`                    | Golò et al., 2024 |

`DROCC-LF` extends DROCC with a limited set of close-negative examples and replaces Euclidean distance with a Mahalanobis-based distance to better characterise local geometry in high-dimensional feature spaces. 
`OLGA/knn` builds a nearest-neighbour graph in embedding space rather than using the real interaction network.

## Repository layout

```
PPI_OC_GP/
├── src/
│   ├── baselinemodels.py     OCSVM baseline + shared metric helpers (AUAC, recall@k)
│   ├── deep_svdd_gene.py     DeepSVDD: MLP encoder trained to collapse positives onto a hypersphere
│   ├── drocc_deepsvdd.py     DROCC: adversarial compactness boundary, one-class on positives
│   ├── drocc_lf.py           DROCC-LF: Mahalanobis projection plus close-negative fine-tuning
│   ├── ocgnn_gene.py         OCGNN: GCN / GAT / GraphSAGE encoders with a deviation-net one-class loss
│   └── olga_gene.py          OLGA: GAT graph autoencoder + logistic one-class loss, on real or kNN graph
├── data/                     benchmark inputs (see below)
│   ├── disease_summary_2017.csv
│   ├── ppi_2017_700_emb.csv
│   ├── edge_list_2017_{500,700,800}.edg
│   ├── train_2017/           49 files
│   └── test_2017/            49 files
└── README.md
```

## Requirements

Developed and tested on Python 3.9.

```
numpy  pandas  scikit-learn  networkx  torch        # all models
dgl                                                 # ocgnn_gene.py
torch-geometric                                     # olga_gene.py
```

Verified against torch 2.2.0, dgl 1.1.2, torch-geometric 2.6.1, scikit-learn 0.24.2, networkx 2.5, numpy 1.26.4, pandas 2.2.3.

A GPU is optional. `deep_svdd_gene.py`, `drocc_deepsvdd.py` and `ocgnn_gene.py` detect CUDA automatically; `olga_gene.py` runs on CPU.

## Data

The benchmark inputs are included in this repository under `data/` (68 MB).

```
data/
├── disease_summary_2017.csv        disease registry, one row per disease
├── ppi_2017_700_emb.csv            node features, 15,554 x 129
├── edge_list_2017_500.edg          PPI edges, combined_score > 500
├── edge_list_2017_700.edg          PPI edges, combined_score > 700   <- used
├── edge_list_2017_800.edg          PPI edges, combined_score > 800
├── train_2017/                     genes known before the cutoff, 49 files
└── test_2017/                      genes first reported from 2017 onward, 49 files
```

`disease_summary.csv` carries `disease_id`, `disease_name` and `Gene_number`, and drives the per-disease loop. It lists 49 diseases; 47 are evaluated, after excluding `ICD10_L20` and `ICD10_F90`.

### File formats

**`ppi_2017_700_emb.csv`** — the feature matrix, one row per gene.

```
string_id, feature_1, feature_2, ..., feature_128
```

`string_id` holds STRING protein identifiers and is renamed to `ensembl` at load time, forming the join key against the disease gene lists. The remaining 128 columns are the Node2Vec embedding dimensions.

**`edge_list_2017_*.edg`** — tab-separated, no header, exactly two columns holding the endpoint protein identifiers of each interaction. Three confidence thresholds are provided; the graph models read the `700` variant, matching the filter applied to the embeddings.

**`train_2017/<disease_id>.csv` and `test_2017/<disease_id>.csv`** — one row per gene.

```
disease_id, disease_name, gene_symbol, ensembl
```

### Derived columns

Every script labels the candidate pool by merging the feature matrix against the per-disease gene lists. Two columns are added in memory:

| Column | Meaning |
|---|---|
| `label` | `1` if the gene is a confirmed disease gene, `0` otherwise |
| `test` | `0` for the training split, `1` for held-out positives and the unlabeled pool |

The training set is `label == 1 & test == 0`. The evaluation set is every row with `test == 1` — the held-out positives plus the full unlabeled candidate pool. Features are selected by dropping `ensembl`, `label` and `test`, so column order in the input file matters.

### Provenance

Derived from STRING v12 protein links for *Homo sapiens* (filtered at `combined_score > 700`), node2vec embeddings computed over the resulting PPI graph, and disease–gene associations from DisGeNET indexed by ICD-10.

## Running the benchmark

Scripts are standalone drivers. Each loops over every disease in `disease_summary_2017.csv` and writes per-disease scores plus an aggregate results file.

```bash
python src/baselinemodels.py
python src/deep_svdd_gene.py
python src/drocc_deepsvdd.py
python src/drocc_lf.py
python src/ocgnn_gene.py --module GraphSAGE
python src/olga_gene.py
```

`ocgnn_gene.py` takes `--module` from `GCN`, `GAT`, `GraphSAGE`; run it once per module to reproduce all three configurations.

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

## Outputs

**Per-gene scores**, one CSV per disease:

```
gene_id, y_true, score
```

Files are sorted so the highest-priority candidates come first. Lower score means more inlier-like. The score scale differs between methods — DROCC emits a classifier logit, DeepSVDD a distance to the hypersphere center, and OCGNN and OLGA a squared distance to the learned center offset by the radius — so compare rankings, not raw values across methods.

**Per-disease results**, aggregated over the disease loop:

```
Dataset, Train Size, Test Size, Positive Test, Negative Test, AUAC, R@5, R@10, R@30, R@100
```

**Model checkpoints**, where the method keeps one: `model_<disease_id>_<year>.tar` for DeepSVDD, `{disease_id}+OC-<module>+bestcheckpoint.pt` for OCGNN, `model_deep_<disease_id>_<year>.pt` for DROCC-deepsvdd, `model_LF1_<year>_<disease_id>.pt` for DROCC-LF. OLGA caches its graphs as `.gpickle` and saves no weights.

### Metrics

| Metric | Definition |
|---|---|
| **AUAC** | Area under the accumulation curve, `1 - mean_rank/N` over the positives. Computed from normalized ranks alone, so it needs no assumption about which unlabeled genes are negative — the standard choice when confirmed positives are a tiny fraction of the pool. |
| **R@k** | Fraction of confirmed test positives among the top `k` percent of the ranking. Computed at 5, 10, 30 and 100. |

Position-sensitive measures — **BEDROC@k** and **BioSim@k** — are aggregated downstream of these scripts from the per-gene score files, and are reported alongside AUAC and R@k in the results table below.

## Results

Mean across 47 diseases. R@150 and R@300 count confirmed positives recovered within the top 150 and 300 ranks; BEDROC columns are early-recognition scores on the same cutoffs.

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

A DROCC-style robust boundary trained on a deep one-class encoder ranks confirmed future disease genes highest overall and at the top of the list, followed by DeepSVDD. Graph-native models trail despite having direct access to the same PPI topology. Performance varies by disease and declines as the number of held-out positives grows, reflecting both the behaviour of rank-based metrics under small samples and the heterogeneous genetic architecture of polygenic disease.

## Citation

```bibtex
@article{benchmarking_one_class_gene_prioritization,
  title   = {Benchmarking one-class learning methods for disease gene
             prioritization using protein--protein interaction networks},
  journal = {To be specified},
  year    = {2026},
  note    = {Code available at https://github.com/TehranUni/PPI_OC_GP}
}
```

## License

To be specified.