# AI detection of keratoconus

I wrote *The Role of Artificial Intelligence in the Detection of Keratoconus: A Meta-Analysis of Clinical Trials (1997–2022)* after researching how AI might improve early detection. I presented my findings and patient perspective at the 2025 World Keratoconus Congress. In the paper, I compared reported AI diagnostic performance across model types, corneal imaging inputs, and disease severity.

## Study data

My `KC.numbers` extraction sheet records 28 study rows. I exported it to [data/kc_studies.csv](data/kc_studies.csv), keeping the study titles, article links, extraction notes, reported population and severity counts, and the original accuracy, sensitivity, and specificity strings. The export preserves entries such as `>95.5`, `—`, and blanks exactly as recorded.

| Reported measure | Exact percentage | Missing | Bound |
| --- | ---: | ---: | ---: |
| Accuracy | 20 | 7 | 1 |
| Sensitivity | 22 | 4 | 2 |
| Specificity | 22 | 5 | 1 |

The export is a record of my extraction sheet. Some source notes indicate incomplete population information or pending review; the data should be checked against the cited articles before a formal update to the paper.

## Analysis

My paper described a statistical analysis in R 4.2.1. The code here is a Python/SciPy implementation that operates on the included extraction sheet. It summarizes the exact accuracy, sensitivity, and specificity entries; runs Shapiro-Wilk on study-level accuracy; tests Spearman correlation for publication year versus accuracy; and uses Kruskal-Wallis comparisons by recorded model family and imaging input. The model and imaging groups are derived from the original text columns without changing them.

Run from the repository root:

```bash
python -m pip install -e .
python -m stem_projects.research
```

The current output is included in [results/analysis.json](results/analysis.json). Run the command again after editing the data to generate `outputs/research_analysis.json`.

| Exploratory result using exact entries | Value |
| --- | ---: |
| Mean reported accuracy, unweighted | 96.369% |
| Median reported accuracy | 97.775% |
| Mean reported sensitivity, unweighted (22 entries) | 95.575% |
| Mean reported specificity, unweighted (22 entries) | 96.203% |
| Year versus accuracy, Spearman ρ | −0.373; p = 0.105 |
| Network-family comparison, Kruskal-Wallis | p = 0.237 |
| Imaging-input comparison, Kruskal-Wallis | p = 0.101 |

These are unweighted calculations from the available sheet. Two network categories contain a single study each. The audit flags a decision-tree study recorded as `CNN` in my extraction sheet (`KC03`), which should be checked against the article before drawing conclusions about the model comparison. The sheet gives counts of mild, moderate, and severe cases, but no separate accuracy value for each severity group. Therefore this implementation does not run a severity-versus-accuracy test from those counts. The [JSON result](results/analysis.json) lists the source notes, excluded accuracy entries, and group sizes so the calculation is auditable.
