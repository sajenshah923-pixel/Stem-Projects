# Source map

This code brings together my STEM portfolio, my keratoconus paper, and my 28-study extraction sheet. The research data export retains the original study titles, source links, notes, counts, and reported metric strings.

| Source | Documented detail | Implementation |
| --- | --- | --- |
| STEM portfolio, p. 2 | Camera push-up feedback, MediaPipe/OpenCV, joint-angle calculations, counts and form cues | `src/stem_projects/pushup.py`, `run_pushup.py` |
| STEM portfolio, p. 3 | Closing-price log return, next-day down/flat/up labels, 16-unit ReLU hidden layer, 3-unit Softmax, Adam | `src/stem_projects/stock.py`, `train_stock.py` |
| Research paper, pp. 3–4 | PubMed study selection and recorded network input/type, severity, accuracy, sensitivity, specificity | `keratoconus_analysis/data/kc_studies.csv` |
| Research paper, p. 3 and pp. 7–8 | Shapiro-Wilk, Spearman, Kruskal-Wallis; discussed results | `src/stem_projects/research.py` |
| My `KC.numbers` extraction sheet | 28 study records, recorded model types, imaging inputs, severity counts, and performance values | `keratoconus_analysis/data/kc_studies.csv` |

The stock notebook, exercise source, videos, and price CSV were not supplied. Chronological holdout, training-only scaling, rep thresholds, input checks, and CI are documented reconstruction choices. The research script computes new results from the extracted spreadsheet and retains all omitted accuracy entries in its audit output.
