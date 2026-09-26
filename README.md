# Sajen Shah STEM projects

I am an applied mathematics student at UCLA interested in machine learning and early detection of keratoconus. This repository brings together runnable versions of two projects from my STEM portfolio and a Python analysis of the study data behind my keratoconus research paper.

| Project | What is implemented | What you provide |
| --- | --- | --- |
| [Stock movement](stock_prediction/README.md) | Log-return feature, three-class next-day labels, 16-unit ReLU/Softmax network, chronological holdout | A local CSV of closing prices |
| [Push-up form](pushup_form/README.md) | Pose landmarks, elbow-angle tracking, repetition state machine, on-screen cues | Webcam/video and an official MediaPipe Pose Landmarker model |
| [Keratoconus research](keratoconus_analysis/README.md) | Study-level data from my extraction sheet, statistical analyses, and results | The 28-study dataset is included |

## Quick start

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Install project-specific dependencies only when running that project:

```bash
python -m pip install -e '.[stock]'
python -m pip install -e '.[vision]'
```

See each project README for inputs and commands. Generated models, videos, and private data are ignored by Git.

## My research and project notes

- I wrote *The Role of Artificial Intelligence in the Detection of Keratoconus: A Meta-Analysis of Clinical Trials (1997–2022)* and presented the work at the 2025 World Keratoconus Congress. I examined how model architecture, corneal imaging inputs, and disease severity relate to diagnostic performance.
- My study extraction sheet contains 28 studies. Its [CSV export](keratoconus_analysis/data/kc_studies.csv) retains all rows and source notes. The [Python analysis](keratoconus_analysis/README.md) uses the 20 exact accuracy values available in that sheet; it records which rows could not be included in accuracy comparisons.
- I built the stock classifier around daily log returns and next-day movement classes. The push-up prototype tracks pose landmarks and joint angles to count repetitions and show form cues. The code in this repository is a cleaned-up reconstruction of these portfolio projects.

The statistical comparisons are exploratory; the research README explains the data and interpretation limits. The exercise and stock programs are demonstrations, not medical or financial decision tools.
