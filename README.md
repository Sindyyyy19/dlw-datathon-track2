# DLW Datathon 2026, Track 2: Intelligent Financial Fraud Detection

Team working repo. Private: do not share outside the team until the datathon is over.

## What to submit (deadline 11:30 AM, Sun 4 Oct)

Everything is in `submission/`:

| File | What |
|---|---|
| `fraud_detection_track2.ipynb` | The notebook (runs top to bottom with no edits) |
| `model.pkl` | Trained model in the handbook's `(model, feature_names)` format |
| `technical_report.pdf` | The one-page report |
| `requirements.txt` | Pinned package versions |
| `*_Prediction.csv` | Predictions as 0/1 labels (handbook format) |
| `*_Prediction_probabilities.csv` | Predictions as fraud probabilities |
| `*_Prediction_with_ids.csv` | Both, with transaction ids, for our own checking |
| `JUDGES_QA.md` | Plain-English answers to likely judge questions. Read before presenting. |

Replace `TEAM_NAME` in the file names and in the notebook's first cell with our team name.

## Run it yourself

**Google Colab:** upload `submission/fraud_detection_track2.ipynb`, upload the two CSVs from `data/` (anywhere: the
notebook finds them), `Runtime > Restart session and run all`. LightGBM and scikit-learn are preinstalled on Colab; if a
version error appears, run `!pip install -r requirements.txt` first.

**Locally:**

    python -m venv .venv && source .venv/bin/activate
    pip install -r submission/requirements.txt jupyter
    cp data/*.csv submission/ && cd submission && jupyter notebook

## Where we stand

| | |
|---|---|
| Model | Soft-vote blend: regularised logistic regression + small LightGBM (3 leaves, 200 trees) |
| Cross-validated PR-AUC | 0.206 (random = 0.018) |
| Cross-validated ROC-AUC | 0.785 |
| At our threshold (0.554) | recall 34%, precision 20%, F1 0.25, 3% of transactions flagged |
| Best-F1 threshold (0.67) | F1 0.29, recall 25% |

A large LightGBM and HistGradientBoosting both scored worse than the logistic baseline: with 353 fraud rows, small
models win. Details and the comparison table are in the notebook, sections 6 and 7.

## Open questions for the organisers

1. Does the private evaluation call `model.predict()` or `model.predict_proba()`?
2. Should the prediction CSV hold probabilities or 0/1 labels?
3. For F1 and recall, is the threshold ours or a fixed 0.5?

Until answered, `model.predict()` returns 0/1 at our threshold and `model.predict_proba()` returns the score.

## Other folders

- `data/`: the organisers' Track 2 training and testing CSVs, unchanged.
- `experiments/`: the scripts behind the notebook (`experiment.py`, `experiment2.py` for the model comparison and
  regularisation sweep; `final_model.py`; `build_notebook.py` regenerates the notebook; `report.html` is the report source).
- `datathon-handbook.pdf`: the official handbook.
