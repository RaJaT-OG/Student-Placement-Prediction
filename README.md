# Student Placement Prediction

## Train from the terminal

From the project root in PowerShell:

```powershell
.venv\Scripts\python.exe src\train_placement.py --dataset all
```

Choose a single dataset with `--dataset student_placement_train.csv` or `--dataset campus_recruitment.csv`.

The workflow audits labels, missing values, duplicates, conflicting duplicate feature rows, and identifier columns; imputes/scales/numerically encodes features inside each model pipeline; uses a stratified 80/20 holdout; selects a model with 3-fold ROC-AUC cross-validation on the training partition; then reports holdout accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix. ID columns are excluded to prevent memorizing records. No preprocessing is fitted on the holdout.

Outputs are written to `outputs/`:

- `training_summary.csv` — a concise score summary for each dataset.
- `*_model_comparison.csv` — cross-validation and holdout scores for each candidate model.
- `*_training_report.json` — dataset audit, class metrics, and classification report.
- `training_run.txt` — terminal log from the last run.
- `student_placement_test_predictions.csv` — test probabilities only when held-out ROC-AUC indicates a usable signal.

Models with held-out ROC-AUC below 0.60 are not saved as deployable pipelines. This prevents high majority-class accuracy from being mistaken for a useful model.

## Important dataset finding

The project contains two different feature sets. `student_placement_train.csv` and `student_placement_test.csv` use 26 student profile features and share a schema. `campus_recruitment.csv` has only 10 different features and a separate target. These datasets are trained independently and their feature rows must not be mixed.

The student profile target is 20.4% positive. On a stratified holdout, the best cross-validation-selected model had 78.9% accuracy versus 79.6% for simply predicting “not placed,” balanced accuracy 50.0%, and ROC-AUC 0.489. It has no useful predictive signal, so the workflow does not save a deployable model or test predictions for it.

The campus dataset produced 79.2% holdout accuracy and 0.872 ROC-AUC with Logistic Regression, versus a 58.4% majority-class accuracy baseline. That is a useful ranking signal, but it is not a 90% accurate model and it cannot score the student-profile test file because the features differ.

The training command uses the packages listed in `requirements.txt`. Install them in the project environment if needed with `pip install -r requirements.txt`.
