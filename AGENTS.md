# Repository Instructions

## Project Shape

- This is a Jupyter-based Titanic Kaggle assignment repo, not a packaged Python app; there is no `README`, dependency manifest, CI config, or test runner config.
- Treat `Titanic_Disaster_Challenge-Final.ipynb` as the assignment notebook and primary source of truth.
- Use `Titanic_Experiment_Sandbox.ipynb` for experiments; do not fold sandbox changes into the final notebook unless explicitly requested.
- Treat `Titanic_Survival_Prediction.ipynb` as an older/work notebook, not the final deliverable.
- Assignment/reference PDFs live in `Info/`; train/test CSVs live in `Datasets/`.

## Notebook Workflow

- Final notebook kernel metadata is `titanic-venv`; sandbox uses `python3`. Prefer the existing environment over inventing setup instructions.
- Do not validate notebooks with plain Python `compile()`; notebooks contain IPython syntax such as `%matplotlib inline`.
- Validate notebook JSON with:
  `python -c "import nbformat; nb=nbformat.read('Titanic_Disaster_Challenge-Final.ipynb', as_version=4); nbformat.validate(nb); print(len(nb.cells))"`
- Execute notebooks to `/tmp/opencode` to avoid leaving executed notebook artifacts in the repo:
  `jupyter nbconvert --to notebook --execute "Titanic_Disaster_Challenge-Final.ipynb" --output "/tmp/opencode/Titanic_Disaster_Challenge-Final.executed.ipynb"`
- After final-notebook changes, validate `submission_title_family_hascabin_cv.csv` has 418 rows, columns `PassengerId, Survived`, unique IDs, and labels only `0/1`.

## Modeling Facts To Preserve

- The final assignment notebook intentionally uses shallow learning only: Logistic Regression, Random Forest, Gradient Boosting, and a soft Voting Ensemble.
- The selected final model is Gradient Boosting; Voting Ensemble is included as an evaluated model but should not be forced as final unless CV F1 wins.
- Current verified final comparison: baseline F1 `0.731`, Gradient Boosting F1 `0.792`, Voting Ensemble F1 `0.778`.
- The `Women and Children First` baseline must use original unscaled age values; do not recompute `Age < 16` after scaling.
- Feature engineering is intentionally compact: `Title`, `FamilySize`, `HasCabin`, `FareLog`, binary `Sex`, binary `Embarked_C`, ordinal `Pclass`.

## Deliverable Files

- `video_script.md`: timed presentation guide.
- `video_script_word_for_word.md`: literal read-aloud presentation script.
- `notebook_cheatsheet.md`: preparation notes for explaining notebook concepts.
- `final_notebook_structure_layout.md`: proposed layout plan for future final-notebook cleanup.

## Git Hygiene

- `.gitignore` excludes environments, checkpoints, caches, and model binaries, but not `Datasets/`, `Info/`, or CSV submissions.
- Do not stage datasets, PDFs, generated CSVs, plots, or sandbox outputs unless the user explicitly asks.
- Keep final assignment edits focused on `Titanic_Disaster_Challenge-Final.ipynb` and related markdown support files.
