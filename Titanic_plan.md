# Titanic Notebook Plan

## Checkpoint 1: EDA
- Load train and test data.
- Inspect shape, dtypes, missing values, and class balance.
- Summarize key patterns with plots for categorical and numerical features.
- Verify no row drops are needed.

## Checkpoint 2: Feature Engineering
- Keep the notebook aligned with the assignment and explicitly document leakage handling.
- Build features with a train-safe workflow.
- Extract `Title` from `Name` and simplify to `Mr`, `Mrs`, `Miss`, `Master`, `Rare`.
- Create `FamilySize` from `SibSp + Parch + 1`.
- Add `HasCabin` as a binary indicator.
- Impute `Age` using train-only medians by `Title`.
- Impute `Fare` using train-only medians by `Pclass`.
- Create `FareLog = log1p(Fare)`.
- Encode `Sex` as binary and `Embarked_C` as a single binary feature.
- Encode `Pclass` ordinally.
- Make the leakage statement explicit: test data may be used only for domain-driven feature construction, while all learned preprocessing stays fit on train only.

## Checkpoint 3: Model Training
- Replace loose preprocessing with an sklearn `Pipeline` + `ColumnTransformer` where possible.
- Fit preprocessing inside the CV loop only.
- Compare three models:
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Tune each model with `GridSearchCV` on F1.
- Produce OOF predictions for each model.
- Compare each model against the baseline from Section 2.

## Checkpoint 4: Final Evaluation
- Select the best model by CV F1.
- Retrain the selected model on the full training set.
- Produce compact error analysis with confusion matrix and FP/FN counts.
- Plot feature importance or permutation importance.
- State clearly what the model learned and where it still fails.

## Checkpoint 5: Final Prediction and Submission
- Predict the Kaggle test set with the final model.
- Show a submission preview table.
- Run sanity checks on shape and label values.
- Save the file as `submission_title_family_hascabin_cv.csv`.
- Keep this section separate from the model-evaluation section.

## Checkpoint 6: Submission Review
- Compare the new submission to prior attempts.
- If the score is still weak, iterate only on leakage-safe feature engineering or pipeline tuning.
- Avoid broad notebook rewrites unless the score improves.
