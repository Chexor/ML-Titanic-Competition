# Final Notebook Structure Layout

Target notebook: `Titanic_Disaster_Challenge-Final.ipynb`

Purpose: define the final layout for restructuring the notebook. The goal is to improve readability, presentation flow, and assignment clarity without changing the core modeling logic.

## Layout Goals

- Keep the notebook understandable within a 10-15 minute presentation.
- Keep the assignment version more polished than the experiment sandbox.
- Preserve the current final model logic unless a clear issue is found.
- Make the story flow from setup, to data, to EDA, to features, baseline, models, evaluation, and submission.
- Avoid making the notebook look like a raw experiment log.
- Keep code cells readable with short, useful comments.
- Keep markdown concise but explanatory.

## Proposed Top-Level Structure

1. Notebook Setup
2. Load Data
3. Exploratory Data Analysis
4. Feature Engineering and Preprocessing
5. Baseline Model
6. Model Training and Validation
7. Model Comparison
8. Final Model Evaluation
9. Kaggle Submission
10. Sources and AI Tools Used

---

## 1. Notebook Setup

### Purpose
Load all required libraries, configure notebook display settings, and suppress warnings. This is the standard starting point for any notebook.

### Proposed Layout

```markdown
## 1. Notebook Setup
```

### Content To Include

- Import `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`.
- Import scikit-learn modules.
- Set `%matplotlib inline`.
- Set warning filter to ignore.
- Set display options for pandas.

---

## 2. Load Data

### Purpose
Load the training and test datasets into the notebook.

### Proposed Layout

```markdown
## 2. Load Data

### 2.1 Load Training Data
### 2.2 Load Test Data
```

### Content To Include

- Load `Datasets/train.csv`.
- Print shape.
- Show first rows.
- Load `Datasets/test.csv`.
- Print shape.
- Show first rows.

### Keep / Change

- Keep all data loading in one clearly labeled section.
- Do not mix data loading with analysis code.

---

## 3. Exploratory Data Analysis

### Purpose
Explore and understand the data before any modeling. Covers data types, distributions, missing values, outliers, and relationships with survival.

### Proposed Layout

```markdown
## 3. Exploratory Data Analysis

### 3.1 Distinguish Attribute Types
### 3.2 Univariate Analysis
### 3.3 Bivariate and Multivariate Analysis
### 3.4 Detect Interactions Among Attributes
### 3.5 Detect Missing Values
### 3.6 Detect Outliers
### 3.7 Key Findings from EDA
```

### Content To Include

#### 3.1 Distinguish Attribute Types
- Identify categorical columns.
- Identify numerical columns.
- Identify identifier columns to drop.
- Identify the target variable.

#### 3.2 Univariate Analysis
- Target distribution (survival counts and percentages).
- Categorical variable value counts:
  - `Sex`
  - `Pclass`
  - `Embarked`
  - `SibSp`
  - `Parch`
- Numerical variable distributions:
  - `Age` (with KDE)
  - `Fare` (with KDE)
- Include simple visualizations.

#### 3.3 Bivariate and Multivariate Analysis
- Survival rate by `Sex`.
- Survival rate by `Pclass`.
- Survival rate by `Embarked`.
- Survival rate by `Title` (if extracted during EDA).
- Correlation matrix for numerical features.
- Pairwise or grouped survival comparisons.

#### 3.4 Detect Interactions Among Attributes
- `Sex` and `Pclass` interaction with survival.
- `Age` and `Title` interaction.
- `FamilySize` and survival (using `SibSp` and `Parch`).

#### 3.5 Detect Missing Values
- Missing value counts and percentages per column.
- Visual representation of missing values.

#### 3.6 Detect Outliers
- Boxplots for `Age` and `Fare`.
- Note on outlier handling strategy.

#### 3.7 Key Findings from EDA
- Summary of most important signals:
  - `Sex` is the strongest predictor.
  - `Pclass` matters.
  - `Age` has missing values.
  - `Cabin` is mostly missing but `HasCabin` could be useful.
- Note on test set similarity to training set.

### Keep / Change

- Consolidate multiple small EDA cells into the subsections above.
- Keep visualizations focused on insights, not raw data dumps.
- Move test set inspection into a dedicated subsection within EDA.
- Do not perform feature engineering during EDA.

---

## 4. Feature Engineering and Preprocessing

### Purpose
Transform the raw data into a clean, encoded feature set ready for machine learning. All transformations must be fitted on the training portion only.

### Proposed Layout

```markdown
## 4. Feature Engineering and Preprocessing

### 4.1 Combine Train and Test Features
### 4.2 Title Extraction
### 4.3 Family Size Feature
### 4.4 Cabin Availability Feature
### 4.5 Missing Value Imputation
### 4.6 Fare Transformation
### 4.7 Categorical Encoding
### 4.8 Feature Scaling
### 4.9 Final Feature Set
```

### Content To Include

#### 4.1 Combine Train and Test Features
- Store `y_train` separately.
- Combine train and test without the target.
- Keep `n_train` for later splitting.

#### 4.2 Title Extraction
- Extract titles from `Name`.
- Group rare titles into `Rare`.

#### 4.3 Family Size Feature
- Create `FamilySize = SibSp + Parch + 1`.

#### 4.4 Cabin Availability Feature
- Create `HasCabin` binary feature from `Cabin`.

#### 4.5 Missing Value Imputation
- Impute `Age` by `Title` median.
- Impute `Fare` by `Pclass` median.
- Impute `Embarked` with training-set mode.
- Fallback for any remaining missing values.

#### 4.6 Fare Transformation
- Create `FareLog = log(1 + Fare)`.

#### 4.7 Categorical Encoding
- Encode `Sex` as binary.
- Encode `Title` ordinally.
- Map `Pclass` ordinally.
- Create `Embarked_C` binary feature.

#### 4.8 Feature Scaling
- Scale `Age`, `FareLog`, `FamilySize` with `StandardScaler`.
- Fit scaler on training portion only.

#### 4.9 Final Feature Set
- List all final features.
- Show correlation heatmap.
- Split back into `X_train` and `X_test`.

### Keep / Change

- Keep all preprocessing in one place.
- Make sure every transformation states it is fitted on the training portion.
- Do not move baseline into this section.

---

## 5. Baseline Model

### Purpose
Define a simple, domain-driven baseline before training machine learning models. This baseline must use the original unscaled age values.

### Proposed Layout

```markdown
## 5. Baseline Model

### 5.1 Women and Children First Rule
### 5.2 Baseline Metrics
```

### Content To Include

- Explain the historical "Women and Children First" rule.
- Implement the baseline rule: predict survived if `Sex == 'female'` or `Age < 16`.
- Show accuracy, precision, recall, F1.
- Show confusion matrix.

### Important

- Use the original unscaled `Age` values for the baseline rule.
- If `Age` has been scaled already, the comparison is invalid.
- The baseline must run after feature engineering is complete, because it needs imputed ages and encoded sex values.

---

## 6. Model Training and Validation

### Purpose
Train and tune four shallow learning models using 5-fold stratified cross-validation.

### Proposed Layout

```markdown
## 6. Model Training and Validation

### 6.1 Cross-Validation Setup
### 6.2 Logistic Regression
### 6.3 Random Forest
### 6.4 Gradient Boosting
### 6.5 Soft Voting Ensemble
```

### Content To Include

#### 6.1 Cross-Validation Setup
- Define `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- Define scoring dictionary: accuracy, precision, recall, F1.
- Explain why F1 is the main selection metric.

#### 6.2 Logistic Regression
- Define model and hyperparameter grid.
- Run `GridSearchCV`.
- Show best hyperparameters.
- Show cross-validated metrics.
- Show confusion matrix.

#### 6.3 Random Forest
- Define model and hyperparameter grid.
- Run `GridSearchCV`.
- Show best hyperparameters.
- Show cross-validated metrics.
- Show confusion matrix.

#### 6.4 Gradient Boosting
- Define model and hyperparameter grid.
- Run `GridSearchCV`.
- Show best hyperparameters.
- Show cross-validated metrics.
- Show confusion matrix.

#### 6.5 Soft Voting Ensemble
- Combine the tuned Logistic Regression, Random Forest, and Gradient Boosting.
- Use `voting='soft'`.
- Evaluate with the same CV setup.
- Show cross-validated metrics.

### Keep / Change

- Keep all models using the same CV setup for a fair comparison.
- Keep Voting Ensemble as a tested model, not forced as final.
- Keep per-model heatmaps and best parameter printing.

---

## 7. Model Comparison

### Purpose
Compare all models against each other and against the baseline using a single summary table and visualization.

### Proposed Layout

```markdown
## 7. Model Comparison

### 7.1 Results Summary Table
### 7.2 Visual Comparison
### 7.3 Model Selection
```

### Content To Include

#### 7.1 Results Summary Table
- Combine all model metrics in one DataFrame.
- Add delta columns vs baseline.
- Sort by F1 descending.

#### 7.2 Visual Comparison
- Bar chart of all metrics.
- Delta vs baseline bar chart.

#### 7.3 Model Selection
- Automatically select the model with the highest cross-validated F1.
- Show selected model name.

### Keep / Change

- Keep `results['F1'].idxmax()` as the selection mechanism.
- No manual overriding of the final model.

---

## 8. Final Model Evaluation

### Purpose
Perform an in-depth evaluation of the selected model using out-of-fold predictions and feature importance.

### Proposed Layout

```markdown
## 8. Final Model Evaluation

### 8.1 Retrain on Full Training Data
### 8.2 Out-of-Fold Error Analysis
### 8.3 Feature Importance
### 8.4 Evaluation Summary
```

### Content To Include

#### 8.1 Retrain on Full Training Data
- Fit the selected model on all training rows.
- Print confirmation of final model name.

#### 8.2 Out-of-Fold Error Analysis
- Use out-of-fold predictions for honest evaluation.
- Show classification report.
- Show false positives and false negatives.
- Show confusion matrix.

#### 8.3 Feature Importance
- If the model has `feature_importances_`, use that.
- Otherwise use permutation importance.
- Show a bar chart of feature importances.

#### 8.4 Evaluation Summary
- Recap of final model performance.
- Comparison with baseline.
- Key takeaways on model behavior.

### Keep / Change

- Keep OOF-based evaluation to avoid optimistic bias.
- Keep the logic working even if Voting Ensemble is selected.
- Keep feature importance explanation for non-tree models.

---

## 9. Kaggle Submission

### Purpose
Generate test predictions and create a valid Kaggle submission file.

### Proposed Layout

```markdown
## 9. Kaggle Submission

### 9.1 Generate Test Predictions
### 9.2 Build Submission File
### 9.3 Submission Sanity Checks
### 9.4 Save Submission CSV
```

### Content To Include

- Predict on `X_test` using the final model.
- Build DataFrame with `PassengerId` and `Survived`.
- Check: 418 rows.
- Check: columns `PassengerId`, `Survived`.
- Check: all `PassengerId` unique.
- Check: `Survived` contains only `0` and `1`.
- Save as `submission_title_family_hascabin_cv.csv`.

### Keep / Change

- Keep the current filename.
- Keep all sanity checks.
- Keep the output in the working directory.

---

## 10. Sources and AI Tools Used

### Purpose
Disclose all sources and AI tools used in the project.

### Proposed Layout

```markdown
## 10. Sources and AI Tools Used

### Sources Consulted
### AI Tools Used
### Scope Note
```

### Content To Include

- Kaggle Titanic competition documentation.
- Scikit-Learn documentation.
- Machine Learning Fundamentals course material.
- OpenCode / GPT-based assistant disclosure.
- Scope note: shallow learning only, no deep learning.

---

## Section Numbering Map

| Current | Proposed |
|---|---|
| Notebook Setup | 1. Notebook Setup |
| Intro / EDA start | 2. Load Data |
| EDA (full) | 3. Exploratory Data Analysis |
| Feature Engineering | 4. Feature Engineering and Preprocessing |
| Baseline | 5. Baseline Model |
| Model Training | 6. Model Training and Validation |
| Model Comparison | 7. Model Comparison |
| Final Evaluation | 8. Final Model Evaluation |
| Submission | 9. Kaggle Submission |
| Sources | 10. Sources and AI Tools Used |

---

## Implementation Notes

### Why Baseline is After Feature Engineering

The baseline rule `Age < 16` and `Sex == 'female'` requires:
- `Age` to be imputed (done in Feature Engineering step 4.5)
- `Sex` to be encoded (done in Feature Engineering step 4.7)

Running the baseline before these steps would use incomplete or incorrect data. Placing the baseline after Feature Engineering ensures clean, consistent inputs.

### Why EDA is Separate from Feature Engineering

EDA explores the raw data to find patterns, missing values, and relationships. Feature Engineering transforms the data into model inputs. Keeping them separate:

- Prevents leakage from test data into training decisions.
- Makes the notebook easier to follow.
- Makes EDA reusable if preprocessing is changed later.

### Why Automatic Model Selection is Retained

The notebook already uses `results['F1'].idxmax()` to select the final model. This is objective and reproducible. For the assignment, this is preferable to manually overriding the result.

If Voting Ensemble ever scores highest, it will be selected automatically. If Gradient Boosting remains best, it stays selected. The code does not need to be changed for either case.

---

## Minimal Implementation Plan

1. Create the new section headings in the notebook (markdown cells only).
2. Move or duplicate code cells to match the new section structure.
3. Ensure the baseline runs after Feature Engineering is complete.
4. Re-run the notebook end-to-end via `jupyter nbconvert --execute`.
5. Validate the submission CSV has 418 rows, correct columns, unique IDs, and labels only `0/1`.
6. Update `AGENTS.md` if section numbers change.

---

## Larger Optional Cleanup

Only do this if the minimal plan is complete and time permits:

- Combine small adjacent markdown cells into single coherent cells.
- Remove redundant plot cells.
- Add a final summary text cell with the best model name and F1 score.
- Ensure every code cell has a short comment explaining its purpose.