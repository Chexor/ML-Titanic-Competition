# Titanic Notebook Cheatsheet

Use this as a preparation guide for explaining and defending `Titanic_Disaster_Challenge-Final.ipynb`.

## 1. Big Picture

The goal is to predict whether a Titanic passenger survived.

This is a supervised binary classification problem:

- supervised: the training data includes the correct answer, `Survived`
- binary: there are only two possible classes, `0` or `1`
- classification: we predict a category, not a continuous number

The full workflow is:

- load and inspect the data
- understand patterns with EDA
- clean missing values
- engineer useful features
- train several shallow ML models
- compare them with cross-validation
- choose the best model by F1 score
- generate a Kaggle submission

The final selected model is `Gradient Boosting` because it has the best cross-validated F1 score.

## 2. Dataset

The training data has 891 passengers.

Important columns:

- `PassengerId`: unique identifier, useful for submission but not for learning
- `Survived`: target variable, `1` survived and `0` did not survive
- `Pclass`: passenger class, proxy for socio-economic status
- `Name`: used to extract title, not used directly
- `Sex`: very strong survival signal
- `Age`: useful but has missing values
- `SibSp`: siblings/spouses aboard
- `Parch`: parents/children aboard
- `Ticket`: dropped, too messy for this notebook
- `Fare`: ticket price, skewed distribution
- `Cabin`: many missing values, converted to `HasCabin`
- `Embarked`: port of embarkation

The test data has 418 passengers and no `Survived` column. We predict that column for Kaggle.

## 3. Exploratory Data Analysis

EDA means exploring the data before modeling.

The purpose is to understand:

- what columns exist
- which columns are numerical or categorical
- where missing values are
- whether the target is balanced
- which features seem related to survival

Key findings:

- Women survived much more often than men.
- First-class passengers had better survival chances than third-class passengers.
- Children had different survival patterns than adults.
- `Age`, `Cabin`, and some test-set `Fare` values need missing-value handling.
- `Fare` is skewed: most people paid low fares, a few paid very high fares.

Intuition:

EDA is like looking at the map before choosing a route. It does not train the model yet, but it tells us what problems and patterns to expect.

## 4. Missing Values

Missing values matter because most Scikit-Learn models cannot handle `NaN` directly.

### Age

We impute missing `Age` values using the median age per `Title`.

Why not just use the global median?

Because titles contain age-related information:

- `Master` often means a boy
- `Miss` often means a younger female passenger
- `Mrs` usually indicates an adult woman
- `Mr` usually indicates an adult man

Intuition:

If we do not know someone's age, their title gives a reasonable clue.

### Fare

Missing `Fare` values are filled using the median fare per `Pclass`.

Why?

Ticket price strongly depends on passenger class. A missing first-class fare should not be filled with the same value as a third-class fare.

### Embarked

Missing `Embarked` values are filled with the most common embarkation port from the training set.

Why?

There are only a few missing values, so the mode is a simple and defensible choice.

### Cabin

The raw `Cabin` column has too many missing values.

Instead of trying to recover exact cabins, we use:

- `HasCabin = 1` if a cabin is recorded
- `HasCabin = 0` otherwise

Intuition:

The exact cabin may be sparse, but having a recorded cabin can still indicate class/status or data availability.

## 5. Feature Engineering

Feature engineering means creating better input variables for the model.

Good features make patterns easier for the model to learn.

### Title

We extract titles from names, such as:

- `Mr`
- `Mrs`
- `Miss`
- `Master`
- `Rare`

Rare titles are grouped together to avoid tiny categories.

Why?

Titles contain social and demographic information. They are more useful than raw names.

### FamilySize

Created as:

```text
FamilySize = SibSp + Parch + 1
```

The `+1` represents the passenger themself.

Why?

Travelling alone or with family may affect survival chances.

Intuition:

Someone with a small family might have help or priority, while very large families may have had more difficulty staying together.

### HasCabin

Created from `Cabin` availability.

Why?

Raw cabin strings are messy and mostly missing, but cabin availability itself is informative.

### FareLog

Created with:

```text
FareLog = log(1 + Fare)
```

Why use log?

Fare is skewed. A few very high fares can dominate the scale. Log transformation compresses extreme values.

Intuition:

The difference between 10 and 20 is more meaningful than the difference between 300 and 310. Log transformation reflects that better.

### Embarked_C

Created as a binary feature:

- `1` if embarked from Cherbourg
- `0` otherwise

Why only Cherbourg?

It keeps the feature set compact and isolates one potentially useful embarkation signal.

## 6. Encoding

Machine learning models need numbers, not raw text.

Encoding converts categories into numeric representations.

Examples:

- `Sex`: female/male becomes `0`/`1`
- `Title`: title categories become ordered numeric codes
- `Embarked_C`: already binary
- `Pclass`: manually mapped so first class has the highest value

Important:

Encoding does not mean the model understands language. It only gives the algorithm numeric inputs.

## 7. Scaling

Continuous features are scaled:

- `Age`
- `FareLog`
- `FamilySize`

Scaling means transforming values so they have a comparable range.

Why?

Logistic Regression can be sensitive to feature scale. Tree-based models are less sensitive, but using one consistent feature table keeps the workflow simple.

Important baseline detail:

The `Women and Children First` baseline must use the original age scale. If `Age` is scaled, the rule `Age < 14` no longer means younger than 14 years.

## 8. Baseline Model

The baseline is a simple rule:

```text
Predict survived if passenger is female or younger than 14.
```

This is called `Women and Children First`.

Why use a baseline?

A baseline answers: does machine learning actually improve on a simple, sensible rule?

Baseline result:

- Accuracy: `0.789`
- F1: `0.731`

If our final model cannot beat this, the extra complexity is not justified.

## 9. Train/Test Split In Kaggle Context

Kaggle gives:

- training data with labels
- test data without labels

We cannot evaluate directly on Kaggle test labels because we do not have them.

Therefore, we use cross-validation on the training set to estimate performance.

The final model is then trained on all training data and used to predict the Kaggle test set.

## 10. Cross-Validation

The notebook uses 5-fold stratified cross-validation.

How it works:

- split the training data into 5 parts
- train on 4 parts
- validate on the remaining part
- repeat until every part has been used as validation once
- average the scores

Stratified means each fold keeps roughly the same survival/non-survival ratio.

Why use it?

It gives a more reliable estimate than one random split, especially with a small dataset.

Intuition:

Instead of judging the model from one exam, cross-validation gives it five smaller exams and averages the result.

## 11. Metrics

### Accuracy

Accuracy is the percentage of correct predictions.

Formula intuition:

```text
correct predictions / all predictions
```

Good for a quick overview, but can be misleading if classes are imbalanced.

### Precision

Precision answers:

When the model predicts survived, how often is it correct?

High precision means fewer false positives.

### Recall

Recall answers:

Of all passengers who actually survived, how many did the model find?

High recall means fewer false negatives.

### F1 Score

F1 combines precision and recall.

Why use F1?

It balances false positives and false negatives better than accuracy alone.

Main selection metric in the notebook:

```text
best_model_name = results['F1'].idxmax()
```

So the final model is chosen by cross-validated F1 score.

## 12. Confusion Matrix

A confusion matrix shows prediction types:

- true negative: predicted did not survive, actually did not survive
- false positive: predicted survived, actually did not survive
- false negative: predicted did not survive, actually survived
- true positive: predicted survived, actually survived

Intuition:

It shows not just how many mistakes the model makes, but what kind of mistakes.

## 13. Models

### Logistic Regression

Despite the name, Logistic Regression is a classification model.

Intuition:

It learns a weighted combination of features and converts that into a survival probability.

Strengths:

- simple
- fast
- interpretable
- good baseline ML model

Weaknesses:

- mostly linear
- may miss complex interactions

### Random Forest

Random Forest is an ensemble of decision trees.

Intuition:

Many trees vote together. Each tree sees patterns in a slightly different way, and the forest combines them.

Strengths:

- handles non-linear patterns
- captures feature interactions
- robust on tabular data

Weaknesses:

- less interpretable than Logistic Regression
- can overfit if not controlled

### Gradient Boosting

Gradient Boosting builds trees sequentially.

Intuition:

Each new tree tries to correct mistakes made by the previous trees.

Strengths:

- strong performance on structured/tabular data
- captures non-linear patterns
- often performs well with careful tuning

Weaknesses:

- can overfit if too complex
- less intuitive than simple models

In this notebook, Gradient Boosting performs best.

### Voting Ensemble

The Voting Ensemble combines:

- Logistic Regression
- Random Forest
- Gradient Boosting

It uses soft voting.

Soft voting means:

```text
average the predicted probabilities from all models
```

Example intuition:

- Logistic Regression says survival probability is 0.70
- Random Forest says 0.60
- Gradient Boosting says 0.80
- Soft voting averages these into about 0.70

Why include it?

Sometimes combining models gives more stable predictions.

What happened here?

Voting Ensemble performed well, but Gradient Boosting still had the best F1 score. So Voting is included as a tested model, but not selected as final.

## 14. Hyperparameter Tuning

Hyperparameters are model settings chosen before training.

Examples:

- number of trees
- maximum tree depth
- learning rate
- regularization strength

The notebook uses `GridSearchCV`.

Grid search means:

- define a small list of possible settings
- try all combinations
- evaluate them with cross-validation
- keep the best combination

Why small grids?

Because the assignment should stay understandable, and a small dataset can overfit if we search too aggressively.

## 15. Final Model Selection

Model comparison results:

| Model | Accuracy | F1 |
|---|---:|---:|
| Baseline | `0.789` | `0.731` |
| Logistic Regression | `0.819` | `0.757` |
| Random Forest | `0.834` | `0.771` |
| Gradient Boosting | `0.847` | `0.792` |
| Voting Ensemble | `0.836` | `0.778` |

Final model:

```text
Gradient Boosting
```

Why?

It has the highest cross-validated F1 score and also improves clearly over the baseline.

Improvement over baseline:

- Accuracy: about `+0.058`
- F1: about `+0.060`

## 16. Out-Of-Fold Predictions

Out-of-fold predictions are predictions made for validation folds during cross-validation.

Why use them for error analysis?

Because each prediction comes from a model that did not train on that specific row.

This is more honest than evaluating only on the same data used for training.

Intuition:

It is closer to asking: how does the model behave on passengers it has not seen before?

## 17. Feature Importance

Feature importance helps explain which inputs the model used most.

For Gradient Boosting, feature importance comes from the trained tree ensemble.

It does not prove causation.

It means:

The model relied more on these features when making splits and predictions.

Likely important features in this problem:

- `Sex`
- `Title`
- `Pclass`
- `FareLog`
- `Age`
- `HasCabin`

## 18. Final Kaggle Submission

The final model predicts survival for the 418 passengers in the test set.

The submission file must contain exactly:

- `PassengerId`
- `Survived`

The notebook saves:

```text
submission_title_family_hascabin_cv.csv
```

Validation checks:

- 418 rows
- correct column names
- unique passenger IDs
- predictions only contain `0` and `1`

## 19. Sources And AI Tools

The notebook lists sources and AI usage.

Sources:

- Kaggle Titanic competition documentation
- Scikit-Learn documentation
- Machine Learning Fundamentals course material

AI tool:

- OpenCode / GPT-based assistant was used for reviewing structure, assignment compliance, and safe refactoring suggestions

Important:

The notebook still represents a shallow-learning workflow. No neural networks or deep learning are used.

## 20. Things To Be Ready To Explain

### Why not use `Name` directly?

Raw names are high-cardinality text and usually do not generalize. Titles extracted from names are cleaner and more meaningful.

### Why not use raw `Cabin`?

Too many missing values and too many sparse cabin labels. `HasCabin` keeps the useful signal in a simple way.

### Why use cross-validation?

The dataset is small. Cross-validation gives a more stable estimate than one split and avoids relying only on Kaggle leaderboard feedback.

### Why choose F1 instead of accuracy?

Accuracy is useful, but F1 balances precision and recall. This is better when both false positives and false negatives matter.

### Why did Voting not become final?

It was evaluated fairly, but it did not have the highest F1 score. Gradient Boosting performed better.

### Why no deep learning?

The assignment focuses on shallow learning. Also, Titanic is a small tabular dataset where deep learning is unnecessary.

### Is the model perfect?

No. It is a strong, interpretable assignment solution, but there are always possible improvements such as more advanced pipelines, more feature engineering, or external validation.

## 21. Short Defense Summary

If asked to summarize the project quickly:

This notebook builds a complete shallow machine learning pipeline for the Titanic Kaggle challenge. I start with EDA, handle missing values, engineer interpretable features like `Title`, `FamilySize`, `HasCabin`, and `FareLog`, then compare Logistic Regression, Random Forest, Gradient Boosting, and a soft Voting Ensemble using 5-fold stratified cross-validation. The final model is selected by F1 score, not by guessing or leaderboard tuning. Gradient Boosting performs best, improving over the `Women and Children First` baseline, and the notebook produces a valid Kaggle submission file.
