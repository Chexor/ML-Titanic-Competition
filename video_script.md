# Video Script: Titanic Survival Prediction

Target length: 10-15 minutes  
Notebook: `Titanic_Disaster_Challenge-Final.ipynb`

## 0. Before Recording

- Open `Titanic_Disaster_Challenge-Final.ipynb`.
- Make sure the notebook has been run from top to bottom.
- Keep the Kaggle submission file visible if you want to show the final output: `submission_title_family_hascabin_cv.csv`.
- Keep this script next to the notebook and use it as a speaking guide, not as text to read word-for-word.

---

## 1. Introduction

Estimated time: 1 minute

Hello, my name is Tim. In this video I will walk through my Titanic survival prediction project.

The goal of this assignment is to build a shallow machine learning model that predicts whether passengers survived the Titanic disaster. I use the classic Kaggle Titanic dataset, which contains passenger information such as class, sex, age, fare, family relations, cabin information, and the target variable `Survived`.

In this notebook I follow the full machine learning workflow:

- data preparation and exploratory data analysis
- feature engineering
- model training and hyperparameter tuning
- model evaluation and selection
- final prediction and Kaggle submission

I use Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn. I do not use deep learning; all models are shallow learning models.

---

## 2. Data Loading And EDA

Estimated time: 2 minutes

First, I load the training dataset from `Datasets/train.csv`.

The training set contains 891 passengers and the target column is `Survived`, where `1` means survived and `0` means did not survive.

During the initial exploration, I check:

- the shape of the dataset
- the column types
- missing values
- basic descriptive statistics
- survival distribution
- relationships between survival and important features

The main EDA findings are:

- `Sex` is a very strong signal: female passengers had a much higher survival rate.
- `Pclass` also matters: first-class passengers had better survival chances than third-class passengers.
- `Age` has missing values, so it needs imputation.
- `Cabin` has many missing values, so using the raw cabin text directly would not be reliable.
- `Fare` is strongly skewed, so a log transformation is useful.

I also inspect the test dataset from `Datasets/test.csv`. It has 418 passengers and does not contain the `Survived` column, because that is what we need to predict for Kaggle.

---

## 3. Feature Engineering

Estimated time: 3 minutes

In the feature engineering section, I build a compact and interpretable feature set.

First, I combine the train and test datasets after separating the target variable. This keeps preprocessing consistent between training and test data. The target `Survived` is not included in this combined dataset, so it is not used during feature engineering.

Then I extract passenger titles from the `Name` column. Instead of using names as raw text, I extract titles like `Mr`, `Mrs`, `Miss`, and `Master`. Rare titles are grouped into a broader `Rare` category. This gives the model useful social information without making the feature too sparse.

Next, I create `FamilySize` by combining `SibSp`, `Parch`, and the passenger themself. This captures whether someone travelled alone, with a small family, or in a larger group.

For `Cabin`, I use a binary `HasCabin` feature. The raw cabin column has too many missing values, but whether a cabin was recorded can still be informative.

For missing ages, I use grouped imputation by passenger title. This is better than using one global median, because different title groups often have different age distributions.

For `Fare`, I fill missing values using the median fare per passenger class, then create `FareLog` with a log transform. This reduces the impact of extreme fare values.

For categorical encoding:

- `Pclass` is mapped manually so first class has the highest encoded value.
- `Title` is ordinally encoded.
- `Sex` is converted into a binary feature.
- Missing `Embarked` values are filled with the training-set mode.
- `Embarked_C` is created as a binary feature for passengers who embarked from Cherbourg.

Finally, I scale the continuous features `Age`, `FareLog`, and `FamilySize`. Scaling is mainly useful for Logistic Regression.

The final feature set is:

- `Pclass`
- `Title`
- `Age`
- `FareLog`
- `FamilySize`
- `HasCabin`
- `Sex`
- `Embarked_C`

---

## 4. Baseline Model

Estimated time: 1 minute

Before training machine learning models, I define a simple domain baseline: women and children first.

The baseline predicts survival for:

- female passengers
- passengers younger than 14

Everyone else is predicted as not surviving.

This gives a meaningful benchmark because it is based on known historical evacuation patterns.

The baseline results are:

- Accuracy: `0.789`
- F1 score: `0.731`

This means any final model should improve on this benchmark to be useful.

One important implementation detail is that the baseline is calculated before scaling `Age`. This keeps the rule `Age < 14` meaningful and avoids comparing against an incorrectly calculated baseline.

---

## 5. Model Training And Validation

Estimated time: 3 minutes

In the model training section, I use 5-fold stratified cross-validation. This gives a more reliable estimate than only checking the Kaggle public leaderboard.

I evaluate every model with the same metrics:

- accuracy
- precision
- recall
- F1 score

I use F1 score as the main selection metric because the dataset is slightly imbalanced and F1 balances precision and recall.

The individual models are:

- Logistic Regression
- Random Forest
- Gradient Boosting

For each model, I use `GridSearchCV` to tune a small set of hyperparameters. I keep the grids small enough to stay understandable for the assignment.

I also add a soft Voting Ensemble. This combines the tuned Logistic Regression, Random Forest, and Gradient Boosting models. Soft voting averages the predicted class probabilities instead of only counting class labels.

The Voting Ensemble is not automatically selected. It is evaluated with the same cross-validation setup as the other models, and it is only selected if it achieves the best F1 score.

The cross-validation results are:

| Model | Accuracy | F1 |
|---|---:|---:|
| Baseline | `0.789` | `0.731` |
| Logistic Regression | `0.819` | `0.757` |
| Random Forest | `0.834` | `0.771` |
| Gradient Boosting | `0.847` | `0.792` |
| Voting Ensemble | `0.836` | `0.778` |

The best model is Gradient Boosting, with:

- Accuracy: `0.847`
- F1 score: `0.792`

Compared with the baseline, Gradient Boosting improves:

- Accuracy by about `0.058`
- F1 score by about `0.060`

The Voting Ensemble performs well, but it does not beat Gradient Boosting. Therefore, the final model remains Gradient Boosting.

---

## 6. Final Model Evaluation

Estimated time: 2 minutes

After selecting the best model, I retrain it on the full training dataset. This gives the final model access to all available labeled data before predicting the Kaggle test set.

For error analysis, I use out-of-fold predictions from cross-validation instead of evaluating only on the same data the model was trained on. This gives a more honest view of typical model errors.

The notebook shows:

- a classification report
- false positives
- false negatives
- a confusion matrix
- feature importance

Feature importance helps explain which engineered variables were most useful for the model. For tree-based models such as Gradient Boosting, built-in feature importances can be used directly.

This section confirms that the selected model improves clearly over the baseline while still remaining interpretable enough for the assignment.

---

## 7. Final Prediction And Submission

Estimated time: 1 minute

In the final section, I use the selected Gradient Boosting model to predict survival for the Kaggle test set.

The submission file contains exactly the required Kaggle format:

- `PassengerId`
- `Survived`

The file is saved as:

`submission_title_family_hascabin_cv.csv`

I also check that:

- the submission has 418 rows
- the columns are correct
- `PassengerId` values are unique
- predicted labels are only `0` or `1`

This confirms that the file is ready for Kaggle submission.

---

## 8. Sources And AI Tools

Estimated time: 30 seconds

At the end of the notebook, I include the sources and tools used.

The main sources were:

- the Kaggle Titanic competition documentation
- Scikit-Learn documentation
- course material from Machine Learning Fundamentals

I also used OpenCode, a GPT-based AI assistant, to help review notebook structure, check assignment compliance, and suggest safe refactoring steps.

The modeling scope is limited to shallow learning models. No neural networks or deep learning models were used.

---

## 9. Closing

Estimated time: 30 seconds

To summarize, this project follows a complete machine learning workflow from EDA to Kaggle submission.

The final model is Gradient Boosting. It beats the domain baseline and also outperforms Logistic Regression, Random Forest, and the soft Voting Ensemble under the same 5-fold cross-validation setup.

The final result is both technically valid and explainable, which makes it suitable for this assignment.

Thank you for watching.
