# Word-for-Word Presentation Script: Titanic Survival Prediction

Target length: 10 to 15 minutes  
Notebook: `Titanic_Disaster_Challenge-Final.ipynb`

## 1. Introduction

Hello, my name is Tim Kindt, a student in Applied Computer Science, specializing in Artificial Intelligence at VIVES University Kortrijk. This presentation walks through my Titanic survival prediction project.

The goal of this project is to predict whether a passenger survived the Titanic disaster, based on passenger information such as class, sex, age, fare, family size, cabin information, and embarkation port.

This is a supervised binary classification problem. It is supervised because the training dataset contains the correct answer, which is the `Survived` column. It is binary because the target has only two possible values: `0`, meaning the passenger did not survive, and `1`, meaning the passenger survived.

This notebook follows a full machine learning workflow. It starts with data loading and exploratory data analysis. Then the data is cleaned and useful features are engineered. After that, several shallow machine learning models are trained and compared. Finally, the best model is selected using cross-validation and a Kaggle-ready submission file is created.

The tools used are Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn. This project does not use deep learning or neural networks. All models are shallow learning models, which fits the scope of the assignment.

## 2. Data Loading And Exploratory Data Analysis

The first step is loading the training dataset from `Datasets/train.csv`.

The training dataset contains 891 passengers. The most important column is `Survived`, because that is the target variable we want to predict.

The test dataset is loaded from `Datasets/test.csv`. This test dataset contains 418 passengers, but it does not include the `Survived` column. That is the column that needs to be predicted for the Kaggle submission.

After loading the data, the notebook inspects the shape of the dataset, the column types, the first few rows, missing values, and summary statistics.

The main goal of exploratory data analysis is to understand the dataset before building a model. The important questions are which columns are useful, which values are missing, and which features seem related to survival.

One of the clearest findings is that sex is a very strong signal. Female passengers had a much higher survival rate than male passengers.

Passenger class is also important. First-class passengers had a higher survival rate than third-class passengers. This makes sense historically, because passenger class is related to socio-economic status and probably also to cabin location and access to lifeboats.

Age is also useful, but it contains missing values. This means it cannot be used directly without imputation.

The `Cabin` column has many missing values. Instead of using the raw cabin text, this is later converted into a simpler feature that only records whether a cabin was known or not.

The `Fare` column is skewed. Most passengers paid lower fares, while a few passengers paid very high fares. Because of this, a log transformation is later applied to make the distribution less extreme.

So, after the EDA, the most promising signals are sex, passenger class, age, fare, family information, and cabin availability.

## 3. Feature Engineering

Next comes feature engineering.

Feature engineering means transforming the raw dataset into a set of features that are easier for machine learning models to use.

First, the training and test datasets are combined into one dataset. Before doing that, the target variable `Survived` is separated, so the target is never used during preprocessing. Combining the data helps apply the same transformations consistently to both training and test passengers.

The first engineered feature is `Title`. Titles are extracted from passenger names, such as `Mr`, `Mrs`, `Miss`, and `Master`.

The full name is not used as raw text, because names are too specific and would not generalize well. Titles are more useful because they contain social and demographic information.

Some titles are rare, such as `Dr`, `Rev`, `Col`, or `Countess`. These are grouped into one `Rare` category. This keeps the feature compact and avoids creating categories with only one or two passengers.

Next, `FamilySize` is created. This feature combines `SibSp`, which is siblings and spouses, and `Parch`, which is parents and children. One is added to include the passenger themself.

So the formula is: `FamilySize = SibSp + Parch + 1`.

This feature is useful because survival may depend on whether someone travelled alone, with a small family, or in a large family group.

Then `HasCabin` is created. The raw `Cabin` column has too many missing values to use directly. But whether a passenger had a recorded cabin can still be useful. So this becomes a binary feature: `1` if the cabin is known, and `0` if it is missing.

For missing ages, grouped imputation by title is used. This means the median age is calculated for each title group and used to fill missing ages within the same title group.

This is better than filling all missing ages with one global median, because titles give useful age-related information. For example, `Master` usually refers to a boy, while `Mr` usually refers to an adult man.

For missing fares, the median fare per passenger class is used. This is more reasonable than using one global fare median, because fares depend strongly on passenger class.

After filling missing fares, `FareLog` is created by applying a log transformation. This reduces the effect of very high fare values and makes the feature less skewed.

For `Embarked`, missing values are first filled with the most common embarkation port in the training data. Then a binary feature called `Embarked_C` is created, which indicates whether the passenger embarked from Cherbourg.

After that, categorical features are encoded into numbers. Machine learning models cannot directly use text categories, so they need numeric input.

`Sex` is encoded as a binary feature. `Title` is encoded using an ordinal encoder. `Pclass` is also mapped manually, so first class gets the highest value and third class gets the lowest value.

Finally, the continuous features are scaled: `Age`, `FareLog`, and `FamilySize`. Scaling is especially useful for Logistic Regression, because that model is sensitive to the scale of input features.

The final feature set contains `Pclass`, `Title`, `Age`, `FareLog`, `FamilySize`, `HasCabin`, `Sex`, and `Embarked_C`.

This feature set is intentionally compact. It includes strong and interpretable signals, without making the notebook too complex.

## 4. Baseline Model

Before training machine learning models, a simple baseline is created.

The baseline is based on the historical rule: women and children first.

The rule predicts survival for female passengers and for passengers younger than 16. Everyone else is predicted as not surviving.

This is a useful baseline because it is simple, understandable, and based on domain knowledge.

The baseline accuracy is `0.789`, and the baseline F1 score is `0.731`.

This means that any machine learning model should beat this baseline to justify the added complexity.

One important detail is that the baseline must use the original age values. If age has already been scaled, then the rule `Age < 16` no longer means younger than 16 years old. The notebook therefore calculates the baseline before scaling age.

## 5. Model Training And Cross-Validation

After the baseline, several machine learning models are trained.

The notebook uses 5-fold stratified cross-validation. This means the training data is split into five folds. The model trains on four folds and validates on the remaining fold. This process is repeated five times, so every fold is used once as validation data.

Stratified cross-validation is used because the class distribution should remain similar in every fold. In other words, each fold should have a similar ratio of survivors and non-survivors.

Cross-validation is important because the dataset is small. It gives a more stable estimate than using only one train-validation split. It also avoids relying only on the Kaggle public leaderboard.

The evaluation metrics are accuracy, precision, recall, and F1 score.

Accuracy tells us the overall percentage of correct predictions.

Precision tells us: when the model predicts survival, how often is it correct?

Recall tells us: of all passengers who actually survived, how many did the model identify?

F1 score combines precision and recall into one balanced metric.

F1 score is used as the main selection metric because the data is slightly imbalanced, and the goal is to balance precision and recall.

The first model is Logistic Regression. Logistic Regression is simple, fast, and interpretable. It is a good first machine learning model for classification.

The second model is Random Forest. Random Forest is an ensemble of decision trees. It can capture non-linear relationships and interactions between features.

The third model is Gradient Boosting. Gradient Boosting also uses decision trees, but it builds them sequentially. Each new tree tries to correct mistakes made by the previous trees.

For each model, `GridSearchCV` tests a small set of hyperparameters. The grids are intentionally not too large, because this is a small dataset and the assignment should remain understandable.

A soft Voting Ensemble is also added. This combines the tuned Logistic Regression, Random Forest, and Gradient Boosting models.

Soft voting means that the ensemble averages the predicted probabilities from the individual models. It does not simply count the final class labels. This can sometimes make predictions more stable.

However, the Voting Ensemble is not automatically chosen. It is evaluated in exactly the same way as the other models, using the same cross-validation setup. It is only selected if it has the best F1 score.

## 6. Model Results

The model comparison results are as follows.

The baseline has an accuracy of `0.789` and an F1 score of `0.731`.

Logistic Regression improves on the baseline, with an accuracy of `0.819` and an F1 score of `0.757`.

Random Forest performs better, with an accuracy of `0.834` and an F1 score of `0.771`.

Gradient Boosting performs best, with an accuracy of `0.847` and an F1 score of `0.792`.

The Voting Ensemble also performs well, with an accuracy of `0.836` and an F1 score of `0.778`, but it does not beat Gradient Boosting.

So the final selected model is Gradient Boosting.

Compared with the baseline, Gradient Boosting improves accuracy by about `0.058` and F1 score by about `0.060`.

This confirms that the final model performs better than the simple women-and-children-first baseline.

## 7. Final Model Evaluation

After selecting Gradient Boosting, it is trained one final time on the full training dataset.

This is useful because once the model has been selected, it should learn from all available labeled data before making predictions for the Kaggle test set.

For error analysis, the notebook uses out-of-fold predictions from cross-validation. This is more honest than only checking predictions on the same data the model was trained on.

The notebook shows a classification report, false positives, false negatives, and a confusion matrix.

The confusion matrix helps show what kinds of mistakes the model makes. It separates correct non-survival predictions, correct survival predictions, false positives, and false negatives.

Feature importance is also inspected. Since Gradient Boosting is tree-based, it can provide built-in feature importance values.

Feature importance does not prove causation, but it shows which features the model relied on most when making predictions.

This helps keep the final model explainable, even though Gradient Boosting is more complex than Logistic Regression.

## 8. Final Prediction And Kaggle Submission

In the final section, the selected Gradient Boosting model predicts survival for the Kaggle test dataset.

The output must follow the Kaggle submission format. It needs exactly two columns: `PassengerId` and `Survived`.

The notebook creates a submission dataframe with those two columns and saves it as `submission_title_family_hascabin_cv.csv`.

The submission file is also validated.

The first check confirms that it has 418 rows, because the test set has 418 passengers.

The second check confirms that the columns are correct.

The third check confirms that every `PassengerId` is unique.

The final check confirms that the predictions only contain `0` and `1`.

These checks confirm that the file is ready to upload to Kaggle.

## 9. Sources And AI Tools

At the end of the notebook, the sources and AI tools used are included.

The sources include the Kaggle Titanic competition documentation, Scikit-Learn documentation, and course material from Machine Learning Fundamentals.

The notebook also mentions that OpenCode, a GPT-based AI assistant, was used to help review the notebook structure, check assignment compliance, and suggest safe refactoring steps.

The modeling scope is clearly limited to shallow learning models. The notebook uses Logistic Regression, Random Forest, Gradient Boosting, and a soft Voting Ensemble built from those models. No neural networks or deep learning models are used.

## 10. Conclusion

To conclude, this notebook implements a complete machine learning workflow for the Titanic survival prediction problem.

The workflow starts with exploratory data analysis, then cleans and transforms the data, engineers interpretable features, trains several shallow machine learning models, compares them with stratified cross-validation, and creates a valid Kaggle submission.

The final model is Gradient Boosting because it achieves the highest cross-validated F1 score.

It also clearly improves over the simple women-and-children-first baseline.

The Voting Ensemble was tested as an additional model, but it did not outperform Gradient Boosting, so it was not selected as the final model.

Overall, the final solution is accurate, explainable, and aligned with the assignment requirements.

Thank you for watching.
