Good day. My name is Tim Kindt, and welcome to my presentation on the Titanic Survival Prediction project.

The sinking of the RMS Titanic in 1912 is one of history's most significant maritime disasters. In this project, I use data science to analyze passenger records and build a machine learning model that predicts survival.

This notebook follows a complete, end-to-end workflow:
Data Loading and Exploratory Data Analysis.
Feature Engineering and Preprocessing.
Baseline Modeling.
Model Training and Validation.
And Final Evaluation and Kaggle Submission.

My approach focuses on methodological soundness. I use interpretable features, rigorous cross-validation, and a strong baseline to ensure the results are reliable. All models used here are shallow learning models implemented with Scikit-Learn.

I start by loading the training dataset, which contains 891 passengers, and the test dataset with 418 passengers.

During the Exploratory Data Analysis, I examine distributions, missing values, and correlations. Several key patterns emerge:

Sex is the strongest predictor: female passengers had a significantly higher survival rate.
Passenger Class matters: first-class passengers survived more often than third-class passengers.
Age has missing values, requiring careful imputation.
Fare is heavily right-skewed, meaning a few passengers paid very high prices.
Cabin is missing for most passengers, so I drop the raw text but use derived features instead.

I also check the test set to ensure its distributions match the training set, confirming that our preprocessing strategy will transfer well.

Next, I engineer a compact set of six features to feed into the models. I combine train and test data to ensure consistent preprocessing.

Title: I extract titles like Mr, Mrs, Miss, and Master from names. Rare titles are grouped into a Rare category. This captures social role and gender information effectively.
GroupSize: Instead of just counting family members, I count how many passengers share the same ticket. This captures travel groups, which likely evacuated together.
Age: I impute missing ages by sampling from the age distribution of each Title group. This preserves natural variance better than a global median. A fixed random seed ensures reproducibility.
FareLog: I fill missing fares with the median per class and apply a log transformation to handle the skewness.
Pclass and Sex are encoded numerically.

After scaling the continuous features, I end up with six clean features: Pclass, Sex, Age, Title, GroupSize, and FareLog.

A note on multicollinearity: Title and Sex are highly correlated, which is expected since titles encode gender. However, tree-based models handle this well, and Title adds nuance beyond just gender.

Before training complex models, I establish a domain-driven baseline: the Women and Children First rule.

This rule predicts survival for all females and children under 14. It achieves an F1 score of 0.735 and an accuracy of roughly 79 percent.

This is a strong benchmark. Any machine learning model must clearly outperform this to justify its complexity. I ensure the baseline uses the original, unscaled Age values so the rule remains meaningful.

I train three models using 5-fold stratified cross-validation:
Logistic Regression.
Random Forest.
And Gradient Boosting.

I also evaluate a Soft Voting Ensemble combining all three. Hyperparameters are tuned using GridSearchCV.

Here are the cross-validated results:
Logistic Regression scores an F1 of 0.752.
Random Forest improves to 0.791.
Gradient Boosting achieves the highest score with an F1 of 0.814.
The Voting Ensemble scores 0.806, which is good but does not beat Gradient Boosting.

Gradient Boosting improves over the baseline by nearly 0.08 points in F1 score. Therefore, I select Gradient Boosting as the final model.

I retrain the Gradient Boosting model on the full training set. No further tuning is done to avoid overfitting; the hyperparameters from cross-validation are fixed.

For error analysis, I use out-of-fold predictions. The confusion matrix shows the model is slightly conservative, with slightly more false negatives than false positives.

The feature importance plot confirms our EDA findings:
Title is the most important feature.
FareLog and Pclass follow closely.
GroupSize and Age also contribute signal.

This confirms the model relies on meaningful, interpretable signals.

Finally, I generate predictions for the 418 passengers in the test set.

The submission file passes all sanity checks:
418 rows.
Correct columns: PassengerId and Survived.
Unique IDs and binary labels.

To summarize:
I built a robust machine learning pipeline that outperforms a strong historical baseline. By focusing on careful feature engineering and rigorous validation, I selected a Gradient Boosting model that achieves a cross-validated F1 score of 0.814.

Thank you for watching. I am happy to answer any questions.
