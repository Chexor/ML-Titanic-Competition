Good day. My name is Tim Kindt, and welcome to this presentation on the Titanic Survival Prediction project.

The sinking of the RMS Titanic in 1912 is one of history's most significant maritime disasters. In this project, data science is used to analyze passenger records and build a machine learning model that predicts survival.

The notebook follows a complete, end-to-end workflow:
Data Loading and Exploratory Data Analysis.
Feature Engineering and Preprocessing.
Baseline Modeling.
Model Training and Validation.
And Final Evaluation and Kaggle Submission.

The approach focuses on methodological soundness. Interpretable features, rigorous cross-validation, and a strong baseline are used to ensure the results are reliable. All models used here are shallow learning models implemented with Scikit-Learn.

The process begins by loading the training dataset, which contains 891 passengers, and the test dataset with 418 passengers.

During the Exploratory Data Analysis, distributions, missing values, and correlations are examined. Several key patterns emerge:

Sex is the strongest predictor: female passengers had a significantly higher survival rate.
Passenger Class matters: first-class passengers survived more often than third-class passengers.
Age has missing values, requiring careful imputation.
Fare is heavily right-skewed, meaning a few passengers paid very high prices.
Cabin is missing for most passengers, so the raw text is dropped but derived features can be considered if needed.

The test set is also checked to ensure its distributions match the training set, confirming that the preprocessing strategy will transfer well.

Next, a compact set of six features is engineered to feed into the models. Train and test data are combined to ensure consistent preprocessing.

Title: Titles like Mr, Mrs, Miss, and Master are extracted from names. Rare titles are grouped into a Rare category. This captures social role and gender information effectively.
GroupSize: Instead of just counting family members, the number of passengers sharing the same ticket is counted. This captures travel groups, which likely evacuated together.
Age: Missing ages are imputed by sampling from the age distribution of each Title group. This preserves natural variance better than a global median. A fixed random seed ensures reproducibility.
FareLog: Missing fares are filled with the median per class, and a log transformation is applied to handle the skewness.
Pclass and Sex are encoded numerically.

After scaling the continuous features, the process yields six clean features: Pclass, Sex, Age, Title, GroupSize, and FareLog.

A note on multicollinearity: Title and Sex are highly correlated, which is expected since titles encode gender. However, tree-based models handle this well, and Title adds nuance beyond just gender.

Before training complex models, a domain-driven baseline is established: the Women and Children First rule.

This rule predicts survival for all females and children under 14. It achieves an F1 score of 0.735 and an accuracy of roughly 79 percent.

This is a strong benchmark. Any machine learning model must clearly outperform this to justify its complexity. The baseline explicitly uses the original, unscaled Age values so the rule remains meaningful.

Three models are trained using 5-fold stratified cross-validation:
Logistic Regression.
Random Forest.
And Gradient Boosting.

A Soft Voting Ensemble combining all three is also evaluated. Hyperparameters are tuned using GridSearchCV.

Here are the cross-validated results:
Logistic Regression scores an F1 of 0.752.
Random Forest improves to 0.791.
Gradient Boosting achieves the highest score with an F1 of 0.802.
The Voting Ensemble scores 0.800, which is good but does not beat Gradient Boosting.

Gradient Boosting improves over the baseline by nearly 0.07 points in F1 score. Therefore, Gradient Boosting is selected as the final model.

The Gradient Boosting model is retrained on the full training set. No further tuning is done to avoid overfitting; the hyperparameters from cross-validation are fixed.

For error analysis, out-of-fold predictions are used. The confusion matrix shows the model is slightly conservative, with slightly more false negatives than false positives.

The feature importance plot confirms the EDA findings:
Title is the most important feature.
Pclass and FareLog follow closely.
GroupSize and Age also contribute meaningful signal.

This confirms the model relies on meaningful, interpretable signals.

Finally, predictions are generated for the 418 passengers in the test set.

The submission file passes all sanity checks:
418 rows.
Correct columns: PassengerId and Survived.
Unique IDs and binary labels.

To summarize:
A robust machine learning pipeline was built that outperforms a strong historical baseline. By focusing on careful feature engineering and rigorous validation, a Gradient Boosting model was selected that achieves a cross-validated F1 score of 0.802 and a strong Kaggle submission score of 0.77511.

Thank you for watching. I am happy to answer any questions.
