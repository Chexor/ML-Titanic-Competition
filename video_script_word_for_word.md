Hello, my name is Tim Kindt. I am a student in Applied Computer Science, specializing in Artificial Intelligence at VIVES University Kortrijk. Welcome to my presentation on the Titanic survival prediction project.

The goal of this project is to build a machine learning pipeline that can predict whether a passenger survived the sinking of the RMS Titanic in 1912.

This is a supervised binary classification problem. Supervised, because we are training the model using a dataset where the correct answer is already known. Binary, because the target has exactly two possible outcomes: zero for did not survive, and one for survived.

The notebook follows a rigorous, end-to-end workflow. It begins with data loading and exploratory data analysis. Then, the data is cleaned and transformed. Next, a strong, domain-driven baseline is established, after which multiple machine learning models are trained, tuned, and compared. Finally, the best model is selected, evaluated, and used to generate a Kaggle-ready submission file.

The tools used are Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn. Following the assignment scope, this project focuses exclusively on shallow learning models. No deep learning or neural networks were used. The focus is on methodological correctness, interpretability, and robust validation.

The workflow starts by loading the training dataset. This dataset contains eight hundred and ninety one passengers and includes our target variable, Survived. The test dataset contains four hundred and eighteen passengers but does not include the survival labels. This is the data we must predict.

After loading the data, the notebook performs Exploratory Data Analysis.

Several critical insights emerge.
First, sex is an incredibly strong predictor. Female passengers had a significantly higher survival rate.
Second, passenger class strongly influenced survival. First-class passengers survived much more often than third-class passengers.
Third, the age column has many missing values, which will require careful imputation. 
Fourth, the fare column is heavily right-skewed. The vast majority of passengers paid very little, while a handful of first-class passengers paid enormous amounts. This extreme skewness will need to be handled.

The notebook also compares the distributions of the test set against the training set. Because the distributions match, we can be confident that the preprocessing strategy will generalize well to the test data.

Next, we move to Feature Engineering. To ensure consistency, the train and test sets are temporarily combined, but the target variable is strictly separated to avoid data leakage.

We engineer a focused set of six features.

The first engineered feature is Title. Instead of using raw names, titles like Mr, Mrs, Miss, and Master are extracted. Rare titles are grouped into a Rare category. This feature is powerful because it captures gender, social status, and age hints all at once.

The second feature is GroupSize. Instead of just counting family members, this feature counts how many passengers shared the exact same ticket number. This is a strong predictor because it captures travel companions, friends, and nannies who likely evacuated as a group.

The third feature is Age. For missing ages, the notebook uses a sampling strategy based on the passenger's title. Instead of filling all missing values with a single global median, we draw random samples from the age distribution of that specific title group. This preserves the natural variance of the data. A fixed random seed is used to ensure the results are completely reproducible.

The fourth feature is FareLog. Missing fares are filled with the median fare of the corresponding passenger class. Then, a log transformation is applied. This compresses the extreme outliers, resulting in a more normal distribution.

Next, we handle Categorical Encoding. Sex is mapped to a simple binary indicator: zero for female, one for male. 
For the Title feature, we use an OrdinalEncoder. We avoided One-Hot Encoding to prevent the curse of dimensionality and sparsity. Tree-based models can natively handle ordinal values by learning the optimal numerical split points. By using ordinal encoding, we keep our feature set extremely compact.

Finally, we apply Feature Scaling to the continuous variables: Age, FareLog, and GroupSize. We use a StandardScaler to give them a mean of zero and a variance of one. 

Here, we must highlight a vital machine learning concept: preventing Data Leakage. The StandardScaler is fitted strictly on the training portion of the data. Only then is it used to transform both the train and test sets. If we had fitted the scaler on the entire dataset, information about the test set's distribution would have leaked into the training process, resulting in overly optimistic validation scores.

After preprocessing, we are left with exactly six clean features: Pclass, Title, Age, Sex, FareLog, and GroupSize.

Before training any complex models, a strong, domain-driven baseline is established: the Women and Children First rule. 

This historical rule predicts survival for all female passengers and for children under fourteen. Everyone else is predicted to not survive. The baseline is calculated using the original, unscaled age values.

This simple rule achieves an F-one score of zero point seven three five. This provides a rigorous benchmark. Any machine learning model we build must clearly outperform this score.

With the features ready, the notebook moves on to model training.

First, let us discuss the choice of evaluation metric. In this dataset, roughly sixty two percent of passengers did not survive. Because of this class imbalance, relying solely on Accuracy can be highly misleading. A naive model that simply guesses nobody survives would automatically achieve sixty two percent accuracy. 

To solve this, the notebook uses the F-one score as the primary selection metric. The F-one score is the harmonic mean of Precision and Recall. It forces the model to find a healthy balance: it must try to identify as many actual survivors as possible, which is Recall, without making too many false positive guesses, which is Precision.

The models are evaluated using a strict Stratified 5-Fold Cross-Validation setup. 
The data is split into five folds, meaning the model trains on eighty percent of the data and validates on the remaining twenty percent, repeating this five times. 
The shuffle parameter ensures that any hidden sorting in the dataset is randomized. 
The random state parameter ensures that every model is evaluated on the exact same data splits, making the comparison perfectly fair. 
Crucially, it is a Stratified split. This guarantees that the thirty eight percent survival rate is preserved across every single training and validation fold.

Three distinct shallow learning models are trained.
One: Logistic Regression. A fast, interpretable linear baseline.
Two: Random Forest. An ensemble method that builds multiple independent decision trees and averages their predictions.
Three: Gradient Boosting. An advanced ensemble method that builds trees sequentially. Each new tree tries to correct the residual errors made by the previous trees.

Hyperparameters are tuned using GridSearchCV. For Gradient Boosting, an early stopping configuration is used with five hundred trees, a validation fraction of twenty percent, and a learning rate of zero point zero five. This prevents overfitting.

Additionally, a Soft Voting Ensemble is created. This model averages the predicted probabilities of the tuned Logistic Regression, Random Forest, and Gradient Boosting models.

Looking at the cross-validation results.

The historical baseline achieved an F-one score of zero point seven three five.
Logistic Regression provides a modest improvement with an F-one score of zero point seven five two.
Random Forest brings a significant jump, achieving an F-one score of zero point seven nine one.
Gradient Boosting performs the best, achieving a cross-validated F-one score of zero point eight zero two.
The Voting Ensemble scores zero point eight zero zero, which is very strong, but does not beat the standalone Gradient Boosting model.

Because it achieved the highest cross-validated F-one score, Gradient Boosting is selected as the final model.

Having selected Gradient Boosting, the model is retrained one final time on the entire training dataset. No further hyperparameter tuning is done at this stage; we strictly reuse the optimal settings found during cross-validation.

To perform an honest error analysis, the notebook uses the out-of-fold predictions. The confusion matrix reveals that the model is slightly conservative: it produces slightly more false negatives than false positives. This means it sometimes predicts a passenger died when they actually survived, but it is highly precise when it does predict survival.

The feature importance is also analyzed. Gradient Boosting provides built-in importance metrics. 
The Title feature is by far the most important, carrying nearly forty seven percent of the predictive weight. This validates our feature engineering, as Title encapsulates gender, age, and social status.
Pclass and FareLog follow closely, confirming that socio-economic status was a critical factor during the evacuation.
GroupSize and Age also contribute meaningful signal. The raw Sex feature has very low importance, but only because the gender signal is already perfectly encoded within the Title feature.

In the final step, the trained Gradient Boosting model is used to generate predictions for the four hundred and eighteen passengers in the test set.

A submission dataframe is constructed and subjected to several automated sanity checks. The notebook verifies that the file contains exactly four hundred and eighteen rows, has the correct column headers, contains no duplicate IDs, and has strictly binary labels. 

After passing all checks, the predictions are saved to submission dot csv. When submitted to Kaggle, this exact pipeline achieved an exceptional public leaderboard score of zero point seven seven five one one.

To summarize.
This project successfully implemented a robust, end-to-end machine learning pipeline. By focusing on meticulous feature engineering, strictly preventing data leakage, utilizing appropriate evaluation metrics like the F-one score, and employing rigorous stratified cross-validation, a highly accurate Gradient Boosting model was developed. The final model is compact, utilizing just six features, yet it significantly outperforms historical baselines and achieves top-tier results.

Thank you for your time and attention. I would now be happy to answer any questions.
