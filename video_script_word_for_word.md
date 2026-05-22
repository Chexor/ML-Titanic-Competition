# Word-for-Word Presentation Script: Titanic Survival Prediction

**Target length:** 10 to 15 minutes  
**Notebook:** `Titanic_Disaster_Challenge-Final.ipynb`

## 1. Introduction

Hello, my name is Tim Kindt, a student in Applied Computer Science, specializing in Artificial Intelligence at VIVES University Kortrijk. Welcome to my presentation on the Titanic survival prediction project.

The goal of this project is to build a machine learning pipeline that can predict whether a passenger survived the tragic sinking of the RMS Titanic in 1912, based on historical passenger records. 

From a machine learning perspective, this is a supervised binary classification problem. It is supervised because we are training the model using a dataset where the correct answer—the `Survived` column—is already known. It is binary because the target has exactly two possible outcomes: `0` for passengers who did not survive, and `1` for those who did.

The notebook follows a rigorous, end-to-end machine learning workflow. It begins with data loading and exploratory data analysis to uncover hidden patterns. Then, the data is cleaned and transformed during the feature engineering phase. We place a heavy emphasis on preventing data leakage and selecting the right evaluation metrics. Next, a strong, domain-driven baseline is established, after which multiple shallow machine learning models are trained, tuned, and compared using a strict cross-validation setup. Finally, the best model is selected, evaluated, and used to generate a Kaggle-ready submission file.

The tools used are Python, Pandas, NumPy, Matplotlib, Seaborn, and Scikit-Learn. Following the assignment scope, this project focuses exclusively on shallow learning models—no deep learning or neural networks were used. The focus is on methodological correctness, interpretability, and robust validation.

## 2. Data Loading And Exploratory Data Analysis

The workflow starts by loading the training dataset from the `Datasets` folder. This training dataset contains 891 passengers and includes our target variable, `Survived`. The test dataset contains 418 passengers but does not include the survival labels—this is the data we must predict for the Kaggle competition.

After loading the data, the notebook performs Exploratory Data Analysis, or EDA. The goal here is to understand the distributions, locate missing values, and find the strongest correlations with survival before we even touch a machine learning model.

Several critical insights emerge from the EDA:
First, sex is an incredibly strong predictor. Female passengers had a significantly higher survival rate than male passengers.
Second, passenger class strongly influenced survival. First-class passengers survived much more often than third-class passengers. Historically, this aligns with the fact that first-class cabins were closer to the boat deck, and wealthier passengers had prioritized access to lifeboats.
Third, the age column has many missing values, which will require careful imputation. 
Fourth, the fare column is heavily right-skewed. The vast majority of passengers paid very little, while a handful of first-class passengers paid enormous amounts. This extreme skewness will need to be handled to prevent it from destabilizing our models.

The notebook also compares the distributions of the test set against the training set. This is a crucial sanity check: because the distributions match, we can be confident that the preprocessing strategy we develop on the training data will generalize well to the test data.

## 3. Feature Engineering and Preprocessing

Next, we move to Feature Engineering. The goal here is to transform the raw dataset into a compact set of highly informative, machine-readable features. To ensure consistency, the train and test sets are temporarily combined, but the target variable is strictly separated to avoid any data leakage.

We engineer a focused set of six features.

The first engineered feature is `Title`. Instead of using raw names, which are too unique and would lead to severe overfitting, titles like `Mr`, `Mrs`, `Miss`, and `Master` are extracted. Rare titles such as `Dr` or `Col` are grouped into a `Rare` category. This feature is very powerful because it captures gender, social status, and age hints all at once.

The second feature is `GroupSize`. Instead of just counting family members, this feature counts how many passengers shared the exact same ticket number. This is a stronger predictor than family size alone, because it captures travel companions, friends, and nannies who likely stayed together and evacuated as a group.

The third feature is `Age`. For missing ages, the notebook uses a sampling strategy based on the passenger's title. Instead of filling all missing values with a single global median, we draw random samples from the age distribution of that specific title group within the training set. This preserves the natural variance of the data. A fixed random seed is used to ensure the results are 100% reproducible.

The fourth feature is `FareLog`. Missing fares are filled with the median fare of the corresponding passenger class. Then, a log transformation is applied. This compresses the extreme outliers in the fare column, resulting in a more normal distribution that is easier for models to process.

Next, we handle Categorical Encoding. Machine learning models require numerical input. `Sex` is mapped to a simple binary indicator: 0 for female, 1 for male. 
For the `Title` feature, we use an `OrdinalEncoder`. It is important to note why we chose ordinal encoding instead of One-Hot Encoding. One-Hot Encoding would create a separate column for every title, expanding the feature space and introducing sparsity. Tree-based models, like Random Forest and Gradient Boosting, can natively handle ordinal values by learning the optimal numerical split points. By using ordinal encoding, we avoid the curse of dimensionality and keep our feature set extremely compact.

Finally, we apply Feature Scaling to the continuous variables: `Age`, `FareLog`, and `GroupSize`. We use a `StandardScaler` to give them a mean of 0 and a variance of 1. 

Here, we must highlight a vital machine learning concept: preventing Data Leakage. The `StandardScaler` is fitted strictly on the training portion of the data. Only then is it used to transform both the train and test sets. If we had fitted the scaler on the entire dataset, information about the test set's distribution would have "leaked" into the training process, resulting in overly optimistic cross-validation scores.

After preprocessing, we are left with exactly six clean features: `Pclass`, `Title`, `Age`, `Sex`, `FareLog`, and `GroupSize`.

## 4. Domain-Driven Baseline

Before training any complex models, a strong, domain-driven baseline is established: the "Women and Children First" rule. 

This historical rule predicts survival for all female passengers and for children under the age of 14. Everyone else is predicted to not survive. To ensure this rule remains meaningful, the baseline is calculated using the original, unscaled age values.

This simple rule achieves an accuracy of roughly 79 percent and an F1 score of 0.735. This provides a rigorous benchmark. Any machine learning model we build must clearly outperform this 0.735 F1 score; otherwise, the model's complexity is not justified.

## 5. Model Training, Metric Selection, and Cross-Validation

With the features ready, the notebook moves on to model training.

First, let us discuss the choice of evaluation metric. In this dataset, roughly 62 percent of passengers did not survive. Because of this class imbalance, relying solely on **Accuracy** can be highly misleading. A naive model that simply guesses "nobody survives" would automatically achieve 62 percent accuracy, despite being completely useless. 

To solve this, the notebook uses the **F1-score** as the primary selection metric. The F1-score is the harmonic mean of Precision and Recall. It forces the model to find a healthy balance: it must try to identify as many actual survivors as possible—which is Recall—without making too many false positive guesses—which is Precision.

The models are evaluated using a strict **Stratified 5-Fold Cross-Validation** setup. 
The data is split into 5 folds, meaning the model trains on 80 percent of the data and validates on the remaining 20 percent, repeating this five times. 
The `shuffle=True` parameter ensures that any hidden sorting in the dataset, like ticket order, is randomized. 
The `random_state=42` parameter ensures that every model is evaluated on the exact same data splits, making the comparison perfectly fair and reproducible. 
Crucially, it is a *Stratified* split. This guarantees that the 38 percent survival rate is preserved across every single training and validation fold, preventing artificially skewed splits.

Three distinct shallow learning models are trained:
1. **Logistic Regression**: A fast, interpretable linear model that serves as our machine learning baseline.
2. **Random Forest**: An ensemble method that builds multiple independent decision trees and averages their predictions, naturally capturing non-linear relationships.
3. **Gradient Boosting**: A more advanced ensemble method that builds trees sequentially. Instead of building independent trees, each new tree specifically tries to correct the residual errors made by the previous trees.

Hyperparameters for these models are tuned using `GridSearchCV`. For Gradient Boosting, an early stopping configuration is used with 500 trees, a validation fraction of 20%, and a learning rate of 0.05. This prevents the model from overfitting by stopping the training process if the validation score does not improve for 20 consecutive iterations.

Additionally, a **Soft Voting Ensemble** is created. This model averages the predicted probabilities of the tuned Logistic Regression, Random Forest, and Gradient Boosting models, which can often increase overall robustness.

## 6. Model Comparison and Selection

Looking at the cross-validation results:

The historical baseline achieved an F1 score of 0.735.
Logistic Regression provides a modest improvement with an F1 score of 0.752.
Random Forest brings a significant jump, achieving an F1 score of 0.791.
However, Gradient Boosting performs the best, achieving an impressive cross-validated F1 score of 0.802.
The Voting Ensemble scores 0.800, which is very strong, but it does not beat the standalone Gradient Boosting model.

Compared to the baseline, Gradient Boosting improves the accuracy by 0.063 and the F1 score by 0.067. Because it clearly achieved the highest cross-validated F1 score, Gradient Boosting is officially selected as the final model.

## 7. Final Model Evaluation

Having selected Gradient Boosting, the model is retrained one final time on the entire training dataset. No further hyperparameter tuning is done at this stage; we strictly reuse the optimal settings found during cross-validation to maximize predictive power without risking overfitting.

To perform an honest error analysis, the notebook uses the out-of-fold predictions. The confusion matrix reveals that the model is slightly conservative: it produces slightly more false negatives than false positives. This means it sometimes predicts a passenger died when they actually survived, but it is highly precise when it does predict survival.

The feature importance is also analyzed. Gradient Boosting provides built-in importance metrics that show which features drove the predictions. 
The `Title` feature is by far the most important, carrying nearly 47 percent of the predictive weight. This perfectly validates our feature engineering, as `Title` encapsulates gender, age, and social status.
`Pclass` and `FareLog` follow closely, confirming that socio-economic status was a critical factor during the evacuation.
`GroupSize` and `Age` also contribute meaningful signal. Interestingly, the raw `Sex` feature has very low importance here, but only because the gender signal is already so strongly encoded within the `Title` feature.

This analysis proves that the model relies on logical, interpretable, and historically accurate signals.

## 8. Kaggle Submission and Conclusion

In the final step, the trained Gradient Boosting model is used to generate predictions for the 418 passengers in the test set.

A submission dataframe is constructed and subjected to several automated sanity checks. The notebook verifies that the file contains exactly 418 rows, has the correct `PassengerId` and `Survived` column headers, contains no duplicate IDs, and has strictly binary 0 and 1 labels. 

After passing all checks, the predictions are saved to `submission.csv`. When submitted to Kaggle, this exact pipeline achieved a phenomenal public leaderboard score of 0.77511, proving its ability to generalize to unseen data.

To summarize:
This project successfully implemented a robust, end-to-end machine learning pipeline. By focusing on meticulous feature engineering, strictly preventing data leakage, utilizing appropriate evaluation metrics like the F1-score, and employing rigorous stratified cross-validation, a highly accurate Gradient Boosting model was developed. The final model is compact, utilizing just six features, yet it significantly outperforms historical baselines and achieves top-tier results.

Thank you for your time and attention. I would now be happy to answer any questions.
