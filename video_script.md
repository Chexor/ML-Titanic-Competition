# Presentation Outline: Titanic Survival Prediction

**Target length:** 10 to 15 minutes  
**Notebook:** `Titanic_Disaster_Challenge-Final.ipynb`

## 1. Introduction (2 mins)
- **Greeting:** "Good day. My name is Tim Kindt. Welcome to my presentation on the Titanic Survival Prediction project."
- **Context:** The sinking of the RMS Titanic in 1912 is one of history's most famous tragedies. This project applies data science to predict passenger survival based on historical records.
- **Problem Type:** Supervised binary classification.
  - *Supervised:* Training data contains the known answers (`Survived` column).
  - *Binary:* Target has only two states (0 = Did not survive, 1 = Survived).
- **Workflow Overview:** 
  - Data Loading & Exploratory Data Analysis (EDA).
  - Feature Engineering & Preprocessing.
  - Baseline Modeling.
  - Model Training, Tuning, and Validation.
  - Final Evaluation & Kaggle Submission.
- **Scope:** The approach prioritizes methodological correctness, interpretability, and robust validation over extreme complexity. Only shallow learning models (Logistic Regression, Random Forest, Gradient Boosting) are used. No deep learning.

## 2. Exploratory Data Analysis (EDA) (2.5 mins)
- **Data Loading:** Train dataset (891 passengers, includes target) and Test dataset (418 passengers, target missing).
- **Key EDA Findings:**
  - **Sex:** The strongest predictor. Females had a significantly higher survival rate.
  - **Pclass (Passenger Class):** Highly influential. First-class passengers survived more often, reflecting socio-economic status and cabin placement.
  - **Age:** Contains many missing values. Requires careful, grouped imputation rather than a global median.
  - **Fare:** Heavily right-skewed. A few paid massive amounts. Needs transformation (log scale) to stabilize models.
  - **Cabin:** Mostly missing. The raw text is dropped.
- **Train vs. Test Check:** Distributions match perfectly, giving confidence that our preprocessing strategy will generalize to unseen data.

## 3. Feature Engineering & Preprocessing (3.5 mins)
- **Objective:** Transform raw data into a compact, highly informative set of 6 features. Train and test are combined temporarily (excluding the target) to ensure consistent preprocessing.
- **Feature 1: Title:** Extracted from names (Mr, Mrs, Miss, Master). Rare titles (Dr, Rev) grouped into `Rare`. Powerfully captures gender, age, and social role.
- **Feature 2: GroupSize:** Derived from shared ticket numbers. Captures travel companions, friends, and nannies who evacuated together—a stronger signal than family size alone.
- **Feature 3: Age:** Imputed by sampling from the age distribution of the passenger's specific `Title` group. Preserves natural variance. Seed fixed for reproducibility.
- **Feature 4: FareLog:** Missing fares filled with median per Pclass. A log transformation compresses extreme outliers into a normal distribution.
- **Categorical Encoding:** 
  - `Sex` is mapped to binary (0/1). 
  - `Title` uses **Ordinal Encoding**. *Theory note:* We avoided One-Hot Encoding to prevent the "curse of dimensionality" and sparsity. Tree-based models handle ordinal values natively by learning optimal numerical splits.
- **Feature Scaling & Data Leakage:**
  - `Age`, `FareLog`, and `GroupSize` are standardized.
  - *Theory note:* The `StandardScaler` is fit **strictly on the training data**. Transforming the test data using the train scaler prevents **Data Leakage**, ensuring the model does not indirectly "peek" at test distributions.
- **Final Set:** Exactly 6 clean features: `Pclass`, `Title`, `Age`, `Sex`, `FareLog`, and `GroupSize`.

## 4. Domain-Driven Baseline (1.5 mins)
- **The Rule:** "Women and Children First." Predicts survival for all females and children under 14.
- **Metrics:** Achieves an Accuracy of ~79% and an F1 score of **0.735**.
- **Purpose:** A rigid, historically accurate benchmark. Any ML model must clearly beat an F1 of 0.735 to justify its added complexity. 
- **Note:** The baseline explicitly uses the original, unscaled Age values.

## 5. Model Training and Cross-Validation (2.5 mins)
- **Metric Selection (Accuracy vs F1):**
  - *Theory note:* Roughly 62% of passengers died. Due to this class imbalance, **Accuracy** is misleading (guessing "everyone dies" yields 62% accuracy). 
  - We use the **F1-score**, which balances Precision (minimizing false alarms) and Recall (finding all actual survivors).
- **Validation Setup:** Stratified 5-Fold Cross-Validation.
  - `n_splits=5`: Good balance of training data (80%) vs validation data (20%).
  - `shuffle=True`: Randomizes hidden ordering (like ticket sequences).
  - `random_state=42`: Ensures perfectly reproducible splits.
  - *Stratified:* Ensures the 38% survival rate is maintained across every fold.
- **The Models:**
  1. **Logistic Regression:** Simple, interpretable linear baseline.
  2. **Random Forest:** Non-linear ensemble of independent decision trees.
  3. **Gradient Boosting:** Advanced ensemble. Builds trees sequentially to correct the residual errors of previous trees.
  - *Note:* Gradient Boosting utilizes early stopping (500 trees, 20% validation fraction) to prevent overfitting.
- **Voting Ensemble:** A Soft Voting classifier averages the predicted probabilities of the three tuned models.

## 6. Model Results & Feature Importance (2 mins)
- **Cross-Validated F1 Scores:**
  - Baseline: 0.735
  - Logistic Regression: 0.752
  - Random Forest: 0.791
  - **Gradient Boosting: 0.802** (The Winner)
  - Voting Ensemble: 0.800
- **Selection:** Gradient Boosting improves over the baseline by 0.067 F1 points. It is selected as the final model.
- **Final Fit & Error Analysis:** 
  - Retrained on the full training set using the fixed CV hyperparameters.
  - Out-of-fold confusion matrix shows the model is slightly conservative (highly precise, but produces slightly more false negatives).
- **Feature Importance:**
  - `Title` dominates (~47%), proving our extraction strategy worked perfectly.
  - `Pclass` and `FareLog` follow, confirming socio-economic status was a key survival factor.
  - Raw `Sex` has low importance *only because* the gender signal is already perfectly encoded within `Title`.

## 7. Kaggle Submission & Conclusion (1 min)
- **Prediction:** The Gradient Boosting model predicts the 418 test passengers.
- **Sanity Checks:** The submission file strictly passes all format checks (418 rows, correct headers, unique IDs, binary labels).
- **Result:** Uploaded to Kaggle, this 6-feature GB model achieved an exceptional public score of **0.77511**.
- **Summary:** A robust pipeline was built using scrupulous feature engineering, data leakage prevention, F1-score optimization, and stratified CV. The compact Gradient Boosting model significantly outperformed the historical baseline.
- **Closing:** "Thank you for your time. I am happy to answer any questions."
