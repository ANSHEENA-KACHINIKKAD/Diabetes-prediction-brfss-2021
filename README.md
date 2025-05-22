# Diabetes Classification Model: Predicting Health Outcomes from BRFSS 2021 Data

## Project Overview:

This project focuses on developing a robust machine learning model to predict the likelihood of an individual having diabetes (including prediabetes) based on various health indicators from the Behavioral Risk Factor Surveillance System (BRFSS) 2021 dataset. The primary goal is to build an accurate and reliable classification system that can aid in early risk assessment and public health initiatives.
A significant challenge addressed in this project was the pronounced class imbalance within the dataset, a common issue in health-related data. To overcome this, the methodology employed advanced preprocessing techniques, synthetic data generation (SMOTE), and rigorous hyperparameter tuning, specifically optimizing for performance on minority classes.

## Dataset:

The dataset utilized is the "Diabetes 012 Health Indicators BRFSS 2021", obtained from the Centers for Disease Control and Prevention (CDC). It comprises self-reported health survey data from U.S.

### Target Variable:

* Diabetes_012:
    * 0: No Diabetes
    * 1:Prediabetes
    * 2:Diabetes

* Key Features (after selection/removal of less relevant ones):

    * HighBP: High Blood Pressure (0=no, 1=yes)
    * HighChol: High Cholesterol (0=no, 1=yes)
    * BMI: Body Mass Index
    * Smoker: Smoked at least 100 cigarettes in lifetime (0=no, 1=yes)
    * Stroke: Ever had a stroke (0=no, 1=yes)
    * HeartDiseaseorAttack: Ever had coronary heart disease or myocardial infarction (0=no, 1=yes)
    * PhysActivity: Physical activity in past 30 days (0=no, 1=yes)
    * Fruits: Consume Fruit 1 or more times per day (0=no, 1=yes)
    * Veggies: Consume Vegetables 1 or more times per day (0=no, 1=yes)
    * HvyAlcoholConsump: Heavy alcohol consumption (adult men having 14+ drinks/week, adult women having 7+ drinks/week) (0=no, 1=yes)
    * GenHlth: General Health (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)
    * DiffWalk: Difficulty walking or climbing stairs (0=no, 1=yes)
    * Age: Age in 13 categories (1='18-24', 2='25-29', ..., 13='80 or older')
  
## Methodology:

The project follows a comprehensive machine learning workflow:

1.Data Loading & Initial EDA:
  * The dataset was loaded and inspected for its structure, data types, and initial statistics.
  * The distribution of the Diabetes_012 target variable was analyzed, confirming a significant class imbalance with the majority of individuals falling into the 'No Diabetes' category.
2.Column Removal:
  * Irrelevant or less impactful columns identified during initial assessment were dropped. These included: Sex, Income, Education, CholCheck, AnyHealthcare, NoDocbcCost, PhysHlth, MentHlth.
3.Exploratory Data Analysis (EDA) & Visualization:
  * Histograms: Visualized the distributions of all features. Key observations included the highly skewed nature of binary features like Stroke, HeartDiseaseorAttack, and HvyAlcoholConsump (where the '1' 
     category was rare), and the right-skewed distribution of BMI. Age showed a higher concentration in older groups.
  * Correlation Heatmap: Provided a high-level view of feature interdependencies.
  * Box Plots: Used to visually identify potential outliers in numerical features.
4.Outlier Detection & Handling:
  * The Interquartile Range (IQR) method was applied to numerical columns (e.g., BMI) to identify and remove extreme values.
  * Insight from Box Plots: BMI showed a considerable number of outliers on both the higher and lower ends. Binary features such as Stroke, HeartDiseaseorAttack, HvyAlcoholConsump, and DiffWalk also 
    appeared to have their '1' values flagged as statistical outliers due to their extreme rarity compared to the '0' values, highlighting the class imbalance within these specific features. While 
    technically "outliers" by IQR, these represent valid rare events crucial for classification.
5.Data Preprocessing:
  * Missing Values: Handled using SimpleImputer (median for numerical, most frequent for categorical).
  * Feature Scaling: StandardScaler was applied to numerical features to normalize their range, which is essential for distance-based algorithms and gradient-descent optimization.
  * Categorical Encoding: OneHotEncoder transformed categorical features into a numerical binary format.
  * A ColumnTransformer within a Pipeline was utilized to ensure consistent application of these transformations across different feature types and to prevent data leakage between training and testing 
    sets.

6.Class Imbalance Handling:
  * Given the severe class imbalance in Diabetes_012, SMOTE (Synthetic Minority Over-sampling Technique) was applied exclusively to the training data. This technique synthesizes new examples for the 
   minority classes (Prediabetes and Diabetes), creating a more balanced dataset for model training and preventing the models from becoming biased towards the majority class.
7.Model Training & Initial Evaluation:
  * Several widely-used classification algorithms were trained on the SMOTE-resampled training data:
     * Logistic Regression
     * Decision Tree Classifier
     * Random Forest Classifier
     * K-Nearest Neighbors (KNN)
  * Initial evaluation involved accuracy_score, classification_report, confusion_matrix, and weighted F1-score. The weighted F1-score was prioritized due to its robustness in evaluating models on 
    imbalanced datasets, considering both precision and recall for all classes.
8.Hyperparameter Tuning:
  * Extensive hyperparameter tuning was performed for all models using GridSearchCV with 3-fold cross-validation.
  * The optimization objective for tuning was the weighted F1-score, ensuring that the chosen hyperparameters yielded the best overall performance across all diabetes classes.
  * class_weight='balanced' or balanced_subsample parameters were also included in the tuning grids for applicable models to further mitigate imbalance effects during model training.
9.Final Model Selection & Saving:
  * The model demonstrating the highest weighted F1-score on the unseen test set after hyperparameter tuning was selected as the optimal model.
  * The complete, best-tuned machine learning pipeline (including all preprocessing steps and the final classifier) was saved using joblib for future deployment or inference.

## Key Findings & Insights

 * Dominant Healthy Population: The initial EDA confirmed that individuals without diabetes (Diabetes_012 = 0) form the vast majority of the dataset, highlighting the need for specialized imbalance handling.
 * Prevalent Risk Factors: Features like HighBP, HighChol, and Smoker are highly prevalent in the dataset, indicating they are common conditions within the surveyed population and likely strong predictors.
 * Rare but Impactful Conditions: While rare, conditions like Stroke and HeartDiseaseorAttack are critical health indicators.
 * The model's ability to correctly classify these minority cases is crucial for 
   effective risk assessment.
 * BMI Distribution: BMI showed a significant range and right-skewness, with many individuals in the overweight/obese categories. Outlier removal helped normalize its distribution.
 * Age's Influence: The dataset's concentration in older age groups reinforces Age as a fundamental predictor for diabetes risk.
 * Tuned Model Performance: After thorough hyperparameter tuning and class imbalance handling with SMOTE, the LogisticRegression classificationn model emerged as the top-performing model, achieving a 
   weighted F1-score of 0.59 on the test set. This demonstrates its effectiveness in distinguishing between healthy individuals, prediabetics, and diabetics.

## Technologies Used

 * Python
 * Pandas: Data manipulation and analysis.
 * NumPy: Numerical operations.
 * Matplotlib: Foundational plotting library.
 * Seaborn: Enhanced statistical data visualization.
 * Scikit-learn: Machine learning algorithms, preprocessing, model selection, and evaluation metrics.
 * Imbalanced-learn: Tools for handling class imbalance (SMOTE).
 * Joblib: Efficient saving and loading of Python objects (trained models).
 * Colab Notebooks: Development and experimentation environment.



