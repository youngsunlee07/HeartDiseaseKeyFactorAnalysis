# Heart Disease Key Factor Analysis Project

## Project Description
This project aims to predict heart disease and identify the most influential factors affecting heart disease using the R programming language. To achieve this, the project utilizes both statistical and machine learning methods to identify and quantify the impact of various predictors on heart disease. Specifically, the Chi-squared test is employed to evaluate the significance of categorical variables in relation to heart disease outcomes. Additionally, various machine learning models are used to further analyze these factors and understand their predictive power.

## Data Description
The dataset, `heart.csv`, includes various predictors for heart disease such as age, blood pressure, cholesterol levels, and more. Both continuous and categorical variables are included:
- **Continuous Variables:** Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- **Categorical Variables:** Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope

## Required Packages
The following R packages are required:
- caret
- e1071
- randomForest
- ROCR
- dplyr
- tidyr
- rpart
- gbm
- pROC
- ggplot2
- kernlab
- gridExtra
- car
- lattice
- purrr

## Data Preprocessing
1. Load the data from the 'heart.csv' file and convert the HeartDisease variable to a factor.
2. Create dummy variables for the categorical variables.
3. Keep the continuous variables as they are.
4. Combine the dummy variables and continuous variables into a final data frame.
5. Split the data into training and testing sets.

## Variable Selection and Interaction Effects Analysis
- Use Chi-Squared tests and Fisher's Exact tests to select important categorical variables.
- Use Point-Biserial correlation to select important continuous variables.
- Visualize significant interaction effects.

## Model Training and Evaluation
Train and evaluate various machine learning models:
- Logistic Regression
- Decision Tree
- k-NN
- Random Forest
- Gradient Boosting
- SVM

Evaluate the performance of each model using the following metrics:
- Precision
- Recall
- F1 Score
- AUC (Area Under the Curve)

## Results Visualization
Visualizations include:
- Important variable identification.
- Interaction effects between variables.
- ROC curves for each model.

## Execution
1. Install the required packages
2. Ensure the path in the 'read.csv' function reflects the location of the 'heart.csv' file. 
3. Run the R script in your R environment.
4. Review the results and analyze the visualized graphs. 

## Contact
For any questions or inquiries, please contact: youngsun.lee07@gmail.com
