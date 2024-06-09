# Heart Disease Key Factor Analysis Project

## Project Description
This project aims to implement and compare various machine learning models for predicting heart disease. The given data is used to train the models, and the performance of each model is evaluated.

## Data Description
The data is loaded from the heart.csv file. The main features are as follows:

## Continuous Variables
- Age: Age
- RestingBP: Resting blood pressure
- Cholesterol: Cholesterol level
- MaxHR: Maximum heart rate
- Oldpeak: ST depression induced by exercise

## Categorical Variables
- Sex: Sex (M = Male, F = Female)
- ChestPainType: Chest pain type (ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic, TA = Typical Angina)
- FastingBS: Fasting blood sugar (1 = Fasting blood sugar > 120 mg/dl, 0 = Fasting blood sugar <= 120 mg/dl)
- RestingECG: Resting electrocardiogram results (Normal = Normal, ST = Having ST-T wave abnormality, LVH = Showing probable or definite left ventricular hypertrophy)
- ExerciseAngina: Exercise-induced angina (Y = Yes, N = No)
- ST_Slope: ST segment slope (Up = Upsloping, Flat = Flat, Down = Downsloping)
These categorical variables are converted to dummy variables in the code:

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
1. Load the data from the heart.csv file and convert the HeartDisease variable to a factor.
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
Visualize important variables and interaction effects.
Visualize the ROC curves of each model.
Visualize the variable importance for each model.

## Execution
1. Install the required packages
2. Place the heart.csv file in the correct path.
3. Run the script to preprocess the data and train the models.
4. Review the results and analyze the visualized graphs.

## Contact
For any questions or inquiries, please contact: youngsun.lee07@gmail.com
