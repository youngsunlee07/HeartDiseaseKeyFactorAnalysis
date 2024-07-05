library(caret)
library(e1071)
library(randomForest)
library(ROCR)
library(dplyr)
library(tidyr)
library(rpart)
library(gbm)
library(pROC)
library(ggplot2) 
library(kernlab)
library(gridExtra)

# Load the data
data <- read.csv("file path\\heart.csv", fileEncoding = "UTF-8") 

# Convert HeartDisease variable
data$HeartDisease <- ifelse(data$HeartDisease == 0, "NoDisease", "HasDisease")
data$HeartDisease <- as.factor(data$HeartDisease)

# Binning continuous variables
continuous_vars <- c("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
continuous_vars_binned <- paste0(continuous_vars, "_binned")

# Adding new categorical columns to the dataframe
for(var in continuous_vars) {
  data[[paste0(var, "_binned")]] <- cut(data[[var]],
                                        breaks = quantile(data[[var]], probs = c(0, 1/3, 2/3, 1), na.rm = TRUE),
                                        labels = c("low", "medium", "high"),
                                        include.lowest = TRUE)
}

# List of categorical variables for dummy variable creation (excluding continuous variables)
predictors <- c("Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", continuous_vars_binned)

# Creation of dummy variables
dummies <- dummyVars(~ ., data = data[, predictors])
data_encoded <- predict(dummies, newdata = data[, predictors])
data_encoded <- as.data.frame(data_encoded) 
data_encoded$HeartDisease <- data$HeartDisease

# Splitting Train/Test data
set.seed(123)
splitIndex <- createDataPartition(data_encoded$HeartDisease, p = 0.8, list = TRUE)
trainData <- data_encoded[splitIndex$Resample1, ] 
testData <- data_encoded[-splitIndex$Resample1, ]  

# Function to calculate Chi-Squared test
calculate_chi_squared <- function(variable, data) {
  # Create a contingency table
  table <- table(data[[variable]], data$HeartDisease)
  # Perform Chi-Squared test
  test <- tryCatch({
    chisq.test(table)
  }, warning = function(w) {
    return(NA)
  }, error = function(e) {
    return(NA)
  })
  # Return the Chi-Squared statistic
  if (!is.na(test$statistic)) {
    return(test$statistic)
  } else {
    return(NA)
  }
}

# Perform Chi-Squared test for categorical dummy variables excluding the target variable
chi_squared_values <- sapply(names(trainData)[-which(names(trainData) == "HeartDisease")], function(var) {
  if (is.numeric(trainData[[var]]) && length(unique(trainData[[var]])) == 2) {
    calculate_chi_squared(var, trainData)
  } else {
    NA  # Treat continuous or non-binary categorical variables as NA
  }
}, simplify = "vector")

# Remove NA values
chi_squared_values <- chi_squared_values[!is.na(chi_squared_values)]

# Extract and print the top 10 variables based on Chi-Squared statistics
top_variables <- names(sort(chi_squared_values, decreasing = TRUE)[1:10])
top_chi_squared_values <- sort(chi_squared_values, decreasing = TRUE)[1:10]

print(top_variables)
print(top_chi_squared_values) 

# Visualization of the top 10 variables by Chi-Squared statistics
top_var_chi_sq_data <- data.frame(
  Variable = names(top_chi_squared_values),
  Chi_Squared = top_chi_squared_values
)

ggplot(top_var_chi_sq_data, aes(x = reorder(Variable, Chi_Squared), y = Chi_Squared)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Variables by Chi-Squared Statistics", x = "Variable", y = "Chi-Squared Value") 

# Generate combinations of all categorical dummy variables
categorical_vars <- names(trainData)[sapply(trainData, function(x) is.numeric(x) && length(unique(x)) == 2)]
all_combinations <- combn(categorical_vars, 2)

# Interactions
interaction_effects_cat <- sapply(1:ncol(all_combinations), function(x) {
  interaction_formula <- as.formula(paste("HeartDisease ~", all_combinations[1, x], "*", all_combinations[2, x]))
  interaction_model <- glm(interaction_formula, data = trainData, family = binomial)
  interaction_term <- paste(all_combinations[1, x], all_combinations[2, x], sep = ":")
  # Check if the interaction term is present in the model summary coefficients
  if (interaction_term %in% rownames(summary(interaction_model)$coefficients)) {
    return(summary(interaction_model)$coefficients[interaction_term, 4])  # t-value for the interaction term
  } else {
    return(NA)  
  } 
}, USE.NAMES = TRUE)
names(interaction_effects_cat) <- apply(all_combinations, 2, function(x) {
  paste(x[1], x[2], sep=":")
})
significant_interactions <- interaction_effects_cat[!is.na(interaction_effects_cat) & interaction_effects_cat < 0.05]
print(significant_interactions)

# Top 10 interactions
top_significant_interactions <- head(significant_interactions[order(abs(significant_interactions), decreasing = TRUE)], 10)

# Dataframe
top_interactions_df <- data.frame(
  Interaction = names(top_significant_interactions),
  tValue = top_significant_interactions
)

# t-value visualization
ggplot(top_interactions_df, aes(x = reorder(Interaction, tValue), y = tValue)) +
  geom_bar(stat = "identity") +
  coord_flip() +  # Flip x and y axes
  labs(title = "Top 10 Interaction Effects", x = "Interactions", y = "t-Value") +
  theme_minimal()

# When using only categorical dummies
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)
models <- list(
  "Logistic Regression" = train(HeartDisease ~ ., data = trainData, method = "glm", trControl = ctrl, metric = "ROC"),
  "Decision Tree" = train(HeartDisease ~ ., data = trainData, method = "rpart", trControl = ctrl, metric = "ROC"),
  "k-NN" = train(HeartDisease ~ ., data = trainData, method = "knn", trControl = ctrl, metric = "ROC"),
  "Random Forest" = train(HeartDisease ~ ., data = trainData, method = "rf", trControl = ctrl, metric = "ROC"),
  "Gradient Boosting" = train(HeartDisease ~ ., data = trainData, method = "gbm", trControl = ctrl, metric = "ROC", verbose = FALSE),
  "SVM" = train(HeartDisease ~ ., data = trainData, method = "svmRadial", trControl = ctrl, metric = "ROC")
)
logistic_model_all_coef <- coef(models$'Logistic Regression'$finalModel)
print(logistic_model_all_coef)
cat("\nLogistic Regression Coefficients (All Variables):\n")
print(logistic_model_all_coef)
metrics <- lapply(models, function(model) {
  prediction <- predict(model, newdata = testData, type = "raw")
  cm <- confusionMatrix(prediction, reference = testData$HeartDisease)
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']
  f1 <- 2 * (precision * recall) / (precision + recall)
  c(Precision = precision, Recall = recall, `F1 Score` = f1)
})
metrics_df <- as.data.frame(do.call(rbind, metrics))
rownames(metrics_df) <- names(models)
colnames(metrics_df) <- c("Precision", "Recall", "F1 Score")
print(metrics_df)
plot_colors <- c("red", "blue", "green", "purple", "orange", "brown")
auc_values <- numeric(length(models))
for(i in seq_along(models)) {
  model <- models[[i]]
  prob <- predict(model, newdata = testData, type = "prob")
  roc_curve <- roc(response=testData$HeartDisease, predictor=prob[,2], levels=c("NoDisease", "HasDisease"), direction=">")
  auc_values[i] <- auc(roc_curve)
  x <- 1 - roc_curve$specificities
  y <- roc_curve$sensitivities
  if(i == 1) {
    plot(x, y, type="l", col=plot_colors[i], xlim=c(0,1), ylim=c(0,1), xlab="False Positive Rate (1 - Specificity)", ylab="True Positive Rate (Sensitivity)", main="ROC Curves")
  } else {
    lines(x, y, col=plot_colors[i])
  }
}
legend("bottomright", legend=names(models), col=plot_colors, lty=1, cex=0.8)
auc_df <- data.frame(Model = names(models), AUC = auc_values)
print(auc_df)
for(model_name in names(models)) {
  model <- models[[model_name]]
  cat("Model:", model_name, "\n")
  if(methods::is(model, "train")) {
    importance <- varImp(model)
    print(importance)
  } else {
    cat("Variable importance not available for this model.\n")
  }
  cat("\n")
}

# Logistic Regression 
logistic_importance <- coef(models$'Logistic Regression'$finalModel)
importance_df <- data.frame(
  Variable = names(logistic_importance),
  Coefficient = unname(logistic_importance)
)
top_variables <- importance_df %>%
  arrange(desc(abs(Coefficient))) %>%
  head(10)
ggplot(top_variables, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity") + 
  coord_flip() +  
  theme_minimal() + 
  xlab("Variable") + 
  ylab("Coefficient") + 
  ggtitle("Top 10 Variable Importance for Logistic Regression") 

# Logistic Regression 
Logistic_Regression_importance <- varImp(models$'Logistic Regression', scale = FALSE)
plot(Logistic_Regression_importance, main = "Variable Importance for Logistic Regression")

# Decision Tree 
tree_importance <- varImp(models$'Decision Tree', scale = FALSE)
plot(tree_importance, main = "Variable Importance for Decision Tree")

# k-NN 
knn_importance <- varImp(models$'k-NN', scale = FALSE)
plot(knn_importance, main = "Variable Importance for k-NN")

# Random Forest 
rf_importance <- varImp(models$'Random Forest', scale = FALSE)
plot(rf_importance, main = "Variable Importance for Random Forest") 

# Gradient Boosting 
gbm_importance <- varImp(models$'Gradient Boosting', scale = FALSE)
plot(gbm_importance, main = "Variable Importance for Gradient Boosting")

# SVM 
svm_importance <- varImp(models$'SVM', scale = FALSE)
plot(svm_importance, main = "Variable Importance for SVM") 


