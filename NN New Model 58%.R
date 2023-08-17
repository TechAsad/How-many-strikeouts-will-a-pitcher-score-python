# Libraries we will need
library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(keras)
library(imputeTS)
library(DescTools)
# Pitcher strikeout Data for model training and evaluation
data <- read_csv("pitcher_strikeout_data.csv")
Abstract(data)

# we will impute missing values in remaining columns, we are not replacing them with mean.
data <- na_kalman(data)
str(data)

# Here we drop the 'ks' (target feature) from original data to make a separate target data
features <- select(data, -ks)
target <- data$ks



numeric_features <- features[, sapply(features, is.numeric)] #separating numeric feature for the scaling 

#categorical attributes for encoding 
#Each unique category in the original columns 'park', 'p_throws', and 'month' 
#has been converted into binary columns, where a '1' indicates the presence of that category, and '0' indicates the absence.

categorical_features <- features[, sapply(features, function(col) !is.numeric(col))]


#encode categorical variables using one-hot encoding
encoded_data <- model.matrix(~ . - 1, data = categorical_features)


preprocessed_data <- cbind(numeric_features, encoded_data)


#scale numeric features
scaled_features <- scale(preprocessed_data)



# Binary encoding on the target variable
target_encoded <- matrix(0, nrow = length(target), ncol = 8)
thresholds <- c(2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
for (i in 1:length(target)) {
  target_encoded[i, ] <- ifelse(target[i] > thresholds, 1, 0)
}


# We are setting specific thresholds for different levels of strikeouts and representing each pitcher's performance as a series of 0s and 1s.

# For example, let's say we have a pitcher with the following strikeouts:

# Pitcher A: 3 strikeouts
# Pitcher B: 5 strikeouts
# Pitcher C: 8 strikeouts
# Pitcher D: 10 strikeouts
# After applying binary encoding with the given thresholds (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5), the output would look like this:

# Pitcher A: [1, 0, 0, 0, 0, 0, 0, 0]
# Pitcher B: [1, 1, 1, 1, 0, 0, 0, 0]
# Pitcher C: [1, 1, 1, 1, 1, 1, 0, 0]
# Pitcher D: [1, 1, 1, 1, 1, 1, 1, 1]

# Each element in the matrix represents a specific level of strikeouts (e.g., '>2.5', '>3.5', '>4.5', etc.).
# If the pitcher's strikeouts are above the threshold, the corresponding element is set to 1, otherwise,
# it is set to 0. This way, we transform the original numeric target variable into a binary representation,
# making it suitable for the machine learning model to predict specific strikeout levels for each pitcher.

# Define the model
model <- keras_model_sequential()

model <- keras_model_sequential() #run again if there was error in first try

#adding layers
model$add(layer_dense(units = 128, activation = 'relu', input_shape = ncol(scaled_features)))
model$add(layer_dropout(rate = 0.1))
model$add(layer_dense(units = 64, activation = 'relu'))
model$add(layer_dropout(rate = 0.1))
model$add(layer_dense(units = 8, activation = 'softmax'))  # 8 classes

model$layers


model$compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.000005),
  metrics = c('accuracy')
)

# Convert target_encoded to factor
target_encoded_factor <- as.factor(apply(target_encoded, 1, function(row) paste(row, collapse = ",")))

set.seed(42)
split_index <- createDataPartition(target_encoded_factor, p = 0.8, list = FALSE)
X_train <- scaled_features[split_index, ]
y_train <- target_encoded[split_index, ]  # Use target_encoded, not scaled_features
X_test <- scaled_features[-split_index, ]
y_test <- target_encoded[-split_index, ]


history <- model$fit(
   x = X_train, 
  y = y_train, 
  epochs = as.integer(100), 
  batch_size = as.integer(100),
  validation_split = 0.2,
  verbose = 1
  )





test_pred_probs <- model$predict(X_test)
round(test_pred_probs, 3)*100



# Model evaluation on 20% data
evaluation <- model$evaluate(X_test, y_test)
accuracy <- evaluation
cat("Model Evaluation:\nAccuracy:", accuracy, "\n")

#class names to show in table
class_names <- c("2.5", "3.5", "4.5", "5.5", "6.5", "7.5", "8.5", "9.5")

#creating the output table to present probabilities and the mean prediction
output_table <- data.frame(
  Player = rownames(X_test)
)

#adding class probabilities to the output table
for(i in 1:length(class_names)) {
  output_table[[paste0(class_names[i], "_Prob")]] <- round(test_pred_probs[,i] * 100, 3)
}


# table
print(output_table)


model %>% save_model_hdf5("NN_model.h5")




