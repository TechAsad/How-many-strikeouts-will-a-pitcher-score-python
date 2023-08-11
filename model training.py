#Librarires we will need
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#Pitcher strikeout Data for model training and evaluation 
data = pd.read_csv("pitcher_strikeout_data.csv") 

#here we droped the 'ks' (target feature) from original data to make a seprate target data
      
features = data.drop("ks", axis=1)
target = data["ks"]

#identifying the missing values
features.replace('NA', np.nan, inplace=True)

#we need to transform the non-numeric features such as 'park', 'p_throws' to 
categorical_cols = ['park', 'p_throws', 'month']

#encoding the non-numeric columns: 'park', 'p_throws', and 'month'
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cols = []
for col in categorical_cols:
    encoded_col = encoder.fit_transform(features[col].values.reshape(-1, 1))
    encoded_cols.append(pd.DataFrame(encoded_col, columns=[f"{col}_{i}" for i in range(encoded_col.shape[1])]))
features_encoded = pd.concat([features] + encoded_cols, axis=1).drop(categorical_cols, axis=1)

#Each unique category in the original columns 'park', 'p_throws', and 'month' 
#has been converted into binary columns, where a '1' indicates the presence of that category, and '0' indicates the absence.

#instead of dropping or putting the mean value, we will impute missing values using 'Iterative Imputer', it is
# a model which predicts the missing value and replace it.
imputer = IterativeImputer(max_iter=10, random_state=42)
features_imputed = imputer.fit_transform(features_encoded)

#to have feature in one scale for better training of the mode, we will scale the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

#binary encoding on the target variable
target_encoded = target.apply(lambda x: [1 if x > 2.5 else 0,
                                         1 if x > 3.5 else 0,
                                         1 if x > 4.5 else 0,
                                         1 if x > 5.5 else 0,
                                         1 if x > 6.5 else 0,
                                         1 if x > 7.5 else 0,
                                         1 if x > 8.5 else 0,
                                         1 if x > 9.5 else 0])

# Convert the lists to NumPy arrays
target_encoded = np.vstack(target_encoded)


#We are setting specific thresholds for different levels of strikeouts and representing each pitcher's performance as a series of 0s and 1s.

#For example, let's say we have a pitcher with the following strikeouts:

#Pitcher A: 2 strikeouts
#Pitcher B: 5 strikeouts
#Pitcher C: 7 strikeouts
#Pitcher D: 9 strikeouts
#After applying binary encoding with the given thresholds (2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5), the output would look like this:


#Pitcher A: [1, 0, 0, 0, 0, 0, 0, 0]
#Pitcher B: [1, 1, 1, 1, 0, 0, 0, 0]
#Pitcher C: [1, 1, 1, 1, 1, 1, 0, 0]
#Pitcher D: [1, 1, 1, 1, 1, 1, 1, 1]


#Each element in the list represents a specific level of strikeouts (e.g., '>2.5', '>3.5', '>4.5', etc.). 
#If the pitcher's strikeouts are above the threshold, the corresponding element is set to 1, otherwise, 
#it is set to 0. This way, we transform the original numeric target variable into a binary representation, 
#making it suitable for the machine learning model to predict specific strikeout levels for each pitcher.

#no we will createe the a neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="softmax")  # Output layer with 8 neurons for 8 different outcomes '>2.5', '>3.5', '>4.5', '>5.5', '>6.5', '>7.5', '>8.5', '>9.5'
])

#In this part, we are defining the architecture of the neural network model. It consists of three layers: 
#the first hidden layer with 32 neurons and the ReLU activation function, 
#the second hidden layer with 16 neurons and the ReLU activation function, and the output layer with 8 neurons and the softmax activation function.

#compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


#loss: The loss function that the model will use during training to measure how well it is performing. 
# In this case, we are using "binary_crossentropy" as the loss function since we are performing binary classification with multiple output neurons.
#optimizer: The optimizer is the algorithm used to update the weights of the neural network during training. 
# "adam" is a popular optimizer that adapts the learning rate during training to improve convergence.
#metrics: The metrics are used to evaluate the performance of the model. In this case, we are using "accuracy" as the metric, 
# which measures the percentage of correctly predicted samples during training and evaluation.

#data into training and testing sets for model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_encoded, test_size=0.2, random_state=42) #training 80% and testing 20%

model.fit(X_train, y_train, epochs=100, batch_size=38, verbose=1) #finally the model training: here we will train the model on training data (80%)

#model evaluation on 20% data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model Evaluation:\nAccuracy: {accuracy}") #we will get the overall accuracy of the model

#saving mode for future usage on new pitcher data
model.save("neural_network_model")


