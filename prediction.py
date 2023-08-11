import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# new data, on wchich we need to make predictions (without 'ks')
new_data = pd.read_csv("new_pitcher_data.csv")  

new_features = new_data.copy()

#all the preprocessing is same as in model training 
new_features.replace('NA', np.nan, inplace=True)

categorical_cols = ['park', 'p_throws', 'month']
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_cols = []
for col in categorical_cols:
    encoded_col = encoder.fit_transform(new_features[col].values.reshape(-1, 1))
    encoded_cols.append(pd.DataFrame(encoded_col, columns=[f"{col}_{i}" for i in range(encoded_col.shape[1])]))
new_features_encoded = pd.concat([new_features] + encoded_cols, axis=1).drop(categorical_cols, axis=1)


imputer = IterativeImputer(max_iter=10, random_state=42)
new_features_imputed = imputer.fit_transform(new_features_encoded)

scaler = StandardScaler()
new_features_scaled = scaler.fit_transform(new_features_imputed)

#Here we will load the model file that we saved in training model code
model = tf.keras.models.load_model("neural_network_model") # only provide the name if this .py file is in same folder

#new predictions
new_predictions = model.predict(new_features_scaled)

#possible outcomes' array from >2.5 to >9.5
possible_outcomes = np.array(['>2.5', '>3.5', '>4.5', '>5.5', '>6.5', '>7.5', '>8.5', '>9.5'])

#mean predicted strikeouts for each player
mean_predicted_strikeouts = np.dot(new_predictions, np.arange(2.5, 10.5, 1.0)) #starts from 3.0 (>2.5) to 11 (>9.5)

#The np.arange(2.0, 10.0, 1.0) creates an array representing the possible outcomes from 2.0 to 9.0 with a step size of 1.0. 
# These are the strikeout values associated with each outcome.

#The np.dot(new_predictions, np.arange(2.0, 10.0, 1.0)) performs a dot product between the model's predictions and the array of possible outcomes.

#probabilities of each strikeout class for each player
class_probabilities = {}
for i, player_prediction in enumerate(new_predictions):
    class_probabilities[f"Player_{i+1}"] = dict(zip(possible_outcomes.astype(str), player_prediction))

#creating DataFrames for mean strikout and class probabilities
output_df = pd.DataFrame({"Player": [f"Player_{i+1}" for i in range(len(mean_predicted_strikeouts))],
                          "Mean Strikeout": mean_predicted_strikeouts})

probabilities_df = pd.DataFrame.from_dict(class_probabilities, orient="index")

#prediction/outputs in a table format
print("Mean Strikeout:")
print(output_df)
print("\nProbabilities of Possible Strikeouts:")
print(probabilities_df)
