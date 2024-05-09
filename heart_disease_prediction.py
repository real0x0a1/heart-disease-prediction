#!/bin/python3

# -*- Author: real0x0a1 (Ali) -*-
# -*- File: heart_disease_prediction.py -*-

# import libraries
import numpy as np
import pandas as pd

# import scikit-learn libraries for model selection, logistic regression, and metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the csv data to a Pandas DataFrame
heart_data = pd.read_csv('./content/data.csv')

# check the distribution of the Target Variable (heart disease presence)
heart_data['target'].value_counts()

# separate the features (X) from the target variable (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# create a Logistic Regression model
model = LogisticRegression()

# train the Logistic Regression model with the training data
model.fit(X_train, Y_train)

# evaluate the model's accuracy on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# evaluate the model's accuracy on the testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

# input data for prediction (replace with your own values)
input_data = (63,   # age
              1,    # sex
              3,    # cp
              140,  # trestbps
              268,  # chol
              0,    # fbs
              0,    # restecg
              160,  # thalach
              0,    # exang
              3.6,  # oldpeak
              0,    # slope
              2,    # ca
              2     # thal
            )

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')