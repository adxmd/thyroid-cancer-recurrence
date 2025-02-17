"""
Adam David
https://www.adamdavid.dev/


Binary Classification to Predict Recurrence of Thyroid Cancer


Dataset: Differentiated Thyroid Cancer Recurrence
    Numerical Features
                            Age 
    Categorical Features
        Binary:
                            Gender
                            Smoking
                            Hx Smoking
                            Hx Radiothreapy
                            Adenopathy
                            Focality
        Ordinal:
                            Thyroid Function
                            Risk
                            Stage
                            Response
        Nominal:
                            Physical Examination
                            Pathology
                            T
                            N
                            M
Link: https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence

First 16 columns of the csv file are the features
Last column of the csv file is the target we wish to predict

We will create a feedforward neural network for binary classification using Tensorflow with Keras API.
"""

import sklearn.preprocessing
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from collections import defaultdict
import matplotlib.pyplot as plt

def preprocessData(x,y):
    # Convert from yes/no to 1/0, where 1 means the cancer has recurred,
    #                            where 0 means the cancer has not recurred.
    y = y.apply(lambda x: 0 if x == 'No' else 1)

    # Identify numerical and categorical columns
    numerical_features = ['Age']
    categorical_features = x.columns.difference(numerical_features)
    
    # Encode binary categorical features as 0 or 1
    binary_features = [
        'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
        'Adenopathy', 'Focality'
    ]
    for feature in binary_features:
        x[feature] = x[feature].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Ordinal encoding for categories that have an inherent order
    ordinal_features = {
        'Thyroid Function': ['Hypothyroid', 'Euthyroid', 'Hyperthyroid'],
        'Risk': ['Low', 'Intermediate', 'High'],
        'Stage': ['I', 'II', 'III', 'IV'],
        'Response': ['Indeterminate', 'Excellent', 'Biochemical incomplete', 'Structural incomplete']
    }
    for feature, order in ordinal_features.items():
        x[feature] = pd.Categorical(x[feature], categories=order, ordered=True).codes
    
    # List of nominal features
    nominal_features = ['Physical Examination', 'Pathology', 'T', 'N', 'M']

    # Label encode the nominal features
    for feature in nominal_features:
        labelEncoder = sklearn.preprocessing.LabelEncoder()
        x[feature] = labelEncoder.fit_transform(x[feature])
    
    # Standardize the numerical features
    scaler = sklearn.preprocessing.StandardScaler()
    x[numerical_features] = scaler.fit_transform(x[numerical_features])
    
    # Check the shape of processed data
    print(f"Shape after preprocessing: {x.shape}")
    
    return x,y

#   Neural Network built using Keras' Functional API 
def buildModel(learningRate = 0.01):
    # Input layer of size 16, representing the 16 features that are inputted
    input_layer = keras.Input(shape=(16,))

    # Create 4 hidden layers,
    # ReLU activation function
    hidden_layer_1 = keras.layers.Dense(64, activation="relu")  #   64 neurons
    hidden_layer_2 = keras.layers.Dense(32, activation="relu")  #   32 neurons
    hidden_layer_3 = keras.layers.Dense(16, activation="relu")  #   16 neurons
    hidden_layer_4 = keras.layers.Dense(4, activation="relu")   #   4 neurons

    # Output layer has a single neuron for binary classification
    # Sigmoid activation function
    output_layer = keras.layers.Dense(1, activation="sigmoid")  

    # Goes through all 4 hidden layers
    hidden1 = hidden_layer_1(input_layer)
    hidden2 = hidden_layer_2(hidden1)
    hidden3 = hidden_layer_3(hidden2)
    hidden4 = hidden_layer_4(hidden3)
    output = output_layer(hidden4)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    # We are using Stochastic Gradient Descent with lr=0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=learningRate)

    # Compile the model with SGD and Binary Cross-Entropy Loss
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Print a summary of the model
    # print(model.summary())

    # Return the keras model
    return model


# A function that creates a keras model to predict whether a patient has recurrence of thryroid cancer
def thyroid_cancer_recurrence_model(filepath):
  # filepath is the path to an csv file containing the dataset
  data = pd.read_csv(filepath)

  #Inputs
  x = data.iloc[:, :-1]
  #Target
  y = data.iloc[:, -1]
  
  #Preprocess the data, convert from categorical data to numerical data
  x,y = preprocessData(x,y)

  #Divide dataset into train and test sets
  x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)

  #Build the functional model we are going to train
  model = buildModel()

  #Train the model using Keras API
  model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=5, shuffle=True, batch_size=16)

  #Test the model and record it's performance on the test dataset
  performance = model.evaluate(x_test, y_test)
  
  print(performance)
  return model, performance

# thyroid_cancer_recurrence_model('/Users/adamdavid/Desktop/Carleton University/4th Year/Winter \'25/COMP 4107 A - Neural Networks/A2/Thyroid_Diff.csv')




def TCR_experiments(filepath):
    # filepath is the path to an csv file containing the dataset
    data = pd.read_csv(filepath)

    #Inputs
    x = data.iloc[:, :-1]
    #Target
    y = data.iloc[:, -1]
    
    #Preprocess the data, convert from categorical data to numerical data
    x,y = preprocessData(x,y)

    #Divide dataset into train and test sets
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
    
    results = defaultdict()

    epochCounts = [5,10,15,20,25,30,35,40]
    learningRates = [0.001, 0.005,0.01, 0.05, 0.1, 0.5]

    for learningRate in learningRates:
        for epoch in epochCounts:
            #Build the functional model we are going to train
            model = buildModel(learningRate)

            #Train the model using Keras API
            model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=epoch, shuffle=True, batch_size=16)

            #Test the model and record it's performance on the test dataset
            performance = model.evaluate(x_test, y_test)

            results[f'{learningRate},{epoch}'] = performance
    
    print("-------------------------")
    print(results)
    print("-------------------------")
    for learningRate in learningRates:
        for epoch in epochCounts:
            print(f"---------------------------\nLearning Rate: {learningRate}, Epochs: {epoch} \nAccuracy:{results[f'{learningRate},{epoch}'][1]} \nLoss:{results[f'{learningRate},{epoch}'][0]} \n---------------------------")
  
    # Colors for each learning rate
    colors = {
        0.001: 'blue',
        0.005: 'pink',
        0.01: 'green',
        0.05: 'cyan',
        0.1: 'orange',
        0.5: 'red'
    }

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy Plot
    for lr in learningRates:
        accuracies = [results[f"{lr},{epoch}"][1] for epoch in epochCounts]
        ax1.plot(epochCounts, accuracies, marker='o', color=colors[lr], label=f'LR={lr}')
    ax1.set_title('Accuracy vs Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss Plot
    for lr in learningRates:
        losses = [results[f"{lr},{epoch}"][0] for epoch in epochCounts]
        ax2.plot(epochCounts, losses, marker='o', color=colors[lr], label=f'LR={lr}')
    ax2.set_title('Loss vs Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    return

TCR_experiments('/Users/adamdavid/Desktop/Carleton University/4th Year/Winter \'25/COMP 4107 A - Neural Networks/A2/Thyroid_Diff.csv')
