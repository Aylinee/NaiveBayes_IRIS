import numpy as np
import pandas as pd
from math import sqrt, pi, exp
import random

# Importing Iris data set
iris = pd.read_csv('C:/Users/xxxx/Desktop/IRIS/Iris.csv')

# Display data
print(iris.head())

# Show classes
print(iris['Species'].unique())

# Data set varieties are 3
print(iris.describe(include='all'))
print(iris.info())

# Remove unneeded column
iris.drop(columns="Id", inplace=True)

# Check if anything is missing
print(iris.isnull().sum())

# Split data into features and target variable
X = iris.iloc[:, 0:4].values
y = iris.iloc[:, 4].values

# Split the data into training and testing sets manually
def train_test_split_manual(X, y, test_size=0.12):  # Change test_size as needed
    indices = list(range(len(X)))
    random.shuffle(indices)
    test_set_size = int(len(X) * test_size)
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# Example: %20 Test Seti ve %80 Eğitim Seti
X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.12)

def calculate_prior(y):
    classes = np.unique(y)
    priors = {c: len(y[y == c]) / len(y) for c in classes}
    return priors

def calculate_mean_and_var(X_train, y_train):
    data = {}
    classes = np.unique(y_train)
    for c in classes:
        X_c = X_train[y_train == c]
        data[c] = [(X_c[:, i].mean(), X_c[:, i].var()) for i in range(X_train.shape[1])]
    return data

def gaussian_pdf(x, mean, var):
    coefficient = 1.0 / sqrt(2.0 * pi * var)
    exponent = exp(- (x - mean) ** 2 / (2 * var))
    return coefficient * exponent

def calculate_likelihood(data, x):
    likelihoods = {}
    for c, stats in data.items():
        likelihood = 1
        for i in range(len(stats)):
            mean, var = stats[i]
            likelihood *= gaussian_pdf(x[i], mean, var)
        likelihoods[c] = likelihood
    return likelihoods

def predict(X, priors, data):
    y_pred = []
    for x in X:
        likelihoods = calculate_likelihood(data, x)
        posteriors = {c: priors[c] * likelihood for c, likelihood in likelihoods.items()}
        y_pred.append(max(posteriors, key=posteriors.get))
    return y_pred

# Calculate priors and likelihoods
priors = calculate_prior(y_train)
data = calculate_mean_and_var(X_train, y_train)

# Make predictions on the test set
y_pred = predict(X_test, priors, data)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

