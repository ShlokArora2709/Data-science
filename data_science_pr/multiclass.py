import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the digits dataset
digits = load_digits()

def plot_digit(index):
    
    #Plot a digit from the dataset.
    
    plt.gray()
    plt.matshow(digits.images[index])
    plt.show()

def split_data():
    
    #Split the digits dataset into training and testing sets.
    
    X = digits.data
    y = digits.target
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def train_model():

    #Train a logistic regression model and print the accuracy.
    model = LogisticRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print("Model Accuracy:", accuracy)

# Plot a digit from the dataset (optional)
# plot_digit(0)

# Split the data into training and testing sets
split_data()

# Train the logistic regression model and print accuracy
train_model()
