# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sn

# Load the digits dataset
digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

def split():
    # Split the dataset into features (X) and target variable (Y)
    X = df.drop(['target'], axis=1)
    Y = df['target']
    global X_train, X_test, y_train, y_test
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def train():
    global model
    # Initialize and train the RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50)
    model.fit(X_train, y_train)
    # Print the accuracy score on the test set
    print("Accuracy:", model.score(X_test, y_test))

def plot_g():
    y_pred = model.predict(X_test)
    # Generate and display the confusion matrix heatmap
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()

# Execute the functions
split()
train()
plot_g()
