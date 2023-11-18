import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample data for logistic regression
data = {
    'Age': [25, 32, 45, 60, 28, 35, 50, 22, 48, 40, 55, 33, 26, 42, 29, 37, 43, 31, 39, 47],
    'Insurance': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to convert 'Yes' to 1 and 'No' to 0 in the 'Insurance' column
def modif():
    df['Insurance'] = df['Insurance'].replace({'Yes': 1, 'No': 0})

# Function to create a scatter plot of Age vs Insurance
def scatter_plot():
    colors = np.where(df['Insurance'] == 1, 'green', 'red')  # Green for 'Yes', Red for 'No'
    plt.scatter(df['Age'], df['Insurance'], c=colors, marker='+')
    plt.xlabel('Age')
    plt.ylabel('Insurance (1: Yes, 0: No)')
    plt.title('Scatter Plot of Age vs Insurance')
    plt.show()

# Function to split the data into training and testing sets
def train_test():
    X = df[['Age']]
    y = df['Insurance']
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Function to train the logistic regression model and print predictions and accuracy
def training():
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    predictions = logreg.predict(X_test)
    accuracy = logreg.score(X_test, y_test)

    print("Predictions:", predictions)
    print("Accuracy:", accuracy)

# Display the original DataFrame
print(df)

# Create and display the scatter plot
scatter_plot()

# Modify the 'Insurance' column
modif()

# Split the data into training and testing sets
train_test()

# Train the logistic regression model and display predictions and accuracy
training()
