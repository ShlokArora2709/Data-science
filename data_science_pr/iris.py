import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the Iris dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])

def train_test():
    # Prepare the features (X) and target variable (y)
    X = df.drop(['target', 'flower_name'], axis=1)
    y = df['target']

    # Split the dataset into training and testing sets
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

def training():
    # Initialize the Support Vector Classification model
    model = SVC()

    # Train the model on the training set
    model.fit(X_train, y_train)

    # Evaluate the model's accuracy on the testing set
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

# Call the functions
train_test()
training()
