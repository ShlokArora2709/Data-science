import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    'company': ['google', 'google', 'google', 'google', 'google', 'google', 'abc pharma', 'abc pharma', 'abc pharma', 'abc pharma', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook', 'facebook'],
    'job': ['sales executive', 'sales executive', 'business manager', 'business manager', 'computer programmer', 'computer programmer', 'sales executive', 'computer programmer', 'business manager', 'business manager', 'sales executive', 'sales executive', 'business manager', 'business manager', 'computer programmer', 'computer programmer'],
    'degree': ['bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters', 'masters', 'bachelors', 'bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters', 'bachelors', 'masters'],
    'salary_more_then_100k': [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

def preprep():
    # Declare global variables
    global X, Y, X_train, X_test, y_train, y_test

    # Separate features (X) and target variable (Y)
    X = df.drop('salary_more_then_100k', axis='columns')
    Y = df['salary_more_then_100k']

    # Initialize LabelEncoders for categorical variables
    le_com = LabelEncoder()
    le_job = LabelEncoder()
    le_deg = LabelEncoder()

    # Encode categorical variables and replace them in the DataFrame
    X['company_n'] = le_com.fit_transform(X['company'])
    X['job_n'] = le_job.fit_transform(X['job'])
    X['degree_n'] = le_deg.fit_transform(X['degree'])

    # Drop the original categorical columns
    del X['company']
    del X['job']
    del X['degree']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

def train_and_eval():
    # Initialize the Decision Tree model
    model = tree.DecisionTreeClassifier()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Evaluate and print the model accuracy on the test data
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy}")

# Perform data preprocessing
preprep()

# Train and evaluate the Decision Tree model
train_and_eval()
