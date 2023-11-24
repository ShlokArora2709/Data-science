import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the digits dataset
digits = load_digits()
df = pd.DataFrame(digits.data)
df['target'] = digits.target

def split():
    # Separate features (X) and target variable (Y)
    X = df.drop(['target'], axis=1)
    Y = df['target']
    
    # Initialize stratified k-fold object
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    # Lists to store scores for each model
    score_lr = []
    score_rf = []
    score_svm = []

    # Loop through k-fold splits
    for train_i, test_i in kf.split(X, Y):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = X.iloc[train_i], X.iloc[test_i], Y.iloc[train_i], Y.iloc[test_i]
        
        # Calculate scores for each model and append to respective lists
        score_lr.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
        score_rf.append(get_score(RandomForestClassifier(n_estimators=50), X_train, X_test, y_train, y_test))
        score_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
        
        # Print scores after each fold
        print("Logistic Regression Scores:", score_lr)
        print("Random Forest Scores:", score_rf)
        print("SVM Scores:", score_svm)

def get_score(model, X_train, X_test, y_train, y_test):

    # Fit the model on training data
    model.fit(X_train, y_train)
    
    # Return the accuracy score on test data
    return model.score(X_test, y_test)

# Run the split function
split()
