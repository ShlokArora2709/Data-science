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
df= pd.DataFrame(digits.data)
df['target']=digits.target


def split():
    X=df.drop(['target'],axis=1)
    Y=df['target']
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

def train():
    global model
    model=RandomForestClassifier(n_estimators=50)
    model.fit(X_train,y_train)
    model.score(X_test,y_test)

def plot_g():
    y_pred=model.predict(X_test)
    cm =confusion_matrix(y_true=y_test,y_pred=y_pred)
    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('Actual Labels')
    plt.show()



split()
train()
plot_g()