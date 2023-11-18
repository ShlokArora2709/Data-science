import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data = {
    'Age': [25, 32, 45, 60, 28, 35, 50, 22, 48, 40, 55, 33, 26, 42, 29, 37, 43, 31, 39, 47],
    'Insurance': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)
def modif():
    df['Insurance'] = df['Insurance'].replace({'Yes': 1, 'No': 0})

def scatter_plot():
    plt.scatter(df['Age'], df['Insurance'],c='red',marker='+')
    plt.show()

def train_test():
    X=df[['Age']]
    y=df['Insurance']
    global X_train,X_test,y_train,y_test
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
    
def training():
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    print(X_test)
    print(logreg.predict(X_test))
    print(logreg.score(X_test,y_test))


'''
print(df)
scatter_plot()
'''
modif()
train_test()
training()