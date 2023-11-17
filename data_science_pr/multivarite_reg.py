import pandas as pd
import numpy as np
from sklearn import linear_model
import joblib

# Sample data for the house price prediction model
data = {
    'Area': [1200, 1500, 1800, 2000, 1300, 1600, 1900, 2200, 1700, 2000],
    'Bedrooms': [2, 3, 2, 4, 3, 2, 3, 4, 2, 3],
    'Age': [5, 8, 10, 2, 6, 3, 7, 9, 4, 8],
    'Price': [250000, 300000, 350000, 400000, 280000, 320000, 380000, 450000, 330000, 400000]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data, columns=['Area', 'Bedrooms', 'Age', 'Price'])

# Function for training the linear regression model
def model_training():
    global reg
    reg = linear_model.LinearRegression()
    reg.fit(df[['Area', 'Bedrooms', 'Age']], df['Price'])

# Function for predicting the price based on user input
def predicting_price():
    area = int(input("Enter the area of the house: "))
    bedroom = int(input("Enter the number of bedrooms in the house: "))
    age = int(input("Enter the age of the house: "))
    # Use the trained model to predict the price
    prediction = reg.predict([[area, bedroom, age]])
    return round(prediction[0])

# Function for saving the trained model to a file
def dump_model():
    joblib.dump(reg, 'house_area_pr')

# Function for loading a previously trained model from a file
def load_model():
    loaded = joblib.load('house_area_pr')
    return loaded

if __name__ == "__main__":
    # Train the model
    model_training()
    
    # Get a price prediction from the user
    price = predicting_price()
    print("Predicted Price is:", price)
    
    # Save the trained model to a file
    dump_model()
