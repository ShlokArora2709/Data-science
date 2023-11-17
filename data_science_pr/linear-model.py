import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Sample data for training the linear regression model
data = {
    'Area': [1000, 1200, 1500, 2000, 2500],
    'Price': [200000, 250000, 300000, 350000, 400000]
}

# Sample data for predicting prices
areas_data = {
    'Area': [1500, 1800, 2000, 1600, 2200, 2500, 1800, 2000, 1700, 1900, 2100, 2400, 2000, 1850, 2200]
}

# Create DataFrames for training and prediction
df = pd.DataFrame(data)
dfa = pd.DataFrame(areas_data)

# Function to create a scatter plot of the training data and the regression line
def scatter_plot():
    plt.scatter(df['Area'], df['Price'], c=['blue'], marker='+')
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price (USD)')
    plt.title('House Prices vs Area')
    # Plotting the regression line
    plt.plot(df['Area'], model.predict(df[['Area']]), c='red')
    plt.show()

# Function to perform linear regression on the training data
def linear_reg():
    global model
    model = linear_model.LinearRegression()
    model.fit(df[['Area']], df['Price'])
    # Print the coefficients and intercept of the linear regression model
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)

# Function to predict prices for a given DataFrame
def predictor(dfe):
    # Predict prices using the trained model
    p = model.predict(dfe)
    # Add the predicted prices to the DataFrame
    dfe['Price'] = p

# Display the original training data
print("Training Data:")
print(df)

# Perform linear regression on the training data
linear_reg()

# Use the trained model to predict prices for the additional data
predictor(dfa)

# Display the DataFrame with predicted prices
print("\nPredicted Data:")
print(dfa)

# Create a scatter plot with the regression line
scatter_plot()
