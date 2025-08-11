'''
2. House Price Prediction (Linear Regression) 

Description: Predict the price of a house based on features like area, number of bedrooms, location, etc.

Steps: 

• Collect or use a dataset (like the Boston housing dataset or custom CSV). 

• Clean and preprocess the data (handle missing values, encode categories). 

• Use exploratory data analysis to understand correlations. 

• Train a Linear Regression model. 

• Evaluate performance using metrics like Mean Squared Error (MSE) or R² score.
'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Creating a dataset
df = pd.read_csv('Data Analytics\\Practise Problems\\Housing.csv')
# print(df.to_string())

dataset = {
    # 'x_values': [[area, bedrooms, bathrooms, stories]]
    'x_value': [[7420, 4, 2, 3], [8960, 4, 4, 4], [9960, 3, 2, 2]],

    # 'y_values': [[price]]
    'y_values': [[13300000], [12250000], [1225000]]
}
# Loading data in Y and Y
X = pd.DataFrame(dataset['x_value'])
Y = pd.DataFrame(dataset['y_values'])

# Training the model
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=23)

model.fit(x_train, y_train)

# prediction
y_pred = model.predict([[6000, 3, 3, 2]])
print(f'Prediction of the model: {y_pred}')

# Evaluting the performance
mean_sq_er = mean_squared_error(y_test, y_pred)
r2s = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mean_sq_er}")
print(f"R2 Score: {r2s}")