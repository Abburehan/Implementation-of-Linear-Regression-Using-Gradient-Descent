# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.
2. Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.
3. Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.
4. Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.
5. Display the predicted value for the target variable based on the linear regression model applied to the new data. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ABBU REHAN
RegisterNumber:  212223240165
*/
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term 
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta

data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding column 
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
## Output:
![ml ex03 1 1](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/44947333-f0ca-4b26-9e89-d2f89874fff0)
![ml ex03 1 2](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/d55ca2e9-b4f3-4e9a-89c8-5f09b2f4fb2a)
![ml ex03 1 3](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/26f2e5d7-a4be-4391-8e87-194cd0dece47)
![ml ex03 1 4](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/0c630e8c-efc5-49b0-8189-bdae5066762b)
![ml ex03 1 5](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/faa95ea7-4bc8-4cb3-a200-7c1cea9d6a79)
![ml ex03 1 6](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/4f46fb6d-af4a-4ba1-b41d-17d707d3e5bf)
![ml ex03 1 7](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/6d56b260-07eb-483e-830d-2ea98eef6c6c)
![ml ex03 1 8](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/143b855d-58d1-4f28-ac2d-677513b95f46)
![ml ex03 1 9](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/9b23de51-338c-4cb6-a499-7197870c97a8)
![ml ex03 1 10](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/8ae9ea4d-3172-41a7-8bef-92b5d89e8cce)
![ml ex03 1 11](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/2f96001c-9103-4137-bbd0-d82d5bd5b154)
![ml ex03 1 12](https://github.com/Abburehan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138849336/45d9a751-06c2-425f-8318-5de0084d6732)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
