# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: ARSHIYA M
RegisterNumber: 212224040029 
*/



import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('student_scores.csv')


X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

print("X (Hours):", *X)
print("Y (Scores):", *Y)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3, random_state=0
)


regressor = LinearRegression()
regressor.fit(X_train, Y_train)


Y_pred = regressor.predict(X_test)

print("Predicted values:", Y_pred)
print("Actual values:", Y_test)


plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, Y_pred, color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()


mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

```

## Output:
<img width="1275" height="140" alt="Screenshot 2025-09-01 093823" src="https://github.com/user-attachments/assets/627d1696-fc3e-4bb3-89ad-114d8f3dec86" />

<img width="782" height="558" alt="Screenshot 2025-09-01 093840" src="https://github.com/user-attachments/assets/172347dc-eab4-4cd9-af85-5ad67ccd7070" />

<img width="802" height="568" alt="Screenshot 2025-09-01 093925" src="https://github.com/user-attachments/assets/e5f70553-87f6-470b-9a68-2a3ba73d4d74" />

<img width="441" height="87" alt="Screenshot 2025-09-01 093938" src="https://github.com/user-attachments/assets/31c16f4a-1118-408f-a8e4-03174f8c408c" />






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
