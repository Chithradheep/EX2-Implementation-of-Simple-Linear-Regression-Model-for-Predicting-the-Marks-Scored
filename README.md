# EX2 Implementation of Simple Linear Regression Model for Predicting the Marks Scored
## DATE:
## AIM:
To implement simple linear regression using sklearn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Get the independent variable X and dependent variable Y by reading the dataset.
2. Split the data into training and test data.
3. Import the linear regression and fit the model with the training data.
4. Perform the prediction on the test data.
5. Display the slop and intercept values.
6. Plot the regression line using scatterplot.
7. Calculate the MSE.

## Program:
```
Program to implement univariate Linear Regression to fit a straight line using least squares.
Developed by: Chithradheep R
RegisterNumber: 2305002003
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ex1.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,Y_train)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x_train,lr.predict(x_train),color='red')
m=lr.coef_
m
b=lr.intercept_
b
pred=lr.predict(X_test)
pred
X_test
Y_test
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test, pred)
print(f'Mean Squared Error (MSE): {mse}')

```

## Output:

![image](https://github.com/user-attachments/assets/addcb517-4910-4041-995c-14d9f44a64e5)
![image](https://github.com/user-attachments/assets/a024ce6b-0b4d-442c-8165-c999079c3c77)
![image](https://github.com/user-attachments/assets/0defcb46-f4a7-4437-91d7-8cbdc1605ebd)
![image](https://github.com/user-attachments/assets/83095545-ac48-4e3d-9a44-bf70d1815d87)
![image](https://github.com/user-attachments/assets/9245895b-5b6b-445c-96d6-f3e1080634f1)
![image](https://github.com/user-attachments/assets/c3315e54-e207-478d-91e1-c2b24492ebbb)
![image](https://github.com/user-attachments/assets/04ee10a0-9a7e-42e8-a58f-1cd8a6c0d256)



## Result:
Thus the univariate Linear Regression was implemented to fit a straight line using least squares using python programming.
