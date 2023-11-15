# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step-1:
Import the libraries and read the data frame using pandas.
### Step-2:
Calculate the null values present in the dataset and apply label encoder.
### Step-3:
Determine test and training data set and apply decison tree regression in dataset.
### Step-4:
calculate Mean square error,data prediction and r2.

## Program:

/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: premji p
RegisterNumber: 212221043004
*/


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])


## Output:
### data.head():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/413cec6d-79f8-4ac6-8c52-940805bc6a97)

### data.info():
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/27201e42-3aa2-441b-bb6f-19686ebb12de)

### isnull() & sum() function:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/4f8725f4-f588-4926-9b3f-2b637867f302)

### data.head() for Position:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/e45f56d2-865b-477b-9f22-0afe6f854129)

### MSE value:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/38532341-f063-47da-872e-f33acd36c7b9)

### R2 value:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/5c4cde32-af52-4c64-b34e-fed25bfee130)

### Prediction value:
![image](https://github.com/Yogabharathi3/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118899387/a12a6b1e-fec1-44e2-8864-6bcb773d9f3c)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming
