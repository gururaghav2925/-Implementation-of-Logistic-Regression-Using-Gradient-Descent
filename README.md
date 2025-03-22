# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Data preprocessing:

3.Cleanse data,handle missing values,encode categorical variables.

4.Model Training:Fit logistic regression model on preprocessed data.

5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.

6.Prediction: Predict placement status for new student data using trained model.

7.End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Guru Raghav Ponjeevith V
RegisterNumber:  212223220027
*/
```
```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
data=pd.read_csv("Placement_Data.csv")
data=data.drop('salary',axis=1) 
data=data.drop('sl_no',axis=1)
data 
```
![image](https://github.com/user-attachments/assets/1b001be1-90a0-448a-97cd-cfbbd08723d1)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['gender']=le.fit_transform(data["gender"])
data["ssc_b"]=le.fit_transform(data["ssc_b"])
data["hsc_b"]=le.fit_transform(data["hsc_b"])
data["hsc_s"]=le.fit_transform(data["hsc_s"])
data["degree_t"]=le.fit_transform(data["degree_t"])
data["workex"]=le.fit_transform(data["workex"])
data["specialisation"]=le.fit_transform(data["specialisation"])
data["status"]=le.fit_transform(data["status"])
data

```
![image](https://github.com/user-attachments/assets/4bbd017d-0e8a-4a5b-8dc7-df6ed906e8db)
```
x=data.iloc[:,:-1].values 
y=data.iloc[:,-1].values
y 


```
![image](https://github.com/user-attachments/assets/3804d421-d4a3-4478-aa3a-346920028454)

```
theta = np.random.randn(x.shape[1]) 
Y=y 
def sigmoid(z): 
   return 1/(1+np.exp(-z))
def loss(theta,X,y): 
   h=sigmoid(X.dot(theta))
   return -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) 
def gradient_descent(theta,X,y,alpha,num_iterations): 
    m=len(y)
    for i in range(num_iterations): 
      h=sigmoid(X.dot(theta)) 
      gradient = X.T.dot(h-y)/m 
      theta-=alpha * gradient
    return theta == gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000) 
def predict(theta,X): 
    h=sigmoid(X.dot(theta)) 
    y_pred=np.where(h>=0.5,1,0) 
    return y_pred 
y_pred = predict(theta,x) 
accuracy=np.mean(y_pred.flatten()==y) 
print("Accuracy: ",accuracy) 
print("Predicted: ",y_pred)
print("Predicts placement for 2 new student :")
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print("Student_1: ",y_prednew) 
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]]) 
y_prednew=predict(theta,xnew) 
print("Student_2: ",y_prednew)

```
## Output:
![image](https://github.com/user-attachments/assets/fe8d8b5d-ecd4-4805-ba07-d1f33360d0c9)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

