# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the Iris Dataset

Step 3. Separate features (x) and labels (y), then split the data into training and test sets using train_test_split().

Step 4. Initialize an SGDClassifier and fit it on the training data (x_train, y_train).

Step 5. Use the trained model to predict the labels for the test set and calculate the accuracy score.

Step 6. Generate and print a confusion matrix to evaluate the model's performance.

Step 7. End
## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: SANDHIYA M
RegisterNumber:  212224220086
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
x=df.drop('target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(x_train,y_train)
y_pred=sgd_clf.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
X and Y Values:

<img width="778" height="291" alt="491897310-44c273f9-3fce-427d-8a8b-f7a026e28f89" src="https://github.com/user-attachments/assets/9402715f-cb1c-4527-b0f6-42b4143ef4a9" />

Accuracy

<img width="171" height="25" alt="491897429-feaec97e-9ce9-4137-b64b-c7df3e752778" src="https://github.com/user-attachments/assets/5cf5b4cd-abc5-4692-8053-fb973f35a508" />


confusion matrix

<img width="178" height="95" alt="491897462-a1a1c99a-ba31-45b2-88c6-c50799f923b9" src="https://github.com/user-attachments/assets/c4c1e8f9-87ec-4989-a58f-cded0c4cfa72" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
