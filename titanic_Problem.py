'''
3. Titanic Survival Prediction 

Description: Predict whether a passenger survived the Titanic disaster based on features like 
age, class, gender, etc. 

Steps:

• Load the Titanic dataset (available on Kaggle or Seaborn). 

• Perform data cleaning and feature engineering. 

• Convert categorical variables using One-Hot Encoding. 

• Train a Logistic Regression model. 

• Evaluate using accuracy, precision, recall, and confusion matrix. 
'''

import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df = pandas.read_csv("Data Analytics\\Practise Problems\\huge_1M_titanic.csv\\huge_1M_titanic.csv")

# Using One-Hot encoding
(df['Age'].fillna(df['Age'].median(), inplace=True))
df = pandas.get_dummies(df, columns=['Sex'], drop_first=True)

df.drop(columns=['Name', 'SibSp', 'Parch', 'Fare', 'Ticket', 'Embarked', 'Cabin', 'PassengerId', 'Pclass'], inplace=True)
# print(df.isnull().sum())

# Training the model
# X = pandas.DataFrame(dataset[['Age', 'Sex', 'Pclass']])
X = df.drop('Survived', axis=1)
Y = df['Survived']

x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=23)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Evaluting the performance
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"Recall Score: {recall_score(y_test, y_pred)}")
print(f"Precision Score: {precision_score(y_test, y_pred)}")
print(f"Confusion Matrix: {confusion_matrix(y_test, y_pred)}")
