'''
1. Iris Flower Classification 
Description: Classify iris flowers into three species (setosa, versicolor, virginica) based on features like petal and sepal length/width using classification algorithms.

Steps: 

• Load the Iris dataset (available in sklearn.datasets).
 
• Explore and visualize the data using Pandas and Matplotlib/Seaborn. 

• Split the dataset into training and testing sets. 

• Train a model using a classifier (e.g., Logistic Regression or KNN). 

• Evaluate model accuracy using confusion matrix and classification report.
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()

# loading data in x and y
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = pd.DataFrame(iris.target)

# Visualization / PLoting
sns.pairplot(X, diag_kind='kde', palette='viridis')
plt.suptitle('Pair Plot of Iris Features by Species')
plt.show()

# Training and testing of data
x_train, x_test, y_train, y_test = tts(X, Y, test_size=.25, random_state=12)

model = LogisticRegression()
model.fit(x_train, y_train)

predict_y = model.predict(x_test)

# Determining accuracy
print(f'Accuracy Score: {accuracy_score(y_test, predict_y)}')