'''
4. Handwritten Digit Recognition 

Description: Classify digits (0–9) from images of handwritten digits using ML. 

Steps: 

• Load the digits dataset from sklearn.datasets or use MNIST. 

• Normalize image pixel values. 

• Train a classifier (like RandomForest or SVM). 

• Test with a separate test set. 

• Display predictions with actual digit images using matplotlib. 
'''

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# loading dataset
digits = load_digits()
X = digits.data
Y = digits.target

# Splitting the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=23)

# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=23)
model.fit(x_train, y_train)

# Testing
y_pred = model.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}') #Accuracy to understand how accurate our model is.

# Visualization with matplotlib
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(10, 4))
for ax, image, actual, pred in zip(axes, x_test, y_test, y_pred):
    ax.imshow(image.reshape(8, 8), cmap='gray')
    ax.set_title(f"Actual: {actual}\nPred: {pred}")
    ax.axis('off')
plt.suptitle('Handwritten Digits: Actual vs. Predicted')
plt.show()
