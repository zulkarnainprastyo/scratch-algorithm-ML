import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(x, weights, bias):
    # Calculate z using the Logistic Regression equation
    z = np.dot(x, weights) + bias

    # Calculate y_pred using the sigmoid function of z
    y_pred = sigmoid(z)

    # Return y_pred (predicted class)
    return y_pred