import numpy as np

def logistic_regression(X, y, learning_rate=0.01, num_iterations=1000):
    # Initialize weights and bias
    n_features = X.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    
    # Define the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Perform gradient descent
    m = X.shape[0]  # Number of training examples
    for _ in range(num_iterations):
        # Calculate z using the Logistic Regression equation
        z = np.dot(X, weights) + bias

        # Calculate y_pred using the sigmoid function of z
        y_pred = sigmoid(z)

        # Calculate the cost function (Cross Entropy Loss)
        cost = (-1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # Calculate the gradients of the cost function with respect to weights and bias
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Update the weights and bias using the Gradient Descent algorithm
        weights = weights - learning_rate * dw
        bias -= learning_rate * db

    # Return the trained weights and bias
    return weights, bias

# Example usage
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])

# Call the logistic_regression function
trained_weights, trained_bias = logistic_regression(X, y)

# Print the trained weights and bias
print("Trained weights:", trained_weights)
print("Trained bias:", trained_bias)
