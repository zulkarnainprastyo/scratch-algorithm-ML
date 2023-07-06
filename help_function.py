def initialize_params(n):
  beta_0 = 0
  beta_other = [random.random() for _ in range(n)]
  return beta_0, beta_other


# Next function is to compute gradients
import numpy as np

def compute_gradients(x, y, beta_0, beta_other, m, n):
    gradient_beta_0 = 0
    gradient_beta_other = [0] * n

    for i, point in enumerate(x):
        pred = logistic_function_np(point, beta_0, beta_other)
        for j, feature in enumerate(point):
            gradient_beta_other[j] += (pred - y[i]) * feature / m
        gradient_beta_0 += (pred - y[i]) / m

    return gradient_beta_0, gradient_beta_other

def logistic_function_np(point, beta_0, beta_other):
    return 1 / (1 + np.exp(-(beta_0 + point.dot(beta_other))))

def cost_function(x, y, beta_0, beta_other, m):
    total_cost = 0
    for i, point in enumerate(x):
        pred = logistic_function_np(point, beta_0, beta_other)
        cost = y[i] * np.log(pred) + (1 - y[i]) * np.log(1 - pred)
        total_cost += cost
    return -total_cost / m

# Example usage
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1, 0])
m, n = x.shape

beta_0 = 0
beta_other = np.zeros(n)

gradient_beta_0, gradient_beta_other = compute_gradients(x, y, beta_0, beta_other, m, n)
cost = cost_function(x, y, beta_0, beta_other, m)

print("Gradient beta_0:", gradient_beta_0)
print("Gradient beta_other:", gradient_beta_other)
print("Cost:", cost)