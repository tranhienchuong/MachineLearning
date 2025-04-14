import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Advertising.csv')
data.columns = data.columns.str.lower()  # đổi toàn bộ tên cột thành chữ thường
X = data[['tv', 'radio', 'newspaper']].values
y = data['sales'].values.reshape(-1, 1)


# Feature normalization
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Add bias (intercept) term
m = len(y)
X_b = np.hstack([np.ones((m, 1)), X_norm])

# Gradient Descent Function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for _ in range(iterations):
        gradient = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Initialize
theta = np.zeros((X_b.shape[1], 1))
alpha = 0.01
iterations = 1000

# Run Gradient Descent
theta_final, cost_history = gradient_descent(X_b, y, theta, alpha, iterations)

# Plot cost over iterations
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()

print("Final parameters:", theta_final)
