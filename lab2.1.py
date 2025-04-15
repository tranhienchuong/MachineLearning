import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Using pandas just for easier initial data loading

# --- 1. Sigmoid Function ---
def sigmoid(z):
    """Computes the sigmoid function."""
    # Clip z to prevent overflow in exp(-z) for large negative z
    # and underflow (exp(-z) becoming infinity) for large positive z.
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

# --- 2. Cost Function ---
def compute_cost(X, y, theta):
    """Computes the cost for logistic regression."""
    m = len(y) # number of training examples
    h = sigmoid(X.dot(theta)) # hypothesis (predictions)

    # Add small epsilon to prevent log(0) errors
    epsilon = 1e-5

    # Calculate cost
    cost = -(1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# --- 3. Gradient Descent ---
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """Performs gradient descent to learn theta."""
    m = len(y)
    cost_history = []

    for i in range(num_iterations):
        h = sigmoid(X.dot(theta)) # hypothesis (predictions)
        gradient = (1/m) * X.T.dot(h - y) # gradient vector
        theta = theta - learning_rate * gradient # update theta

        # Calculate and store cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Optional: Print cost every N iterations
        if (i + 1) % 10000 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Cost: {cost:.6f}")

    return theta, cost_history

# --- 4. Prediction Function ---
def predict(X, theta):
    """Predicts binary outcomes (0 or 1) using learned parameters."""
    probabilities = sigmoid(X.dot(theta))
    return [1 if p >= 0.5 else 0 for p in probabilities]

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data using Pandas for convenience, then convert to NumPy
    try:
        # Assumes 'marks.txt' is in the same directory
        data_pd = pd.read_csv('marks.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])
        X_orig = data_pd[['Exam1', 'Exam2']].values # Original features for plotting
        y = data_pd['Admitted'].values   # Target
    except FileNotFoundError:
        print("Error: 'marks.txt' not found. Please ensure the file is in the correct directory.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # 2. Prepare Data for Training
    m = len(y)

    # --- Feature Scaling (Standardization) ---
    # Important for gradient descent convergence
    mean = np.mean(X_orig, axis=0)
    std_dev = np.std(X_orig, axis=0)
    X_scaled = (X_orig - mean) / std_dev
    # -----------------------------------------

    # Add intercept term (column of ones) to the scaled features
    X_b = np.c_[np.ones((m, 1)), X_scaled] # shape (m, n+1) where n=2

    # 3. Initialize Parameters
    initial_theta = np.zeros(X_b.shape[1]) # shape (n+1,)

    # 4. Set Hyperparameters
    learning_rate = 0.1
    num_iterations = 100000

    print("Starting Gradient Descent (manual implementation)...")
    # 5. Run Gradient Descent
    theta, cost_history = gradient_descent(X_b, y, initial_theta, learning_rate, num_iterations)

    print("\nGradient Descent Finished.")
    print("Optimized Parameters (Theta):", theta)
    print("Final Cost:", cost_history[-1])

    # 6. Calculate and Print Accuracy on Training Data
    predictions = predict(X_b, theta)
    accuracy = np.mean(np.array(predictions) == y) * 100
    print(f"\n--- Training Accuracy ---")
    print(f"Accuracy: {accuracy:.2f}%") # Note: This is TRAINING accuracy
    print("------------------------\n")

    # --- 7. Visualization ---
    print("Generating plot...")

    # Create a meshgrid for plotting the decision boundary
    # We need to scale the meshgrid points just like the training data
    h = .02 # step size in the mesh
    x_min, x_max = X_orig[:, 0].min() - 1, X_orig[:, 0].max() + 1
    y_min, y_max = X_orig[:, 1].min() - 1, X_orig[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Prepare meshgrid points for prediction (scale and add intercept)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = (mesh_points - mean) / std_dev
    mesh_points_b = np.c_[np.ones((mesh_points_scaled.shape[0], 1)), mesh_points_scaled]

    # Predict outcomes for each point in the meshgrid
    Z = np.array(predict(mesh_points_b, theta)) # Use the learned theta
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary (filled contour)
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plot the original data points (using original, unscaled X for correct plot locations)
    admitted_indices = np.where(y == 1)
    not_admitted_indices = np.where(y == 0)

    plt.scatter(X_orig[admitted_indices, 0], X_orig[admitted_indices, 1], c='green', marker='o', label='Admitted')
    plt.scatter(X_orig[not_admitted_indices, 0], X_orig[not_admitted_indices, 1], c='red', marker='x', label='Not Admitted')

    # Add plot labels and title
    plt.title('Logistic Regression Decision Boundary (Manual Implementation)')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend(title='Status', loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display the plot
    plt.show()
    print("Plot generated and displayed.")

    # --- 8. Example Prediction ---
    # Predict for a student with Exam1=45, Exam2=85
    example_scores = np.array([[45, 85]])
    # Scale the example scores using the SAME mean and std_dev from training data
    example_scores_scaled = (example_scores - mean) / std_dev
    # Add intercept term
    example_scores_b = np.c_[np.ones((1, 1)), example_scores_scaled]
    # Make prediction
    prediction_prob = sigmoid(example_scores_b.dot(theta))[0]
    prediction_outcome = 1 if prediction_prob >= 0.5 else 0

    print(f"\n--- Example Prediction ---")
    print(f"Prediction for scores [45, 85]:")
    print(f"  Probability of Admission: {prediction_prob:.4f}")
    print(f"  Predicted Outcome: {'Admitted' if prediction_outcome == 1 else 'Not Admitted'}")
    print("------------------------")