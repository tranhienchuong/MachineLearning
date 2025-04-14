import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Note: Assuming 'marks.txt' is in the same directory as the script
try:
    data = pd.read_csv('marks.txt', header=None, names=['Exam1', 'Exam2', 'Admitted'])
except FileNotFoundError:
    print("Error: 'marks.txt' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if file not found

# Separate features (X) and target (y)
X = data[['Exam1', 'Exam2']]
y = data['Admitted']

# Split the data into training and testing sets (optional but good practice)
# Using 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random_state for reproducibility

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# --- Visualization (Optional) ---

# Create a meshgrid for plotting the decision boundary
h = .02 # step size in the mesh
x_min, x_max = X['Exam1'].min() - 1, X['Exam1'].max() + 1
y_min, y_max = X['Exam2'].min() - 1, X['Exam2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on the meshgrid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plot the training points
sns.scatterplot(x='Exam1', y='Exam2', hue='Admitted', data=data, palette='bright', edgecolor='k')

plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.legend(title='Admitted', loc='upper right')
plt.show()

# Example prediction for a new student
new_student_scores = np.array([[50, 70]]) # Example: Exam1=50, Exam2=70
prediction = model.predict(new_student_scores)
probability = model.predict_proba(new_student_scores)

print(f"\nPrediction for scores {new_student_scores[0]}: {'Admitted' if prediction[0] == 1 else 'Not Admitted'}")
print(f"Probability [Not Admitted, Admitted]: {probability[0]}")