"""
Iris Classifier Training Script

This script trains a classification model on the Iris dataset,
saves the trained model, and generates a confusion matrix visualization.
"""

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Save the trained model
model_path = 'outputs/iris_model.joblib'
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix - Iris Classifier')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

confusion_matrix_path = 'outputs/confusion_matrix.png'
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
print(f"Confusion matrix saved to: {confusion_matrix_path}")
plt.close()

print("\nTraining complete! Check the outputs/ folder for results.")
