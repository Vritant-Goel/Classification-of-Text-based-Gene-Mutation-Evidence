import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the training dataset
train_data = pd.read_csv('training_texts', sep=',')

# Text transformation model - CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(train_data['Text'])

# Split the training data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(count_matrix, train_data['Class'], test_size=0.5, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=1000, max_depth=20, min_samples_leaf=5, random_state=42)
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(rf_confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.xticks(ticks=range(9), labels=range(1, 10))
plt.yticks(ticks=range(9), labels=range(1, 10))
plt.show()

# Calculate accuracy percentage classwise
class_labels = sorted(train_data['Class'].unique())
classwise_accuracy = []
for label in class_labels:
    accuracy = 100 * accuracy_score(y_test[y_test == label], rf_predictions[y_test == label])
    classwise_accuracy.append(accuracy)

# Calculate average accuracy
average_accuracy = sum(classwise_accuracy) / len(classwise_accuracy)

# Plot the accuracy percentage classwise
plt.figure(figsize=(10, 6))
colors = ['#990b0b', '#c47e0e', '#658c0a', '#0e0bba', '#e014da', '#144d02', '#0c8ff2', '#ba02b4', '#02373b']
plt.bar(range(1, 10), classwise_accuracy, color=colors)
plt.title("Accuracy Percentage - Classwise")
plt.xlabel("Class")
plt.ylabel("Accuracy Percentage")
plt.ylim([0, 100])
plt.xticks(range(1, 10))
for i, acc in enumerate(classwise_accuracy):
    plt.text(i + 1, acc, f"{acc:.2f}%", ha='center', va='bottom', color='black')
plt.axhline(average_accuracy, color='black', linestyle='--', label=f"Average Accuracy: {average_accuracy:.2f}%")
plt.legend()
plt.show()
