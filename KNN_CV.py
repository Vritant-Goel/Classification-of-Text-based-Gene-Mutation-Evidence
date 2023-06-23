import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# Convert class labels from 1-9 to 0-8
y_train = y_train - 1
y_test = y_test - 1

# KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions, labels=range(9))

# Calculate accuracy percentage classwise
classwise_accuracy = []
for i in range(9):
    class_samples = y_test[y_test == i]
    class_predictions = knn_predictions[y_test == i]
    class_accuracy = accuracy_score(class_samples, class_predictions)
    classwise_accuracy.append(class_accuracy)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(knn_confusion_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted Class")
plt.ylabel("Class")
plt.xticks(ticks=range(9), labels=range(1, 10))
plt.yticks(ticks=range(9), labels=range(1, 10))
plt.show()

# Plot the bar graph for accuracy
plt.figure(figsize=(10, 6))
bar_colors = ['#990b0b', '#c47e0e', '#658c0a', '#0e0bba', '#e014da', '#144d02', '#0c8ff2', '#ba02b4', '#02373b']
plt.bar(range(1, 10), [acc * 100 for acc in classwise_accuracy], color=bar_colors)
for i, acc in enumerate(classwise_accuracy):
    plt.text(i + 1, (acc * 100) + 1, '{:.2f}%'.format(acc * 100), ha='center', color='black')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy - KNN')
plt.xticks(ticks=range(1, 10))
plt.yticks(ticks=range(0, 101, 10))
plt.axhline(knn_accuracy * 100, color='black', linestyle='--')
plt.legend(['Avg Acc'])
plt.text(8.5, (knn_accuracy * 100) - 5, '{:.2f}%'.format(knn_accuracy * 100), ha='right', color='black')
plt.show()

print("Average Accuracy: {:.2f}%".format(knn_accuracy * 100))
