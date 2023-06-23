import pandas as pd
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
data = pd.read_csv('training_texts', sep = ',')

# Tokenize your text data into sentences
sentences = [text.split() for text in data['Text']]

# Train the Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=100)

# Create the document vectors by averaging word vectors
document_vectors = []
for sentence in sentences:
    vectors = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
    if vectors:
        document_vector = sum(vectors) / len(vectors)
    else:
        document_vector = [0] * 100  # Use zero vector for sentences without any known words
    document_vectors.append(document_vector)

# Convert the document vectors to a DataFrame
X = pd.DataFrame(document_vectors)
y = data['Class']

# Initialize and fit the Random Forest model
random_forest = RandomForestClassifier(max_depth=5, min_samples_leaf=5, random_state=42)
random_forest.fit(X, y)

# Predict the classes
y_pred = random_forest.predict(X)

# Calculate the accuracy score
accuracy = accuracy_score(y, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix - Random Forest (Word2Vec)')
plt.show()

# Calculate the accuracy score for each class
class_accuracy = [accuracy_score(y[y == cls], y_pred[y == cls]) for cls in range(1, 10)]

# Define custom colors for individual bars
colors = ['#990b0b', '#c47e0e', '#658c0a', '#0e0bba', '#e014da', '#144d02', '#0c8ff2', '#ba02b4', '#02373b']

# Plot the accuracy score for each class
plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, 10), class_accuracy, color=colors)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy Score - Random Forest (Word2Vec)')
plt.ylim(0, 1)
plt.xticks(range(1, 10))

# Add individual accuracy scores on top of the bars
for i, acc in enumerate(class_accuracy):
    plt.text(i + 1, acc, f'{acc:.2%}', ha='center', va='bottom', color='black')

# Add the average accuracy to the legend
plt.axhline(y=accuracy, color='black', linestyle='--', label=f'Average Accuracy: {accuracy:.2%}')
plt.legend()

plt.show()
