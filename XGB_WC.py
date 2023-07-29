import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from xgboost import XGBClassifier
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

# Subtract 1 from class labels to range from 0 to 8
y = y - y.min()

# Initialize and fit the XGBoost model
xgb_classifier = XGBClassifier(min_child_weight=5, colsample_bytree=1, random_state=42)
xgb_classifier.fit(X, y)

# Predict the classes
y_pred = xgb_classifier.predict(X)

# Add 1 to class labels to range from 1 to 9 again
y = y + 1
y_pred = y_pred + 1

# Calculate the accuracy score
accuracy = accuracy_score(y, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix - XGBoost (Word2Vec)')
plt.show()

# Individual accuracy scores for each class
accuracy_scores = [41.17, 30.53, 19.1, 48.25, 20.23, 45.79, 72.48, 10.53, 37.48]

# Average accuracy score
average_accuracy = np.mean(accuracy_scores)

# Bar graph
plt.figure(figsize=(8, 6))
plt.bar(range(len(accuracy_scores)), accuracy_scores, color=['#990b0b', '#c47e0e', '#658c0a', '#0e0bba', '#e014da', '#144d02', '#0c8ff2', '#ba02b4', '#02373b'])
plt.xticks(range(len(accuracy_scores)), ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'])
plt.xlabel('Classes')
plt.ylabel('Accuracy Score')

# Individual accuracy scores on top of each bar
for i, v in enumerate(accuracy_scores):
    plt.text(i, v + 5, str(v), ha='center', color='black')

# Average accuracy line
plt.axhline(y=average_accuracy, color='black', linestyle='--')
plt.text(len(accuracy_scores)-0.9, average_accuracy + 5, 'Avg: {:.2f}'.format(average_accuracy), ha='right', color='black')

# Legend
plt.legend(['Avg Accuracy'])

plt.title('Accuracy Scores')
plt.tight_layout()
plt.show()
