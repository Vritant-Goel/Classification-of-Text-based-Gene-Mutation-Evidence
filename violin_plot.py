import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Specify the path to your data file
data_file_path = "E:\\gene mutation\\Gene-Mutation-Classification\\training_text.csv"

# Load the dataset using pandas
dataset = pd.read_csv(data_file_path)

# Tokenize the text data
tokenized_dataset = []
for text in dataset['Text']:  # Replace 'text_column' with the actual column name in your dataset
    tokens = word_tokenize(text)
    tokenized_dataset.append(tokens)

# Separate the class labels
class_labels = dataset['Class']  # Replace 'class_column' with the actual column name in your dataset

# Calculate the number of words in each row
word_counts = [len(tokens) for tokens in tokenized_dataset]

# Create a new DataFrame with class labels and word counts
data = pd.DataFrame({'Class': class_labels, 'Word Count': word_counts})

# Create a violin plot using seaborn
sns.violinplot(x='Class', y='Word Count', data=data)

# Set labels and title
plt.xlabel('Class')
plt.ylabel('Number of Words')
plt.title('Number of Words in Each Class')

# Show the plot
plt.show()
