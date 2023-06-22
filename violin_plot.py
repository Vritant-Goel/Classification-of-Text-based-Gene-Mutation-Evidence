import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize

# Specify the file path
data_file_path = 'training_texts'

# Read the data from the file
data = pd.read_csv(data_file_path)

# Preprocess the text by tokenizing and removing punctuation
data['Text'] = data['Text'].apply(lambda x: ' '.join(word_tokenize(str(x))))

# Calculate the word counts for each class
class_word_counts = data.groupby('Class')['Text'].apply(lambda x: len(str(x).split())).reset_index()
class_word_counts.columns = ['Class', 'Word_Count']

# Print the word counts for each class
print(class_word_counts)

# Create the violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(x='Class', y='Word_Count', data=class_word_counts, inner='quartile')

# Customize the plot
plt.title('Distribution of Word Counts by Class')
plt.xlabel('Class')
plt.ylabel('Number of Words')
plt.xticks(rotation=45)

# Show the plot
plt.show()
