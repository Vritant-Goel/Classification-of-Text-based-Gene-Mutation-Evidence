import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

# Read the text file
data = pd.read_csv('training_texts', sep=',')

# Assuming you have a class column in your dataset
your_class_column = data['Class']

# Create an instance of CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform the data to get word counts
word_counts = vectorizer.fit_transform(data['Text'])

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame to store the word frequency data
top_words = pd.DataFrame(word_counts.toarray(), columns=feature_names)

# Add a column for the class
top_words['Class'] = your_class_column

# Calculate the total count of words in each class
class_word_counts = top_words.groupby('Class').sum()

# Calculate the frequency percentage of each word in each class
top_words.iloc[:, :-1] = (top_words.iloc[:, :-1] / class_word_counts) * 100

# Get the top 10 most frequent words for each class
top_10_words = top_words.iloc[:, :-1].apply(lambda row: row.nlargest(10), axis=1)

# Plot the heatmap
sns.heatmap(top_10_words, annot=True, fmt='.2f', cmap='YlGnBu', cbar=False)

# Customize the plot labels and title
plt.xlabel('Frequency Percentage Range')
plt.ylabel('Class')
plt.title('Top 10 Most Frequent Words for Each Class')

# Display the plot
plt.show()
