import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# Load the dataset using pandas, specifying the file name and delimiter as ','
dataset = pd.read_csv('training_variants', delimiter=',')

# Access the individual columns
ids = dataset['ID']
genes = dataset['Gene']
variations = dataset['Variation']
classes = dataset['Class']

# Print the first few rows of the dataset
print(dataset.head())

# Read the text file using pandas
dataset_text = pd.read_csv('training_text', delimiter='\|\|', engine = 'python')

# Access the ID column
ids = dataset_text['ID']

# Access the text column
texts = dataset_text['Text']

print(dataset_text.head())

stop_words = set(stopwords.words('english'))

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        # replace every special char with space
        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)
        # replace multiple spaces with single space
        total_text = re.sub('\s+',' ', total_text)
        # converting all the chars into lower-case.
        total_text = total_text.lower()
        
        for word in total_text.split():
        # if the word is a not a stop word then retain that word from the data
            if not word in stop_words:
                string += word + " "
        
        dataset_text[column][index] = string

#text processing stage.
for index, row in dataset_text.iterrows():
    if type(row['Text']) is str:
        nlp_preprocessing(row['Text'], index, 'Text')
    else:
        print("there is no text description for id:",index)

result = pd.merge(dataset, dataset_text, on='ID', how='left')
print(result.head())

print(result[result['Text'].isnull()])

result.loc[result['Text'].isnull(),'Text'] = result['Gene'] +' '+result['Variation']

# Specify the output file path
output_file_path = 'training_texts'

# Save the merged dataset as a .csv file
result.to_csv(output_file_path, index=False)




