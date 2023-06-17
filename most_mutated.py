import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the text file
df = pd.read_csv('training_variants', sep=',')

# Create an empty dictionary to store the top mutated genes for each class
top_mutated_genes = {}

class_colors = ['#990b0b', '#c47e0e', '#658c0a', '#0e0bba', '#e014da', '#144d02', '#0c8ff2', '#ba02b4', '#02373b']

# Iterate over each class from 1 to 9
for class_num in range(1, 10):
    # Filter the dataset for the current class
    class_data = df[df['Class'] == class_num]
    
    # Calculate the mutation frequencies for each gene
    gene_frequencies = class_data['Gene'].value_counts()
    
    # Select the top five mutated genes
    top_genes = gene_frequencies.head(5).index.tolist()
    
    # Store the top mutated genes for the current class in the dictionary
    top_mutated_genes[class_num] = top_genes

# Create subplots for each class
fig, axs = plt.subplots(3, 3, figsize=(14, 17))
fig.suptitle("Top Mutated Genes by Class", fontsize=14)

# Iterate over each class and create a bar graph
for i, class_num in enumerate(range(1, 10)):
    # Get the top mutated genes for the current class
    top_genes = top_mutated_genes[class_num]
    
    # Filter the dataset for the current class and top genes
    class_data = df[(df['Class'] == class_num) & (df['Gene'].isin(top_genes))]
    
    # Calculate the mutation frequencies for each gene
    gene_frequencies = class_data['Gene'].value_counts()
    
    # Plot the bar graph
    ax = axs[i // 3, i % 3]
    ax.bar(gene_frequencies.index, gene_frequencies.values, color=class_colors[class_num - 1])
    ax.set_title(f"Class {class_num}")
    ax.set_xlabel("Genes")
    ax.set_ylabel("Mutation Frequency")
    ax.tick_params(axis='x')

# Adjust the layout and spacing between subplots
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.subplots_adjust(hspace=0.3)
plt.show()
