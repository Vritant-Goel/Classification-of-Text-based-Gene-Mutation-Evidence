import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('training_variants', sep = ',')

class_counts = df['Class'].value_counts().sort_index()

total_genes = class_counts.sum()      # Total number of genes for each class

color_list = ['#990b0b', '#c47e0e', '#658c0a', '#0e0bba', '#e014da', '#144d02', '#0c8ff2', '#ba02b4', '#02373b']

plt.bar(class_counts.index, class_counts.values, color=color_list)    # creating bar graph with diff. colors

plt.xticks(class_counts.index)     # Showing the class number for each bar

for i, count in enumerate(class_counts.values):        # Number and % of genes in each class
    percentage = (count / total_genes) * 100
    plt.text(class_counts.index[i], count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')

plt.xlabel('Class')
plt.ylabel('Number of Genes')
plt.title('Distribution of Genes in Each Class')
plt.show()
