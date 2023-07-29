# Classification of Text-based Gene Mutation Evidence

Gene mutation classification is an important aspect of cancer tumour detection, as it helps identify specific genetic mutations that contribute to the development and progression of cancer.

Text evidence plays a crucial role in this process, providing valuable information on the molecular and cellular mechanisms underlying cancer.

Machine learning algorithms can be trained on such text evidence to identify patterns and associations between specific gene mutations and cancer types and predict the functional effects of these mutations on gene expression and cellular pathways.

Manual gene classification is a time-consuming process where pathologists interpret every genetic mutation from the clinical evidence manually. 

The criterion of manual classification is still unknown.

The main aim of this project is to propose a multiclass classifier to classify the genetic mutations based on clinical evidence (i.e., the text description of these genetic mutations) using Natural Language Processing (NLP) techniques.

Three different text vectorizers were used in this project namely - 
  1. CountVectorizer (CV)
  2. TF IDF Vectorizer (TFIDF)
  3. Word2Vec (WC)

Various classifiers used are - 
  1. Logistic Regression (LR)
  2. Random Forest (RF)
  3. XGBoost (XGB)
  4. K Nearest Neighbor (KNN)
  5. Naive Bayes (NB)

Various results were compared in order to find the best-performing classifier only to conclude that a Recurrent Neural Network would work better than all the above-mentioned classifiers.

The dataset used can be downloaded from Kaggle using the following link - 
  https://www.kaggle.com/c/msk-redefining-cancer-treatment/data?select=training_text.zip
  
  https://www.kaggle.com/c/msk-redefining-cancer-treatment/data?select=training_variants.zip

In 'data_changes.py' code file, a third dataset is created. It comprises both of the datasets linked above after preprocessing. Most of the classifiers work on this dataset.



