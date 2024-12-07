# Sentiment Analysis and Clustering on Product Reviews

This repository contains two parts: **Sentiment Analysis** using Naive Bayes classifiers and **Clustering** using K-Means on a product review dataset. The project involves text preprocessing, feature extraction, classification, and evaluation with various metrics, along with clustering using neural embeddings.

---

## Dataset

The dataset consists of product reviews, star ratings (1, 2, 4, 5), and their corresponding sentiments (0 for negative and 1 for positive).

### Columns Description

- **`text`**: The customer review text.
- **`stars`**: The star rating assigned by the customer (1–5).
- **`sentiment`**: Derived sentiment (0 = Negative, 1 = Positive).

Additionally, embeddings generated using the **`siebert/sentiment-roberta-large-english`** model are provided for clustering tasks.

---

## Part 1: Sentiment Analysis

### Text Preprocessing

- **Removed punctuation** and converted text to lowercase.
- **Tokenized reviews** and removed stop words using the NLTK library.
- Applied **stemming using PorterStemmer** to reduce words to their base form.

### Bag of Words (BoW) Features

- Created **BoW features** using `CountVectorizer` from sklearn.
- Trained a **Multinomial Naive Bayes (MNB)** classifier.

### TF-IDF Features

- Created **TF-IDF features** using `TfidfVectorizer`.
- Trained an **MNB classifier** using these features.

### Performance Metrics

Metrics evaluated using `classification_report`:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

### Observations

- **BoW features** yielded better accuracy (~70%) compared to **TF-IDF features** (~59%).
- Challenges remain in predicting **minority classes** (e.g., 2-star reviews).

---

## Part 2: Clustering

### K-Means Clustering

- Used embeddings from **`siebert/sentiment-roberta-large-english`** for clustering.
- Applied **K-Means clustering** with:
  - **k-means++ initialization.**
  - **Forgy (random) initialization.**
- Plotted the **Elbow Curve** to determine the optimal number of clusters by analyzing **WCSS (Within-Cluster Sum of Squares).**

### Evaluation Metrics

Implemented the following clustering metrics **from scratch**:

- **Purity**: Measures the proportion of correctly classified samples.
- **NMI (Normalized Mutual Information)**: Evaluates mutual information between clusters and ground truth.
- **Rand Index**: Measures the similarity between predicted and ground truth clusters.

---

## Challenges Faced

- **Imbalanced Dataset**: The dataset had a low representation of certain star ratings (e.g., 2 stars), leading to poor performance on minority classes.
- **Preprocessing Complex Text**: Handling a wide variety of linguistic styles, slang, and abbreviations in customer reviews required extensive text preprocessing.
- **Feature Selection**: Determining the best feature representation (BoW vs. TF-IDF) was challenging, given the trade-off between simplicity and informativeness.
- **Clustering Without Labels**: Evaluating the quality of clusters without ground truth for clustering presented difficulties, necessitating the development of custom metrics like **Purity** and **NMI**.

---

## Future Work

- **Enhance Model Performance**: Experiment with advanced NLP models such as **BERT** or **RoBERTa** for improved sentiment classification accuracy.
- **Address Imbalance**: Implement techniques like oversampling, undersampling, or synthetic data generation (e.g., **SMOTE**) to better handle class imbalance.
- **Hybrid Clustering Approaches**: Combine K-Means with hierarchical clustering to improve cluster interpretability and accuracy.
- **Interactive Visualizations**: Develop dashboards to visualize cluster formations and sentiment distributions dynamically.
- **Hyperparameter Tuning**: Optimize K-Means and Naive Bayes classifiers through grid search or other tuning techniques.
- **Scalability**: Adapt the project for larger datasets and real-time prediction pipelines.

---

## Requirements

- **Python 3.x**
- **Libraries**:

```bash
pip install numpy pandas scikit-learn nltk matplotlib
