Dataset

The dataset consists of product reviews, star ratings (1, 2, 4, 5), and their corresponding sentiments (0 for negative and 1 for positive).
Column	Description
text	The customer review text.
stars	The star rating assigned by the customer (1–5).
sentiment	Derived sentiment (0 = Negative, 1 = Positive).

Additionally, embeddings generated using the siebert/sentiment-roberta-large-english model are provided for clustering tasks.
Part 1: Sentiment Analysis
Text Preprocessing

    Removed punctuation and converted text to lowercase.
    Tokenized reviews and removed stop words using the NLTK library.
    Applied stemming using PorterStemmer to reduce words to their base form.

Bag of Words (BoW) Features

    Created BoW features using CountVectorizer from sklearn.
    Trained a Multinomial Naive Bayes (MNB) classifier.

TF-IDF Features

    Created TF-IDF features using TfidfVectorizer.
    Trained an MNB classifier using these features.

Performance Metrics

Metrics evaluated using classification_report:

    Accuracy
    Precision
    Recall
    F1-Score

Observations:

    BoW features yielded better accuracy (~70%) compared to TF-IDF features (~59%).
    Challenges remain in predicting minority classes (e.g., 2-star reviews).

Part 2: Clustering
K-Means Clustering

    Used embeddings from siebert/sentiment-roberta-large-english for clustering.
    Applied K-Means clustering with:
        k-means++ initialization.
        Forgy (random) initialization.
    Plotted the Elbow Curve to determine the optimal number of clusters by analyzing WCSS (Within-Cluster Sum of Squares).

Evaluation Metrics

Implemented the following clustering metrics from scratch:

    Purity: Measures the proportion of correctly classified samples.
    NMI (Normalized Mutual Information): Evaluates mutual information between clusters and ground truth.
    Rand Index: Measures the similarity between predicted and ground truth clusters.

Requirements

    Python 3.x
    Libraries:

    pip install numpy pandas scikit-learn nltk matplotlib

How to Run

    Clone the repository:

git clone https://github.com/Archit-10/Sentiment-Analysis-Using-Naive-Bayes.git
cd sentiment-analysis-clustering

Install dependencies:

pip install -r requirements.txt

Run Sentiment Analysis:

python sentiment_analysis.py

Run Clustering:

    python clustering.py

Results and Observations
Sentiment Analysis

    BoW Accuracy: ~70%
    TF-IDF Accuracy: ~59%
    BoW outperforms TF-IDF in this case due to simpler feature representation.

Clustering

    Optimal clusters determined using the Elbow Method.
    Metrics like Purity, NMI, and Rand Index highlight the quality of clustering.
