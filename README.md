# Sentiment-Analysis-Using-Naive-Bayes

This project involves performing sentiment analysis on a product review dataset. The goal is to classify reviews based on their sentiment using Naive Bayes classifiers. We will explore two different feature extraction techniques: Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TfIdf). The dataset consists of customer reviews with star ratings belonging to four classes (1, 2, 4, 5).
Dataset

The dataset comprises product reviews and their corresponding star ratings. The data is provided in a split format with separate training and testing indices available in the train_test_index.pickle file.
Steps to Complete the Project
1. Data Preprocessing

    Text Cleaning: Remove punctuation, convert text to lowercase, and remove any non-alphanumeric characters.
    Stop Word Removal: Remove common stop words that do not contribute to the sentiment of the review.
    Stemming/Lemmatization: Reduce words to their root forms to standardize the text and reduce dimensionality.

2. Feature Extraction

    Bag of Words (BoW): Convert text into numerical feature vectors using word counts. Choose words that optimize model performance.
    TfIdf: Convert text into numerical feature vectors using Term Frequency-Inverse Document Frequency to emphasize important words.

3. Model Training and Evaluation

    Train-Test Split: Use the provided train_test_index.pickle file to split the data into training and testing sets.
    Model Training: Train Naive Bayes classifiers using the BoW and TfIdf features.
    Model Evaluation: Evaluate model performance using metrics such as accuracy, precision, recall, and F1 score. Report any interesting observations.

Prerequisites

    Python 3.x
    Required Libraries: pandas, numpy, sklearn, nltk

Instructions

    Clone the Repository:

    sh

git clone <repository_url>
cd <repository_directory>

Install Dependencies:

sh

pip install -r requirements.txt

Download Dataset:

    Download the dataset from the provided link and place it in the data directory.

Run Preprocessing:

    Execute the preprocessing script to clean the text and extract features.

sh

python preprocess.py

Train and Evaluate Models:

    Run the training script to train Naive Bayes classifiers and evaluate their performance.

sh

    python train_evaluate.py

Results

    BoW Features: Report accuracy, precision, recall, and F1 score.
    TfIdf Features: Report accuracy, precision, recall, and F1 score.
    Observations: Provide insights based on the performance metrics.

Conclusion

This project demonstrates the application of Naive Bayes classifiers for sentiment analysis on a product review dataset. By comparing BoW and TfIdf feature extraction techniques, we gain insights into their effectiveness for this task.
