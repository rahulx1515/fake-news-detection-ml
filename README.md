# AI-Powered Fake News Detection Using Text Classification

## Project Overview

Fake news has become a major problem with the rapid growth of online news and social media. Misleading or false information can spread quickly and influence public opinion.

This project develops a Machine Learning pipeline to classify news articles as Fake or Real using Natural Language Processing (NLP) and different Machine Learning techniques.

The project focuses on text preprocessing, feature engineering, model training, and evaluation.

## Objectives

- Perform Exploratory Data Analysis (EDA) on the news dataset.
- Clean and preprocess news articles.
- Convert text data into numerical features.
- Implement Bag of Words (BoW).
- Implement TF-IDF.
- Implement Word Embeddings using Word2Vec.
- Train different Machine Learning models.
- Evaluate the models using standard classification metrics.
- Compare the performance of different models.
- Identify the most suitable model for fake news classification.

## Dataset

The dataset contains separate collections of fake and real news articles.

Main columns:
- title - Title of the news article
- text - Main content of the article
- subject - Subject/category of the article
- date - Publication date

Target labels:
- 0 - Fake News
- 1 - Real News

The raw dataset files are:
- Fake.csv
- Real.csv

They are stored locally inside:
data/raw/

The raw dataset is excluded from GitHub using .gitignore.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- NLTK
- Gensim
- SciPy
- Jupyter Notebook

## Project Workflow

Dataset
│
├── Exploratory Data Analysis
│
├── Text Preprocessing
│
├── Feature Engineering
│   ├── Bag of Words
│   ├── TF-IDF
│   └── Word2Vec Embeddings
│
├── Model Training
│   ├── Logistic Regression
│   ├── K-Nearest Neighbors
│   ├── Random Forest
│   └── Neural Network
│
├── Model Evaluation
│   ├── Accuracy
│   ├── Precision
│   ├── Recall
│   ├── F1-Score
│   └── Confusion Matrix
│
└── Model Comparison

## Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the characteristics of the dataset.

The analysis includes:
- Dataset size and structure
- Dataset columns
- Missing values
- Duplicate records
- Fake vs Real news distribution
- Article length
- Word count
- Title length
- Subject/category distribution
- Publication date distribution
- Frequently occurring words
- Example Fake and Real news titles

EDA is implemented in:
notebooks/01_data_exploration.ipynb

## Text Preprocessing

The text preprocessing pipeline includes:

1. Handling missing values
2. Removing duplicate records
3. Combining title and article text
4. Converting text to lowercase
5. Removing URLs
6. Removing HTML tags
7. Removing punctuation
8. Removing numerical values
9. Removing extra whitespace
10. Removing stopwords
11. Tokenizing the text

The processed dataset is stored locally as:
data/processed/cleaned_news.csv

Implementation:
notebooks/02_preprocessing.ipynb

## Feature Engineering

Three different techniques are explored for converting news text into numerical representations.

### 1. Bag of Words

Bag of Words represents documents based on the occurrence of words and n-grams.

Implementation:
CountVectorizer

### 2. TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) assigns importance to words based on their occurrence within documents and across the dataset.

Implementation:
TfidfVectorizer

### 3. Word2Vec Embeddings

Word2Vec learns numerical representations of words based on their surrounding context.

The learned word vectors are combined to create a numerical representation of each news article.

Implementation:
Word2Vec

Feature engineering is implemented in:
notebooks/03_feature_engineering.ipynb

## Machine Learning Models

### Logistic Regression

Logistic Regression is used as a classification model for predicting whether a news article is fake or real.

### K-Nearest Neighbors

KNN is a non-parametric algorithm that classifies an article based on the labels of nearby training examples.

### Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to perform classification.

### Neural Network

A simple neural network is implemented using a hidden layer with ReLU activation. Word2Vec document representations are used as input features.

Model training is implemented in:
notebooks/04_model_training.ipynb

## Model Evaluation

The models are evaluated using the following metrics:

### Accuracy

Measures the percentage of correctly classified news articles.

### Precision

Measures how many of the articles predicted as a particular class actually belong to that class.

### Recall

Measures how many of the actual articles belonging to a class were correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

The evaluation is implemented in:
notebooks/05_evaluation.ipynb

## Project Structure

fake-news-detection-ml/
│
├── data/
│   ├── processed/
│   │   ├── bow_vectorizer.pkl
│   │   ├── cleaned_news.csv
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── word2vec.model
│   │   ├── word2vec.model.syn1neg.npy
│   │   ├── word2vec.model.wv.vectors.npy
│   │   ├── X_test_bow.npz
│   │   ├── X_test_tfidf.npz
│   │   ├── X_test_w2v.npy
│   │   ├── X_train_bow.npz
│   │   ├── X_train_tfidf.npz
│   │   ├── X_train_w2v.npy
│   │   ├── y_test.csv
│   │   └── y_train.csv
│   │
│   └── raw/
│       ├── Fake.csv
│       └── Real.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
│
├── results/
│   ├── evaluation_results.csv
│   └── model_comparison.csv
│
├── src/
│
├── .gitignore
└── README.md

## How to Run the Project

### 1. Clone the Repository

git clone https://github.com/rahulx1515/fake-news-detection-ml.git

### 2. Navigate to the Project Directory

cd fake-news-detection-ml

### 3. Install the Required Libraries

pip install pandas numpy matplotlib scikit-learn nltk gensim scipy jupyter

### 4. Download NLTK Stopwords

Run the following Python code:

import nltk
nltk.download("stopwords")

### 5. Add the Dataset

Place the following files inside:
data/raw/

- Fake.csv
- Real.csv

### 6. Run the Notebooks in Order

1. 01_data_exploration.ipynb
2. 02_preprocessing.ipynb
3. 03_feature_engineering.ipynb
4. 04_model_training.ipynb
5. 05_evaluation.ipynb

## Results

The project compares the performance of:

- Logistic Regression
- K-Nearest Neighbors
- Random Forest
- Neural Network

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Final results and visualizations are stored in:
results/

Final performance values will be added after completing and verifying all evaluation experiments.

## Future Improvements

Possible future improvements include:

- Using pretrained word embeddings
- Experimenting with transformer-based models
- Hyperparameter tuning
- Cross-validation
- Testing additional datasets
- Improving handling of class imbalance
- Developing a web-based prediction interface
- Adding real-time news verification
- Using external evidence sources for fact verification

## License

This project is developed for educational and academic purposes.
