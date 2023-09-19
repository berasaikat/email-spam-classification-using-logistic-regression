# Load the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import wordcloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')


# Load the three datasets
def load_datasets():
    totaldata = pd.read_csv("./data/totaldata.csv", index_col=0)
    SpamSubset = pd.read_csv("./data/spamsubset.csv", index_col=0)
    Spam = pd.read_csv("./data/spam1.csv", index_col=0)

    SpamSubset.drop(["Unnamed: 0"], axis=1, inplace=True)
    SpamSubset = SpamSubset.reset_index().drop(['index'], axis=1)

    return pd.concat([totaldata, SpamSubset, Spam], axis=0, ignore_index=True)


# Visualize the data
def visualize_data(df):
    plt.figure(figsize=(5, 5))
    custom_palette = ["#FF5733", "#33FF57"]
    sns.histplot(data=df, x='Label', hue='Label', bins=2, palette=custom_palette)
    plt.xlabel("Email Type")
    plt.title("Spam and Normal Email Frequency")
    plt.legend(["Spam", "Normal"])
    plt.show()


# Pre-processing the text data
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\d+', '', text)
        text = re.sub('\W+', ' ', text.lower())
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text
    else:
        return ""
    

def preprocess_text(df):
    df['Body'] = df['Body'].apply(clean_text)
    stopwords_list = stopwords.words('english')
    df['Body'] = df['Body'].map(lambda x: " ".join(word for word in x.split() if word not in stopwords_list))
    lemmatizer = WordNetLemmatizer()
    df['Body'] = df['Body'].map(lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split()))
    pattern = r'[^\x00-\x7F]+'
    df['Body'] = df['Body'].apply(lambda x: re.sub(pattern, '', x))
    return df


# Generate Word Cloud for Spam Emails
def generate_word_cloud(df):
    text_data = ' '.join(df.loc[df['Label'] == 1, 'Body'].astype(str))
    wordcloud_obj = wordcloud.WordCloud(width=800, height=400, background_color='white')
    wordcloud_img = wordcloud_obj.generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_img, interpolation='bilinear')
    plt.axis("off")
    plt.show()


# Split the dataset into train and test
def split_data(df):
    X, Y = df.Body, df.Label
    train_X, test_X, train_y, test_y = train_test_split(X, Y, shuffle=True)
    return train_X, test_X, train_y, test_y


# Train a Logistic Regression model with TF-IDF vectorization
def train_logistic_regression(train_X, train_y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=1, max_iter=2000))
    ])

    param_grid = {
        'tfidf__max_features': [1000, 5000, 10000],
        'clf__C': [0.01, 0.1, 1.0],
    }
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=45)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, verbose=1, n_jobs=-1)
    grid_search.fit(train_X, train_y)

    print("Best Parameters: ", grid_search.best_params_)
    print("Best Accuracy: ", grid_search.best_score_)

    return grid_search.best_estimator_


# Evaluate the model and display results
def evaluate_model(model, test_X, test_y):

    # Evaluate the best model on the test data
    best_model = model
    test_accuracy = best_model.score(test_X, test_y)
    print("Test Accuracy with Best Model: ", test_accuracy)

    y_pred = model.predict(test_X)
    confusion = confusion_matrix(test_y, y_pred)
    report = classification_report(test_y, y_pred, target_names=["Class 0", "Class 1"])

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(report)