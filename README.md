# Email Spam Classification Using Logistic Regression

In this project, I employed a Logistic Regression model for the task of spam classification. The goal was to distinguish between spam and non-spam (normal) emails using machine learning techniques.

## Data Preprocessing

To prepare the text data for modeling, we conducted a series of preprocessing steps using Natural Language Processing (NLP) tools. These steps included:

1. **Text Cleaning**: We removed digits, non-alphanumeric characters, and converted all text to lowercase. Additionally, we eliminated punctuation to ensure consistency in the text.

2. **Stopword Removal**: Common stopwords (e.g., "and," "the," "in") were removed from the text as they often do not carry significant information for classification.

3. **Lemmatization**: We applied lemmatization to reduce words to their base or root form. This step helped in standardizing words and reducing dimensionality.

4. **Character Encoding**: To handle non-ASCII characters, we removed any characters not in the standard English alphabet.

## Data Tabularization and Labeling

We utilized a tabular format to represent the preprocessed text data. The dataset consisted of two main columns: "Body" for email content and "Label" to indicate whether an email was spam (1) or normal (0).

## Train-Test Split

To evaluate the performance of our classification model, we divided the dataset into two parts: a training set and a test set. The "shuffle=True" parameter was set to ensure the data's randomness during the split. By default, 75% of the data was used for training, while the remaining 25% was used for testing.

## Model Training and Hyperparameter Tuning

We constructed a Logistic Regression model and applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to the text data. TF-IDF is a technique used to transform text into numerical features that capture the importance of words in documents.

To optimize the model's performance, we conducted a grid search over hyperparameters using cross-validation. The key hyperparameters tuned were:

- `tfidf__max_features`: The maximum number of features (words) to consider in TF-IDF vectorization.
- `clf__C`: The inverse regularization strength for the Logistic Regression model.

## Results

After training and hyperparameter tuning, we obtained the following results:

- **Best Parameters**: The hyperparameters that yielded the best performance were `{'clf__C': 1.0, 'tfidf__max_features': 10000}`.

- **Best Accuracy**: The model achieved a cross-validated accuracy of approximately 96.61% on the training data.

- **Test Accuracy with Best Model**: When evaluating the best model on the test data, we achieved an accuracy of approximately 96.33%.

- **Classification Report**: The classification report provides a more detailed assessment of the model's performance. It includes precision, recall, and F1-score for both classes (spam and normal emails) and overall accuracy.

Here is the classification report for the model:
|         | Precision |  Recall  | F1-Score | Support |
|---------|-----------|----------|----------|---------|
| Class 0 |   0.97    |   0.97   |   0.97   |  2825   |
| Class 1 |   0.95    |   0.96   |   0.95   |  1838   |
|---------|-----------|----------|----------|---------|
| Accuracy|           |          |   0.96   |  4663   |
| Macro Avg |   0.96  |   0.96   |   0.96   |  4663   |
| Weighted Avg |   0.96  |   0.96   |   0.96   |  4663   |

These results demonstrate that the Logistic Regression model, with the selected hyperparameters, performs effectively in distinguishing between spam and normal emails. The high precision, recall, and F1-score values reflect the model's ability to make accurate predictions.
