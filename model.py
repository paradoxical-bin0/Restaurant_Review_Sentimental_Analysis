# Hotel Review Sentiment Analysis

# Importing libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Warnings
import warnings

warnings.filterwarnings('ignore')

# Preliminary Data Analysis
data = pd.read_csv('Restaurant reviews.csv')
shape = data.shape
# print(shape)
# print(data.head(5))
# print(data.info())

# Data Cleaning
# Checking
null_count = data.isnull().sum().sort_values(ascending=False)
percentage_of_null_values = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
missing_data = pd.concat([null_count, percentage_of_null_values], axis=1, keys=['Null Count', 'Percentage Null'])
# print(missing_data)

# Dropping
data.drop(columns=['7514', 'Reviewer', 'Time', 'Pictures'], inplace=True)
data.dropna(inplace=True)
data.dropna()

# Re-checking
null_count = data.isnull().sum().sort_values(ascending=False)
percentage_of_null_values = (data.isnull().sum() / len(data) * 100).sort_values(ascending=False)
missing_data = pd.concat([null_count, percentage_of_null_values], axis=1, keys=['Null Count', 'Percentage Null'])
# print(missing_data)

# Viewing ratings and adding new column good or bad
# print(round(data.Rating.value_counts(normalize=True) * 100, 2))
round(data.Rating.value_counts(normalize=True) * 100, 2).plot(kind='bar')
plt.title('Percentage Distributions by ratings')
# plt.show()


# Define a function to categorize the reviews as 'good' or 'bad'
data = data[data['Rating'] != 'Like']


def categorize_review(rating):
    if float(rating) >= 3:
        return 'Good'
    else:
        return 'Bad'


# Apply the function to create the 'Review Type' column
data['Review_Type'] = data['Rating'].apply(categorize_review)

# Reviewing
# print(round(data.Review_Type.value_counts(normalize=True) * 100, 2))
round(data.Review_Type.value_counts(normalize=True) * 100, 2).plot(kind='bar')
plt.title('Percentage Distributions by review type')
# plt.show()

# Apply first level cleaning
import re
import string

# This function converts to lower-case, removes square bracket, removes numbers and punctuation


def text_clean_1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


cleaned1 = lambda x: text_clean_1(x)
# Let's take a look at the updated text
data['cleaned_description'] = pd.DataFrame(data.Review.apply(cleaned1))
# print(data.head(10))

# Apply a second round of cleaning


def text_clean_2(text):
    text = re.sub('[‘’“”…]', '', text)  # It removes any occurrence of the characters '‘', '’', '“', '”', or '…' from the text.
    text = re.sub('\n', '', text)  # It removes any newline characters from the text, effectively replacing them with nothing, which removes line breaks from the text.
    return text

cleaned2 = lambda x: text_clean_2(x)

# Let's take a look at the updated text
data['cleaned_description_new'] = pd.DataFrame(data['cleaned_description'].apply(cleaned2))
# print(data.head(10))

column_names = data.columns
# print(column_names)

# Model Training
from sklearn.model_selection import train_test_split

X = data.cleaned_description_new
y = data.Review_Type

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=225)

# print('X_train :', len(X_train))
# print('X_test :', len(X_test))
# print('y_train :', len(y_train))
# print('y_test :', len(y_test))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression(solver="lbfgs")


from sklearn.pipeline import Pipeline
model = Pipeline([('vectorizer', tvec), ('classifier', clf2)])

model.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix

predictions = model.predict(X_test)

# print(confusion_matrix(predictions, y_test))

# Model Prediction
from sklearn.metrics import accuracy_score, precision_score, recall_score

# print("Accuracy : ", accuracy_score(predictions, y_test))
# print("Precision : ", precision_score(predictions, y_test, average='weighted'))
# print("Recall : ", recall_score(predictions, y_test, average='weighted'))

# example = ["I'm happy"]
# result = model.predict(example)
#
# print(result)

# Dumping the model object

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

# Reloading model object

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict(['I\'m happy']))

