# preprocess.py
# preprocesses MIB dataset for Glove embedding

import numpy as np
import glob
import pandas as pd
import re
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

DIR = "datasets_full.csv/datasets_full.csv/"
URL_PATTERN = "^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$"
COLUMN_NAMES = ['retweet_count', 'reply_count', 'favorite_count',
                'num_hashtags', 'num_urls', 'num_mentions', 'isGenuine']


def isAllCaps(word):
    for c in word:
        if c.islower():
            return False
    return True


def hasRepeatedLetters(word):
    prev = ''
    prev2 = ''
    for c in word:
        if c == prev:
            if c == prev2:
                return True
        prev2 = prev
        prev = c
    return False


tweet_files = glob.glob(DIR + '*/tweets.csv')
# print(tweet_files)
data = []
file_paths = []
for file_path in tweet_files:
    tweet_set = pd.read_csv(file_path, delimiter=',', dtype=str, nrows=1000)
    tweet_set["isGenuine"] = 0
    if file_path == "datasets_full.csv/datasets_full.csv/genuine_accounts.csv/tweets.csv":
        tweet_set["isGenuine"] = 1
    data.append(tweet_set)
    file_paths.append(file_path)

# redact #[...] with <hashtag>, @[...] with <user>, urls with <url>
# for x in range(len(data)):
#     for i, row in enumerate(data[x]["text"]):
#         rowlist = row.split()
#         rowlist = [word.strip() for word in rowlist]
#         rowlist = [word if not word.strip().startswith(
#             '#') else "<hashtag>" for word in rowlist]
#         rowlist = [word if not word.strip().startswith(
#             '@') else "<user>" for word in rowlist]
#         rowlist = [word if not isAllCaps(
#             word.strip()) else word.lower() + " <allcaps>" for word in rowlist]
#         rowlist = [word if not hasRepeatedLetters(
#             word.strip()) else word + " <3+repeated>" for word in rowlist]
#         rowlist = [word.lower() for word in rowlist]
#         rowlist = [re.sub(URL_PATTERN, "<url>", word) for word in rowlist]
#         data[x]["text"][i] = " ".join(rowlist)


dataset = pd.concat([d[COLUMN_NAMES] for d in data])

y = dataset["isGenuine"]
X = dataset.drop('isGenuine', axis=1)
# X = X.fillna(value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
sme = SMOTEENN(smote=SMOTE(k_neighbors=10))
X_train, y_train = sme.fit_resample(X_train, y_train)
# rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
gbc = GradientBoostingClassifier(loss='exponential')

abc = AdaBoostClassifier(n_estimators=1000,
                         learning_rate=0.5, base_estimator=gbc)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

y_scores = model.decision_function(X_test)

print("Precision Score", precision_score(y_test, y_pred))
print("Recall Score", recall_score(y_test, y_pred))
print("F1 Score", f1_score(y_test, y_pred))
print("ROC/AUC Score", roc_auc_score(y_test, y_scores))
#
# for (df, file_path) in zip(data, file_paths):
#     print("Filepath: " + file_path)
#     print(df)
