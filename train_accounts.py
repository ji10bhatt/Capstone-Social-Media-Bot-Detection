# train_tweets.py
# trains and tests classifier using account metadata

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
COLUMN_NAMES = ["isGenuine", "statuses_count", "followers_count", "friends_count", "favourites_count",
                "listed_count", "default_profile", "geo_enabled", "profile_use_background_image", "verified", "protected"]

account_files = glob.glob(DIR + '*/users.csv')
data = []
print(account_files)
file_paths = []
for file_path in account_files:
    account_set = pd.read_csv(file_path, delimiter=',', dtype=str)
    account_set["isGenuine"] = 0
    if file_path == "datasets_full.csv/datasets_full.csv/genuine_accounts.csv/users.csv":
        account_set["isGenuine"] = 1
    data.append(account_set)
    file_paths.append(file_path)

dataset = pd.concat([d[COLUMN_NAMES] for d in data])
y = dataset["isGenuine"]
X = dataset.drop('isGenuine', axis=1)
X = X.fillna(value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)


sme = SMOTEENN(smote=SMOTE(k_neighbors=10))
X_train, y_train = sme.fit_resample(X_train, y_train)

# Select base estimator
# rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
gbc = GradientBoostingClassifier(loss='exponential')

abc = AdaBoostClassifier(n_estimators=1000,
                         learning_rate=0.5, base_estimator=gbc)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy: 0.988865692414753
# Precision Score 0.9744318181818182
# Recall Score 0.98
# F1 Score 0.9772079772079771
# ROC/AUC Score 0.9984616901038245

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

y_scores = model.decision_function(X_test)

print("Precision Score", precision_score(y_test, y_pred))
print("Recall Score", recall_score(y_test, y_pred))
print("F1 Score", f1_score(y_test, y_pred))
print("ROC/AUC Score", roc_auc_score(y_test, y_scores))
