import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import math
import matplotlib.pyplot as plt

df = pd.read_csv('emails.csv')
X = df["text"]
y = df["spam"]

count_vectorizer = CountVectorizer(ngram_range = (1, 1), stop_words = 'english', max_df = 0.7, min_df = 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 100, test_size = 0.2)
count_vectorizer.fit(X)
X_train_count = count_vectorizer.transform(X_train)
X_test_count =  count_vectorizer.transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_count, y_train)
y_predictions = knn.predict(X_test_count)

accuracy = metrics.accuracy_score(y_test,y_predictions)
error = 1 - accuracy
interval = interval = 1.96 * math.sqrt( (error * (1 - error)) / X_test_count.shape[0])

print(f'Accuracy Score: {accuracy * 100}')
print(f'Confusion Matrix:\n{metrics.confusion_matrix(y_test,y_predictions)}')
print(f'Confidence Interval: {interval}')

y_scores = knn.predict_proba(X_test_count)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_scores[:, 1])
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()