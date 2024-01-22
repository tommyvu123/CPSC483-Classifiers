import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
from math import sqrt

def message_preprocessing(data_set):
    data_set['text'] = data_set['text'].str.replace(r'\W', ' ', regex=True)
    data_set['text'] = data_set['text'].str.lower()
    removed_extra_whitespace = []

    for message in data_set['text']:
        cleaned_up = re.sub(' +', ' ', message)
        removed_extra_whitespace.append(cleaned_up)

    return pd.Series(removed_extra_whitespace)

email_spam = pd.read_csv('./emails.csv')

df_text = message_preprocessing(email_spam)
df_spam = email_spam['spam']

count_vectorizer = CountVectorizer()
df_text_count = count_vectorizer.fit_transform(df_text)

x_training, x_testing, y_training, y_testing = train_test_split(df_text_count, df_spam, test_size=0.2, random_state=42)

svm_classifier = make_pipeline(StandardScaler(with_mean=False, with_std=False), SVC(kernel='linear', probability=True))

svm_classifier.fit(x_training, y_training)

y_hat = svm_classifier.predict(x_testing)
confusion_result = confusion_matrix(y_testing, y_hat)
print(f'Confusion matrix: \n{confusion_result}\n')

accuracy = svm_classifier.score(x_testing, y_testing)
print(f'Comparison based on % accuracy: {accuracy}\n')

true_positives = confusion_result[0][0]
false_positivies = confusion_result[0][1]
true_negatives = confusion_result[1][1]
false_negatives = confusion_result[1][0]

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positivies)
precision = true_positives / (true_positives + false_positivies)

print(f'Sensitivity: {sensitivity}, Specificity: {specificity}, Precision: {precision}\n')

y_score = svm_classifier.predict_proba(x_testing)[::,1]

fpr, tpr, thresholds = roc_curve(y_testing, y_score, pos_label=1)
print(f'ROC=> false positive rate: \n{fpr}\n, true positive rate: \n{tpr}\n, thresholds: \n{thresholds}\n')

plt.figure()
lw = 2
plt.plot(fpr, tpr, color="blue", lw=lw, label="ROC curve")
plt.plot([0,1], [0,1], color="black", lw=lw, linestyle="--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Testing Actual Data vs Testing Predicted Data (ROC)')
plt.legend(loc="lower right")
plt.show()

interval = 1.96 * sqrt((accuracy*(1-accuracy))/ df_text.size * 0.2)
print('%.3f' % interval)