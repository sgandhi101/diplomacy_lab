import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

state_dept = pd.read_json('tweet_collection/state_dept.json', lines=True)  # Import dataset
state_dept = state_dept.assign(classification=0)
state_dept = state_dept.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

cdc = pd.read_json('tweet_collection/cdc.json', lines=True)  # Import dataset
cdc = cdc.assign(classification=1)
cdc = cdc.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

energy = pd.read_json('tweet_collection/energy.json', lines=True)  # Import dataset
energy = energy.assign(classification=2)
energy = energy.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

fbi = pd.read_json('tweet_collection/fbi.json', lines=True)  # Import dataset
fbi = fbi.assign(classification=3)
fbi = fbi.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

education = pd.read_json('tweet_collection/education.json', lines=True)  # Import dataset
education = education.assign(classification=4)
education = education.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

print("State Dept", state_dept.shape)
print("CDC", cdc.shape)
print("Energy", energy.shape)
print("FBI", fbi.shape)
print("Education", education.shape)

aggregate = state_dept.append(cdc)
aggregate = aggregate.append(energy)
aggregate = aggregate.append(fbi)
aggregate = aggregate.append(education)

X = aggregate['tweet']
y = aggregate['classification']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20)
vector_data = CountVectorizer()  # or term frequency

X_train_tf = vector_data.fit_transform(train_X)
X_test_tf = vector_data.transform(test_X)

error = []
lowestError = 1  # Start off with the worst possible error and this will get compared every iteration to find the lowest
iterations = 10
for i in range(1, iterations):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_tf, train_y)
    pred_i = knn.predict(X_test_tf)
    error.append(np.mean(pred_i != test_y))
    x = np.mean(pred_i != test_y)
    if x < lowestError:
        lowestError = x

print("*" * 100)
# Print out all our error rate data to determine the accuracy of the model
print("Lowest Error Rate", lowestError * 100, "%")
print("Median Error Rate", np.median(error) * 100, "%")
print("Median Accuracy Rate:", 100 - (np.median(error) * 100), "%")

# Output a graph with matplotlib in order to see our results graphed
plt.figure(figsize=(12, 6))
plt.plot(range(1, iterations), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

print("*" * 100)
decision_tree = DecisionTreeClassifier()  # Initialize the decision tree
decision_tree.fit(X_train_tf, train_y)  # Fit the training data
y_pred = decision_tree.predict(X_test_tf)  # Use this to predict

score = metrics.accuracy_score(test_y, y_pred)
print("accuracy:   %0.3f" % score)

print(metrics.classification_report(test_y, y_pred, target_names=['State Dept', 'CDC', 'Energy', 'FBI', 'Education']))

print("confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))

print("*" * 50)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)

y_pred = naive_bayes_classifier.predict(X_test_tf)

score = metrics.accuracy_score(test_y, y_pred)
naive_classification_report = metrics.classification_report(test_y, y_pred)
skplt.metrics.plot_confusion_matrix(test_y, y_pred, normalize=True)
plt.show()

print("Confusion Matrix:")
print(metrics.confusion_matrix(test_y, y_pred))
print(naive_classification_report)
print("Naive Bayes Accuracy:", round(score, 2))
