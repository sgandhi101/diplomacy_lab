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

state_dept = pd.read_json('tweet_collection/state_dept.json', lines=True)  # Import State Dept Dataset
state_dept = state_dept.assign(classification=0)
state_dept = state_dept.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

cdc = pd.read_json('tweet_collection/cdc.json', lines=True)  # Import CDC Dataset
cdc = cdc.assign(classification=1)
cdc = cdc.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

energy = pd.read_json('tweet_collection/energy.json', lines=True)  # Import Energy Dept Dataset
energy = energy.assign(classification=2)
energy = energy.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

fbi = pd.read_json('tweet_collection/fbi.json', lines=True)  # Import FBI Dataset
fbi = fbi.assign(classification=3)
fbi = fbi.drop(
    ['date', 'video', 'id', 'conversation_id', 'created_at', 'timezone', 'user_id', 'username', 'name', 'place',
     'language', 'mentions',
     'urls', 'replies_count', 'retweets_count', 'likes_count', 'hashtags', 'cashtags', 'link', 'retweet', 'quote_url',
     'thumbnail', 'near', 'geo', 'source', 'user_rt_id', 'user_rt', 'retweet_id', 'reply_to', 'retweet_date',
     'translate', 'trans_src', 'trans_dest', 'photos', 'time'],
    axis=1)

education = pd.read_json('tweet_collection/education.json', lines=True)  # Import Education Dept Dataset
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

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
vector_data = CountVectorizer()

x_train_transformed = vector_data.fit_transform(x_train)
x_test_transformed = vector_data.transform(x_test)

error = []
lowestError = 1
iterations = 10
for i in range(1, iterations):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train_transformed, y_train)
    pred_i = knn.predict(x_test_transformed)
    error.append(np.mean(pred_i != y_test))
    x = np.mean(pred_i != y_test)
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
decision_tree.fit(x_train_transformed, y_train)  # Fit the training data
y_pred = decision_tree.predict(x_test_transformed)  # Use this to predict

score = metrics.accuracy_score(y_test, y_pred)
print("accuracy:   %0.3f" % score)

print(metrics.classification_report(y_test, y_pred, target_names=['State Dept', 'CDC', 'Energy', 'FBI', 'Education']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print("*" * 50)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(x_train_transformed, y_train)

y_pred = naive_bayes_classifier.predict(x_test_transformed)

score = metrics.accuracy_score(y_test, y_pred)
naive_classification_report = metrics.classification_report(y_test, y_pred,
                                                            target_names=['State Dept', 'CDC', 'Energy', 'FBI',
                                                                          'Education'])
skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
plt.show()

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
print(naive_classification_report)
print("Naive Bayes Accuracy:", round(score, 2))
