import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import oracledb
from TweetProcess.dbConfig import user, passwd, dsn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


connection = oracledb.connect(
        user=user[0],
        password=passwd[0],
        dsn=dsn)
cursor = connection.cursor()

res = cursor.execute("select CONTAINS,ISCYBERBULLYING from Z_COMPHERENSIVE_CYBERBULLYING")
#res = cursor.execute("select text,tip from sample_data")
res = res.fetchall()


X, t = [item[0] for item in res], [item[1] for item in res]

# X_train, X_test, t_train, t_test = train_test_split(
#     X, t, test_size=0.3, shuffle=True, random_state=1)


# Y_pred = cursor.execute("select TEXT,TIP from TURKISH_CYBERBULLYING")
# Y_pred = Y_pred.fetchall()

# Test verisinin DB'den alınması
DB_test = cursor.execute("select lower(TWEET) from output where status=1 and rownum<1000")
DB_test = DB_test.fetchall()

cols = ['tweet_id', 'text', 'tweet_lang', 'clean_tweet', 'cleaned_tweet']
use_cols = ['tweet_id', 'cleaned_tweet']
res = pd.read_csv("F:\\Yüksek Lisans\\Veri Seti\\clear_tweets.csv", delimiter='|', header=1,
                  names=cols, usecols=use_cols)
X_test = res['cleaned_tweet']
X_test = X_test.dropna()
X_test = list(X_test[:2500])
# Y_test, Y_target = [item[0] for item in Y_pred], [int(item[1]) for item in Y_pred]
#X_test = [item[0] for item in X_test]

print(len(X_test))

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression())])

text_clf = text_clf.fit(X, t)

predicted = text_clf.predict(X_test)
#1401464248711192578
pred = []
for k in range(len(X_test)):
    pred.append([X_test[k], predicted[k]])

df = pd.DataFrame(pred)
df.to_csv("F:\\Yüksek Lisans\\Veri Seti\\Sample_Predicted_Data_st.csv", sep='|', index=False,
              mode='a', header=False)
# print(f"Accuracy is: {accuracy_score(t_test,predicted)*100}")

# print(f" F1-Score: {f1_score(t_test,predicted)}")
