import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session


filepath = 'F:\\Yüksek Lisans\\Veri Seti\\turkish-cyber-bullying_.csv'
df = pd.read_csv(filepath, sep=',', names=['text', 'tip'])


tweet = df['text'].values #return ndarray
iscyberbullying = df['tip'].values #return ndarray

tweet_train, tweet_test, iscyberbullying_train, iscyberbullying_test = train_test_split(
    tweet, iscyberbullying, test_size=0.33, random_state=1000)

vectorizer = CountVectorizer()
vectorizer.fit(tweet_train)

x_train = vectorizer.transform(tweet_train)
x_test = vectorizer.transform(tweet_test)

##ForDeepLearning
input_dim = x_train.shape[1]  # Number of features
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, iscyberbullying_train, epochs=100, verbose=False,
                    validation_data=(x_test, iscyberbullying_test),
                    batch_size=10)

clear_session()

loss, accuracy = model.evaluate(x_train, iscyberbullying_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, iscyberbullying_test, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))


classifier = LogisticRegression()
classifier.fit(x_train, iscyberbullying_train)

score = classifier.score(x_test, iscyberbullying_test)
print("Accuracy:", score)
