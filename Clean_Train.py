from Pipeline import message_cleaning
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

tweets_df = pd.read_csv('train.csv')
tweets_df= tweets_df.drop(['id'],axis = 1)
tweets_df['length']= tweets_df['tweet'].apply(len)

tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)
#print(tweets_df_clean[5])

vectorizer = CountVectorizer(analyzer = message_cleaning, dtype = 'uint8')
tweets_countvectorizer = vectorizer.fit_transform(tweets_df['tweet'])
print(tweets_countvectorizer.shape)

tweets = pd.DataFrame(tweets_countvectorizer.toarray())
X = tweets
Y = tweets_df['label']

print(X.shape)
print(Y.shape)

X_train,X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,Y_train)
