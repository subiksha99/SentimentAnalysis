import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
import nltk
from Pipeline import message_cleaning
from Generate_wordclouds import generate_wordcoud


from nltk.corpus import stopwords
import string

import sklearn
from sklearn.feature_extraction.text import CountVectorizer


tweets_df = pd.read_csv('train.csv')
#print(tweets_df.head())

tweets_df= tweets_df.drop(['id'],axis = 1)
print(tweets_df.head())

#finding if any null data points exist
sns.heatmap(tweets_df.isnull(), yticklabels = False, cbar = False, cmap = 'Blues')
plt.show()

#Analysing the number of positive and negative tweets
sns.countplot(tweets_df['label'], label = 'count')
plt.show()

#Finding the length of the tweets present in the dataset
tweets_df['length']= tweets_df['tweet'].apply(len)
tweets_df['length'].hist(bins=100,color='r')
plt.show()

print(tweets_df.describe())

#Printing the shortest,longest and average length tweets
print('The shortest tweet is :', tweets_df[tweets_df['length']==11]['tweet'].iloc[0])
print('The tweet with average length is :', tweets_df[tweets_df['length']==84]['tweet'].iloc[0])
print('The longest tweet is:', tweets_df[tweets_df['length']==274]['tweet'].iloc[0])


#Dividing dataset into positive and negative tweets
positive= tweets_df[tweets_df['label']==0]
negative= tweets_df[tweets_df['label']==1]

#print(positive.head())
#print(negative.head())

# word cloud of data
sentences = tweets_df['tweet'].tolist()

#print(len(sentences))
one_giant_sentence = " ".join(sentences)

generate_wordcoud(one_giant_sentence)
# negative data as one sentence

neg_sentences = negative['tweet'].tolist()

neg_giant_sentence = " ".join(neg_sentences)

#positive data as one sentence
pos_sentences = positive['tweet'].tolist()

pos_giant_sentence = " ".join(pos_sentences)


#tweets_df_clean = tweets_df['tweet'].apply(message_cleaning)
#vectorizer = CountVectorizer(analyser = message_cleaning)
#tweets_countvectorizer = CountVectorizer(analyser = message_cleaning, dtype = 'uint8').fit_transform(tweets_df['tweet']).toarray()
