import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import string

def message_cleaning(message):
    punctuation_removed = [char for char in message if char not in string.punctuation]
    punctuation_removed_1 = ''.join(punctuation_removed)
    clean_data = [ word for word in punctuation_removed_1.split() if word not in stopwords.words('english')]
    return clean_data
