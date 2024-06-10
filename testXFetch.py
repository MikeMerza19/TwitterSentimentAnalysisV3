import pandas as pd
from ntscraper import Nitter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

from datasets import load_dataset
import numpy as np

from deep_translator import GoogleTranslator

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)

def text_translate(textTweet):
    translated_text = GoogleTranslator(source='auto', target='en').translate(textTweet)
    return translated_text

def text_preprocessing(text):
    stopwords = set()
    with open("static/en_stopwords.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))
    lemmatizer = WordNetLemmatizer()
    try:
        url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
        user_pattern = r"@[^\s]+"
        entity_pattern = r"&.*;"
        neg_contraction = r"n't\W"
        non_alpha = "[^a-z]"
        cleaned_text = text.lower()
        cleaned_text = re.sub(neg_contraction, " not ", cleaned_text)
        cleaned_text = re.sub(url_pattern, " ", cleaned_text)
        cleaned_text = re.sub(user_pattern, " ", cleaned_text)
        cleaned_text = re.sub(entity_pattern, " ", cleaned_text)
        cleaned_text = re.sub(non_alpha, " ", cleaned_text)
        tokens = word_tokenize(cleaned_text)
        # provide POS tag for lemmatization to yield better result
        word_tag_tuples = pos_tag(tokens, tagset="universal")
        tag_dict = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
        final_tokens = []
        for word, tag in word_tag_tuples:
            if len(word) > 1 and word not in stopwords:
                if tag in tag_dict:
                    final_tokens.append(lemmatizer.lemmatize(word, tag_dict[tag]))
                else:
                    final_tokens.append(lemmatizer.lemmatize(word))
        return " ".join(final_tokens)
    except:
        return np.nan
    

df = pd.read_csv('cleaned_dataset_twitter_final.csv')
df = df.dropna()
# df = df.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
# df = df[['text', 'Translated Text', 'Cleaned Text', 'label']]
# df["Translated Text"] = df["text"].apply(text_translate)
# df["Cleaned Text"] = df["Translated Text"].apply(text_preprocessing)
# df.to_csv('cleaned_dataset_twitter_final.csv')

print(df.head())




# df = pd.read_parquet('C:/Users/WebDev/Desktop/Twitter Sentiment Analysis v3.0/test-00000-of-00001-a1bf8c09fedae1d2.parquet')
# df.to_csv('C:/Users/WebDev/Desktop/Twitter Sentiment Analysis v3.0/test_dataset.csv')

# df = pd.read_csv('cleaned_datasetv2.csv')
# df = df.dropna()

# stopset = set(stopwords.words('english'))
# vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

# y = df.values[:, 4]
# X = vectorizer.fit_transform(df.values[:, 3])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

# clf = MultinomialNB()
# clf.fit(X_train, y_train) 

# # test_predict_array = temp_df["Cleaned Tweet"].to_numpy()
# # test_predict_vector = vectorizer.transform(test_predict_array)
# sentiment = clf.predict(vectorizer.transform(["i want to die"]))    
# proba = clf.predict_proba(vectorizer.transform(["i want to die"]))
# # proba_df = pd.DataFrame(proba, columns=["proba_non_suicide", "proba_suicide"])

# print(sentiment)
# print(proba)

