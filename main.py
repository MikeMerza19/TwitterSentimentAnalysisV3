from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn import metrics
import pandas as pd
import numpy as np
import string
import nltk
from datasetsHF import load_dataset
import datasetsHF as ds
nltk.download('stopwords')
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import pad_sequences
import pickle
import snscrape.modules.twitter as sntwitter
import datetime as dt
import nltk

nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image

import pandas as pd
from ntscraper import Nitter

from sklearn import metrics

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from googletrans import Translator

scraper = Nitter()

translator = Translator()
source_lang = "tl"
translated_to = "en"

def get_tweets(username, mode, number_of_tweets):
    tweets = scraper.get_tweets(username, mode = mode, number = number_of_tweets)
    final_tweets = []

    for tweet in tweets['tweets']:
        data = [tweet['is-retweet'] ,tweet['user']['name'], tweet['date'], tweet['text']]
        final_tweets.append(data)

    data = pd.DataFrame(final_tweets, columns = ['Retweet?', 'User Name', 'Date', 'Extracted Tweet'])

    return data

def text_translate(textTweet):
    translated_text = translator.translate(textTweet, src=source_lang, dest=translated_to)
    return translated_text.text

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


data = get_tweets('elonmusk', 'user', 10)

def testtt(item):
    # temp_df = item
    # temp_df["Cleaned Tweet"] = temp_df["Extracted Tweet"].apply(text_preprocessing)
    # temp_df = temp_df[(temp_df["Cleaned Tweet"].notna()) & (temp_df["Cleaned Tweet"] != "")]

    temp_df = item
    temp_df["Translated Tweet"] = temp_df["Extracted Tweet"].apply(text_translate)
    temp_df["Cleaned Tweet"] = temp_df["Translated Tweet"].apply(text_preprocessing)
    temp_df = temp_df[(temp_df["Cleaned Tweet"].notna()) & (temp_df["Cleaned Tweet"] != "")]

    dtSet = ds.load_dataset('vibhorag101/suicide_prediction_dataset_phr', split='train')
    df = dtSet.to_pandas()
    df = df.dropna()

    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

    y = df.values[:, 1]
    X = vectorizer.fit_transform(df.values[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    test_predict_array = temp_df["Cleaned Tweet"].to_numpy()
    test_predict_vector = vectorizer.transform(test_predict_array)
    sentiment = clf.predict(test_predict_vector)    
    proba = clf.predict_proba(test_predict_vector)
    proba_df = pd.DataFrame(proba, columns=["proba_non_suicide", "proba_suicide"])
    proba_df_converted = (proba_df * 100).round(2)

    temp_df["Sentiment"] = pd.Series(sentiment)
    new_df = pd.concat([temp_df, proba_df_converted], axis=1)
    new_df = new_df[(new_df["Sentiment"].notna())]
    new_df["Row No."] = np.arange(new_df.shape[0])


    print(new_df)

testtt(data)