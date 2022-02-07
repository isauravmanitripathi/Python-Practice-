import pickle 
import numpy as np 
import pandas as pd
from sklearn.externals import jolib 
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_selection import SelectFromModel 
from sklearn.feature_extraction.text import TfidVectorizer 
import nltk 
from nltk.stem.porter import * 
import string 
import re 

from vaderSentiment.vaderSentiment import SentimentIntesityAnalyzer as VS 
from textstat.textstat import * 

stopwords = stopwords = nltk.corpus.stopwords.word("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

sentiment_analyzer = VS()

stemmer = PorterStemmer()

def preprocess(text_string):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)

    return tokens 

def basic_tokenize(tweet):
    tweet = " ".join(re.split("[^a-zA-Z./!?]*", tweet.lower())).strip()
    return tweet.split()

def get_pos_tags(tweets):

    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags

def count_twitter_objs(text_string):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, "MENTIONHERE", parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    return(parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))

def other_features_(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = preprocess(tweet)
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables + 0.01))/ float(num_words + 0.001), 4)
    num_unique_terms = len(set(words.split()))

    FKRA = round(float(0.39 * float(num_words) / 1.0) + float(11.8 * avg_syl) - 15.59, 1)

    FRE = round(206.835 - 1.015 * (float(num_words)/ 1.0) - (84.6 * float(avg_syl)), 2)

    twitter_objs = count_twitter_objs(tweet) #Count #, @, and http://
    features = [FKRA, FRE, syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['compound'],
                twitter_objs[2], twitter_objs[1],]
    return features

def get_oth_features(tweets):
    feats = []
    for t in tweets:
        feats.append(other_features_(t))
    return np.array(feats)


def transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer):
    tf_array = tf_vectorizer.fit_tranfor(tweets).toarray()
    tfidf_array = tf_array * idf_vector
    print("Built TF-IDF array")

    pos_tags = get_pos_tags(tweet)
    pos_array = pos_vectorizer.fit_tranform(pos_tags).toarray()
    print("Built POS Array")

    oth_array = get_oth_features(tweets)
    print("Built other feature array")

    M = np.concatenate([tfidf_array, pos_array, oth_array], axis = 1)
    return pd.DataFrame(M)

def predictions(X, model):
    y_preds = model.predict(X)
    return y_preds

def class_to_name(class_label):
    if class_label == 0:
        return "Hate Speech"
    if class_label == 1:
        return "Offensive Language"
    if class_label == 2:
        return "Neither"
    return "No Label"

def get_tweets_predictions(tweets, perform_prints = True):
    fixed_tweet = []
    for i, t_orig in enumerate(tweets):
        s = t_orig
        try:
            s = s.encode("latin1")
        except:
            try:
                s = s.encode("utf-8")
            except:
                pass
        if type(s) != unicode:
            fixed_tweets.append(unicode(s, errors="ignore"))
        else:
            fixed_tweets.append(s)
    assert len(tweets) == len(fixed_tweets), ("shouldn't remove any tweets")
    tweets = fixed_tweets
    print len(tweets), (" tweets to classify")

    print ("Loading trained classifier... ")
    model = joblib.load('final_model.pkl')

    print ("Loading other information...")
    tf_vectorizer = joblib.load('final_tfidf.pkl')
    idf_vector = joblib.load('final_idf.pkl')
    pos_vectorizer = joblib.load('final_pos.pkl')
    #Load ngram dict
    #Load pos dictionary
    #Load function to transform data

    print ("Transforming inputs...")
    X = transform_inputs(tweets, tf_vectorizer, idf_vector, pos_vectorizer)

    print ("Running classification model...")
    predicted_class = predictions(X, model)

    return predicted_class


if __name__ == '__main__':
    print ("Loading data to classify...")

  

    df = pd.read_csv('trump_tweets.csv')
    trump_tweets = df.Text
    trump_tweets = [x for x in trump_tweets if type(x) == str]
    trump_predictions = get_tweets_predictions(trump_tweets)

    print ("Printing predicted values: ")
    for i,t in enumerate(trump_tweets):
        print t
        print 
        class_to_name(trump_predictions[i])

    print ("Calculate accuracy on labeled data")
    df = pd.read_csv('../data/labeled_data.csv')
    tweets = df['tweet'].values
    tweets = [x for x in tweets if type(x) == str]
    tweets_class = df['class'].values
    predictions = get_tweets_predictions(tweets)
    right_count = 0
    for i,t in enumerate(tweets):
        if tweets_class[i] == predictions[i]:
            right_count += 1

    accuracy = right_count / float(len(df))
    print ("accuracy"), accuracy

