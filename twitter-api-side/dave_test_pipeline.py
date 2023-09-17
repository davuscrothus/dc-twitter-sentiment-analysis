import tweepy
import nltk
import pandas as pd
from datetime import datetime, timedelta, date
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

# Input the important user credentials, ideally these would be encrypted
TWITTER_X_CONSUMER_KEY = 'AaHDae8jAkG3H9Ku2eO6hTf7G'
TWITTER_X_CONSUMER_SECRET = 'RYiH18DXSLAlV8FDMDo7Jxc6RMGzHRNoY8iTXnT1EBU3I24eTu'
TWITTER_X_ACCESS_TOKEN = '1701967006843449345-lSjaTVnM288fCWRoDWfIho6zHgZOT5'
TWITTER_X_ACCESS_TOKEN_SECRET = 'F6vLYUwtob9b5zVyRKeIK1fAom2vnxzUVdPnoJVyxFIbz'

TWITTER_X_BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAD69pwEAAAAACugyBA1idqZMcGuK5jsznQpG9Qk%3DYmoIAEmRLLClkvDTjdcwx4ir6dJ4ZarMWZ5iOhLFhORaxJ0aEP'

TEXTBLOB_MODEL_VERSION = 1.0
VADER_MODEL_VERSION = 1.0

FLASK_APP_NAME = ''

LEGO_HASHTAG_TO_SEARCH = ['#LEGOICONS']

NUM_TWEETS_TO_ANALYSE = 10

# Set Up Twitter/X Authorisation Client for all further calls.      
client = tweepy.Client(TWITTER_X_BEARER_TOKEN, return_type=dict)

# Get useful components from the Natural Language Toolkit for Pre-Prep of Tweets
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Due to limited time, I have chosen to keep to English language only
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# I thought it prudent to show a multi-model solution, initialises the vader analysis
vader_analyzer = SentimentIntensityAnalyzer()

# Create a Flask App
app = Flask(__name__)

# Create a list of all tweets retrieved from the API
def make_df_of_raw_tweet_text(query_text):
    tweet_dict = client.search_recent_tweets(query=query_text)
    tweet_list = []
    for raw_tweet_data in tweet_dict['data']:
        raw_tweet_text = raw_tweet_data['text']
        tweet_list.append(raw_tweet_text)
    df_tweets_raw = pd.DataFrame(data=tweet_list, columns=['tweet_text_raw'])
    df_tweets_raw['request_timestamp'] = datetime.utcnow().isoformat()
    return df_tweets_raw


def preprocess_text(raw_tweet):
    lower_raw_tweet = raw_tweet.lower()
    tokens = word_tokenize(lower_raw_tweet)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def perform_textblob_sentiment_analysis(processed_tweet):
    analysis = TextBlob(processed_tweet)
    # Classify sentiment (positive, negative, or neutral)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'
    
    
def perform_vader_sentiment_analysis(processed_tweet):
    sentiment_scores = vader_analyzer.polarity_scores(processed_tweet)
    return sentiment_scores


def make_df_with_analyses(df_tweets_raw):
    # Pre-Process Tweet texts
    for index, tweet in df_tweets_raw.iterrows():
        processed_text = preprocess_text(tweet['tweet_text_raw'])
        textblob_sentiment_score = perform_textblob_sentiment_analysis(processed_text)
        vader_sentiment_score = perform_vader_sentiment_analysis(processed_text)
        df_tweets_raw.loc[index, 'cleaned_tweet_text'] = processed_text
        df_tweets_raw.loc[index, 'textblob_score'] = textblob_sentiment_score
        df_tweets_raw.loc[index, 'neg_vader_score'] = vader_sentiment_score['neg']
        df_tweets_raw.loc[index, 'neu_vader_score'] = vader_sentiment_score['neu']
        df_tweets_raw.loc[index, 'pos_vader_score'] = vader_sentiment_score['pos']
        df_tweets_raw.loc[index, 'comp_vader_score'] = vader_sentiment_score['compound']
    # Remove Duplicates
    df_tweets_cleaned = df_tweets_raw.drop_duplicates(ignore_index=True, inplace=False)
    return df_tweets_cleaned


@app.route('/vader_sentiment', methods=['GET'])
def get_sentiment():
    hashtag = request.args.get('hashtag', default=None)
    if hashtag is None:
        return jsonify({"error": "Please provide a hashtag parameter"}), 400

    # Collect tweets and calculate sentiment
    df_tweets_raw = make_df_of_raw_tweet_text(hashtag)
    df_tweets_sentiments = make_df_with_analyses(df_tweets_raw)
    overall_sentiment_score = round(df_tweets_sentiments['average_compound_vader_score'].mean(), 3)

    return jsonify({"sentiment_score": overall_sentiment_score})

@app.route('/get_data', methods=['GET'])
def get_sentiment_df():
    hashtag = request.args.get('hashtag', default=None)
    if hashtag is None:
        return jsonify({"error": "Please provide a hashtag parameter"}), 400

    # Collect tweets and calculate sentiment
    df_tweets_raw = make_df_of_raw_tweet_text(hashtag)
    df_tweets_sentiments = make_df_with_analyses(df_tweets_raw)

    return jsonify(df_tweets_sentiments.to_dict())


# Running the module will execute three things - a historic run for both vader and textblob, and a real-time stream for vader as an MVP.
if __name__ == "__main__":
    app.run(debug=True)



