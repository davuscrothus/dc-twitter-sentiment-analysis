import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify

# Input the important user credentials, ideally these would be encrypted
TWITTER_X_CONSUMER_KEY = ''
TWITTER_X_CONSUMER_SECRET = '' 
TWITTER_X_ACCESS_TOKEN = ''
TWITTER_X_ACCESS_TOKEN_SECRET = ''

FLASK_APP_NAME = ''

# Set Up Twitter/X Authorisation for all further calls.      
auth = tweepy.OAuthHandler(TWITTER_X_CONSUMER_KEY, TWITTER_X_CONSUMER_SECRET)
auth.set_access_token(TWITTER_X_ACCESS_TOKEN, TWITTER_X_ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

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

def preprocess_text(tweet):
    tweet = tweet.lower()
    tokens = word_tokenize(tweet)
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
    sentiment_scores = vader_analyzer.polarity_scores(tweet.text)
    return sentiment_scores

# Search for tweets
def search_tweets(query, num_tweets):
    tweets = []
    for tweet in tweepy.Cursor(api.search, q=query, lang="en").items(num_tweets):
        tweets.append(tweet)
    return tweets

# Define an app route for the sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment_api():
    data = request.get_json()
    search_query = data.get('search_query', '')
    num_tweets_to_analyze = data.get('num_tweets', 10)

    tweets = search_tweets(search_query, num_tweets_to_analyze)
    results = []

    for tweet in tweets:
        sentiment_scores = perform_vader_sentiment_analysis(tweet)
        results.append({
            'tweet_text': tweet.text,
            'sentiment_scores': sentiment_scores
        })

    return jsonify({'sentiment_results': results})


# Running the module will execute three things - a historic run for both vader and textblob, and a real-time stream for vader as an MVP.
if __name__ == "__main__":
    app.run(debug=True)
    search_query = "LEGO"  # Replace with the desired search query
    num_tweets_to_analyze = 10  # Replace with the number of tweets to analyze

    tweets = search_tweets(search_query, num_tweets_to_analyze)

    for tweet in tweets:
        textblob_sentiment_scores = perform_textblob_sentiment_analysis(tweet)
        print("Tweet:", tweet.text)
        print("Sentiment Scores:", textblob_sentiment_scores)
        print()




