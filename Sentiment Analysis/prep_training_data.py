import nltk
from nltk.corpus import twitter_samples
import pandas as pd
from data_cleaning import process
from sklearn.model_selection import train_test_split

#I will be training the model on sentiment analysis of tweets from e NLTK. I will be using the same cleaning process as I did for the SMS data.
nltk.download('twitter_samples', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

# Getting all positive tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
# Getting all negative tweets
negative_tweets = twitter_samples.strings('negative_tweets.json')

#combine the positive and negative tweets into one list and make a dataframe. First we need to add a column to each list to indicate the sentiment
positive_tweets = [[tweet, 1] for tweet in positive_tweets]
negative_tweets = [[tweet, 0] for tweet in negative_tweets]
tweets = positive_tweets + negative_tweets
tweets_df = pd.DataFrame(tweets, columns=['text', 'label'])

tweets_df=process(tweets_df)

train_df, test_df = train_test_split(tweets_df, test_size=0.2, random_state=42, stratify=tweets_df['label'])
