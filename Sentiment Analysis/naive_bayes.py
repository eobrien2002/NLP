from nltk import classify
from nltk import NaiveBayesClassifier
from prep_training_data import train_df, test_df
from data_cleaning import df
import matplotlib.pyplot as plt
import seaborn as sns


def create_classifier(train_df, test_df):
    # Convert the dataframes into lists of tuples
    train_data = [(create_features_dict(x['text']), x['label']) for _, x in train_df.iterrows()]
    test_data = [(create_features_dict(x['text']), x['label']) for _, x in test_df.iterrows()]

    # Create the classifier
    classifier = NaiveBayesClassifier.train(train_data)

    # Test the classifier
    accuracy = classify.accuracy(classifier, test_data)
    print("Accuracy is: ", accuracy)

    # Print the most informative features
    classifier.show_most_informative_features(20)

    return classifier


# Helper function to create a dictionary of features and their values for a given text
def create_features_dict(text):
    words = set(text.split())
    return {word: True for word in words}


classifier = create_classifier(train_df, test_df)







#Now lets look at the results of the model on the SMS data

sentiments = []

# Iterate through the column and predict each response's sentiment, append sentiment to new list
for message in df['text']:
    sentiments.append(str((classifier.classify(dict([token, True] for token in message)))))

# add the list back to our DataFrame
df['Sentiment_NB'] = sentiments

df['Sentiment_NB'].value_counts()




df.Sentiment_NB=df.Sentiment_NB.apply(lambda x: int(x))

df_sent = df.groupby(['country']).Sentiment_NB.mean().reset_index()

df_sent.sort_values(by='Sentiment_NB') 

fig, ax = plt.subplots(figsize=(40,15))
sns.barplot(x='country', y='Sentiment_NB', data=df_sent,ax=ax)
plt.show()

