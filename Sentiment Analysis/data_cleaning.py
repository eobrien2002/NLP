import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

df = pd.read_csv('clean_nus_sms.csv', index_col=0)
df.dropna(subset='Message',inplace=True)
df['text']=df.Message 

def process(df):
        
    
    df['clean_message']=df.text.apply(lambda x: x.lower())
        #Using regular expressions to remove any URLs

    def remove_urls(text):
            url_pattern = re.compile(r'https?://\S+|www\.\S+')
            return url_pattern.sub(r'', text)

    df.clean_message = df.clean_message.apply(lambda text: remove_urls(text))

    def remove_junk(text):
        #remove special characters
        special_pattern=re.compile(r'[^a-zA-Z0-9\s]')
        text = special_pattern.sub(r'', text)

        username_pattern=re.compile(r'@\w+')
        text = username_pattern.sub(r'', text)

        hashtag_pattern=re.compile(r'#\S+')
        text = hashtag_pattern.sub(r'', text)
            
        return text

    df.clean_message = df.clean_message.apply(lambda text: remove_junk(text))


    def remove_html(text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    
    df.clean_message = df.clean_message.apply(lambda text: remove_html(text))

    df['message_tokenized'] = df.apply(lambda x: word_tokenize(x['clean_message']), axis=1)

    def lemmatize(tokens):
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
        

    df['text']=df.message_tokenized.apply(lambda x: lemmatize(x))
    
    df.text=df.text.apply(lambda x: ' '.join(x))
    df = df[df['text'].apply(lambda x: len(x.split()) > 5)]

    if 'country' in df.columns:
        df = df.replace({'country':{'SG':'Singapore', 
                            'USA':'United States',
                            'india':'India',
                            'INDIA':'India',
                            'srilanka':'Sri Lanka',
                            'UK':'United Kingdom',
                            'BARBADOS':'Barbados',
                            'jamaica':'Jamaica',
                            'MY':'Malaysia',
                            'unknown':'Unknown'}})

    return df



df = process(df)

