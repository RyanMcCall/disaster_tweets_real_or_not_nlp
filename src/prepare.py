################## IMPORTS ##############################
# Used for file existance validation
import os

# Dataset manipulation resource
import pandas as pd

# Resources for NLP
import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# Modelling resources
from sklearn.model_selection import train_test_split

#########################################################

def data_exists():
    return os.path.isfile("data/prepared/train_train.csv") and os.path.isfile("data/prepared/train_val.csv") and os.path.isfile("data/prepared/train_test.csv") and os.path.isfile('data/prepared/test.csv')

def split_data():
    data = pd.read_csv('data/raw/train.csv')

    train_val, test_df = train_test_split(data, test_size=.2, random_state=13, stratify=data.target)
    train_df, val_df = train_test_split(train_val, test_size=.25, random_state=13, stratify=train_val.target)
    
    return train_df, val_df, test_df

def basic_text_clean(string):
    string = string.lower()
    string = (unicodedata.normalize('NFKD', string)
                         .encode('ascii', 'ignore')
                         .decode('utf-8', 'ignore')
             )
    string = re.sub(r"[^a-z0-9'\s]", '', string)
    return string

def tokenize(string):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(string, return_str=True)

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    stopword_list = stopwords.words('english')
    
    for word in extra_words:
        stopword_list.append(word)
    
    for word in exclude_words:
        stopword_list.remove(word)
        
    words = string.split()
    filtered_words = [word for word in words if word not in stopword_list]
    return ' '.join(filtered_words)

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    return ' '.join(stems)

def lemmatize(string):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    return ' '.join(lemmas)

def fix_keyword_spacing(string):
    return re.sub(r'%20', ' ', string)

def prepare_dataframe(df):
    clean_tokens = (df.text.apply(basic_text_clean)
                          .apply(tokenize)
                          .apply(remove_stopwords)
    )

    for token in clean_tokens:
        token = ' '.join(token).split()

    df['cleaned'] = clean_tokens
    df['stemmed'] = clean_tokens.apply(stem)
    df['lemmatized'] = clean_tokens.apply(lemmatize)

    df.keyword = df.keyword.fillna("No keyword")
    df.location = df.location.fillna("No location")
    df.keyword = df.keyword.apply(fix_keyword_spacing)
    
    return df

def store_data(train_df, val_df, test_df, true_test_df):
    train_df.to_csv('data/prepared/train_train.csv', index=False)
    val_df.to_csv('data/prepared/train_val.csv', index=False)
    test_df.to_csv('data/prepared/train_test.csv', index=False)
    true_test_df.to_csv('data/prepared/test.csv', index=False)
    
def prepare_data():
    train, val, test = split_data()
    true_test = pd.read_csv('data/raw/test.csv')
    
    train = prepare_dataframe(train)
    val = prepare_dataframe(val)
    test = prepare_dataframe(test)
    true_test = prepare_dataframe(true_test)
    
    store_data(train, val, test, true_test)

########### MAIN FUNCTION ##########################

def run():
    print("Prepare: Cleaning acquired data...")
    if data_exists():
        print("Data already prepared")
    else:
        prepare_data()
    print("Prepare: Completed!")

####################################################