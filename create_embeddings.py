import pandas as pd
import pyarrow
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import numpy as np
import sys

def tokenize_and_remove_stopwords(text):
    '''
    Tokenize the text and remove the stopwords
    
    Input: Text of lenth n (type: str)
    Output: filtered_tokens - a list of tokens (type: list)
    '''
    
    tokens = word_tokenize(text)  # Tokenize the text
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]  # Remove stop words
    return filtered_tokens



def calculate_avg_embedding(list_words, model):
    '''
    Calculate the average word embeddings for a given list of words
    
    Inputs: List_words - a list of words for which to calculate the word embeddings (list)
            model - the pre-trained word2vec model
            
    Outputs: the average word emvedding for the list of words (np.array of size (300))
    '''
    
    len_l = len(list_words)
    empty_arr = np.empty([len_l,300])
    
    for i in range(len_l):
        
        try:
            empty_arr[i] = model[list_words[i]] 
        except:
            empty_arr[i] = np.zeros(300)
            
    
    mean_embedding = np.mean(empty_arr, axis = 0)
    
    return mean_embedding




if __name__ == "__main__":
    
    #dating or bff
    target_var = sys.argv[1]
    
    print("Here")
    #load pre-trained word2vec model
    path = api.load("word2vec-google-news-300", return_path=True)
    #print("Path is", path)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)
    
    print("There")
    
    #nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    #load the appropriate dataset (bff or dating)
    if target_var == 'bff':
        
        #read the data
        matches = pd.read_feather("data/bff_matches")
        stacked_interests = pd.read_feather("data/user_interests_bff")
        
    elif target_var == 'dating':
        
        #read the data
        matches = pd.read_feather("data/dating_matches")
        stacked_interests = pd.read_feather("data/user_interests_dating")
    
    else:
        print("You have entered an invalid input")
        quit()
        
    #include a column where 1 is voted yes and 0 is voted no
    matches['ACTIVE_VOTE_RESULT_BINARY'] = matches['ACTIVE_VOTE_RESULT'].apply(lambda x: 1 if x == 2 or x == 6 else 0)
    matches['PASSIVE_VOTE_RESULT_BINARY'] = matches['PASSIVE_VOTE_RESULT'].apply(lambda x: 1 if x == 2 or x == 6 else 0)
    matches['MATCH_BINARY'] = (matches['PASSIVE_VOTE_RESULT_BINARY'] + matches['ACTIVE_VOTE_RESULT_BINARY']) - 1
    matches['MATCH_BINARY'] = matches['MATCH_BINARY'].replace(-1, 0)
    
    
    #save as a feather file
    if target_var == 'bff':
        matches.to_feather('data/matches_processed_bff')
    
    
    if target_var == 'dating':
        matches.to_feather('data/matches_processed_dating')
        
    
    #drop the duplicates interests
    interests_df = stacked_interests[['INTEREST_ID', 'INTEREST_NAME']]
    interests_df = interests_df.drop_duplicates()

    #tokenize and remove stop words
    interests_df['TOKENIZED_INTEREST_NAME'] = interests_df['INTEREST_NAME'].apply(tokenize_and_remove_stopwords)

    #word2vec embeddings
    interests_df['interest_embedding'] = interests_df.apply(lambda x: calculate_avg_embedding(x.TOKENIZED_INTEREST_NAME, word2vec_model), axis=1)

    interests_df = interests_df.reset_index(drop = True)
    
    
    #save the interests
    if target_var == 'bff':
        
        interests_df.to_feather("data/interests_table_bff")
        
    if target_var == 'dating':
        
        interests_df.to_feather("data/interests_table_dating")
        

        
        