import pandas as pd
import pyarrow
import numpy as np
import random 
from collections import defaultdict
import sys

#Functions and classes
def user_counts(matches, active_user_id, passive_user_id, column_name):
    
    '''
    This function counts how many passive users there are per each active user
    
    Inputs: 
        matches: a dataset with matches(type: pandas dataframe)
        active_user_id: the name of the column by which we want to group (type: str)
        passive_user_id: the name of the column that we want to count (type: str)
        column_name: the desired name for the newly created column (type: str)
        
    Output: 
        - a dataframe with in the shape of matches with an additional column which shows
        the count by the group column
    '''
    
    # Group by 'active_user' and count the number of passive users for each active user
    user_counts = matches.groupby(active_user_id)[passive_user_id].count().reset_index()
    
    # Rename the column to indicate the count of passive users
    user_counts.rename(columns={passive_user_id: column_name}, inplace=True)
    
    return matches.merge(user_counts, on=active_user_id)




def remove_one_sign_smaples(matches, active_user_id, outcome_column):
    
    '''
    This function removes all the rows where for an active user we only
    have positive or negative samples
    
    Inputs:
        - matches: the dataframe that contains the user pairs (type: dataframe)
        - active_user_id: the column name of the column to group by (type: str)
        - outcome_column: the column name of the outcome variable (type: str)
        
    Outputs: 
        - filtered_df: a dataframe that is the same shape as matches however,
        with the users that have only positive or only negative samples removed
        
    '''
    
    grouped = matches.groupby(active_user_id)[outcome_column].unique()

    valid_queries = grouped[(grouped.apply(lambda x: len(x) > 1))].index
    filtered_df = matches[matches[active_user_id].isin(valid_queries)]
    
    return filtered_df


def all_zeros(embeddings_array):
    '''
    Check if all elements in the input numpy array are equal to zero.

    This function takes a numpy array `embeddings_array` as input and checks whether
    all elements in the array are equal to zero. It returns True if all elements are
    zeros, and False otherwise.

    Inputs:
        - embeddings_array: the input numpy array containing the embeddings to be checked (type: numpy.ndarray)
        

    Outputs:
        - True if all elements in the array are zeros, False otherwise (type: bool)
    '''
    return np.all(embeddings_array == 0)



if __name__ == "__main__":
    
    #dating or bff
    target_var = sys.argv[1]
    
    if target_var == 'bff':
        
        #load the data to be pre-processed
        matches = pd.read_feather("data/matches_processed_bff")
        user_interests = pd.read_feather("data/user_interests_bff")
        interests = pd.read_feather("data/interests_table_bff")
        
    elif target_var == 'dating':
        
        #load the data to be pre-processed
        matches = pd.read_feather("data/matches_processed_dating")
        user_interests = pd.read_feather("data/user_interests_dating")
        interests = pd.read_feather("data/interests_table_dating")
        
    else:
        print("You have entered an invalid input")
        quit()
        
        
    
    #merge the user_interests table with the interest table to ge the embeddings
    user_interest_new = user_interests.merge(interests, on = ['INTEREST_ID', 'INTEREST_NAME'], how = 'inner')
    #get a single row by user
    user_level_interest = user_interest_new.groupby(['USER_ID']).agg({
        'INTEREST_ID': list,
        'interest_embedding': list,
        'INTEREST_NAME': list
    }).reset_index()

    user_level_interest['avg_interest_embeddings'] = user_level_interest.apply(lambda x: np.mean(x.interest_embedding, axis = 0), axis = 1)
    
    if target_var == 'bff':
        #save the dataset
        user_level_interest.to_feather("data/user_level_interest_bff")
    
    if target_var == 'dating':
        #save the dataset
        user_level_interest.to_feather("data/user_level_interest_dating")

    print("Saved the user level interest table")
    
    
    # if we want to use the match colmn then we need to remove all NaNs for it
    matches = matches[(matches['PASSIVE_VOTE_RESULT'].notna()) & (matches['ACTIVE_VOTE_RESULT'].notna())]

    #find all the zero rows and corresponding IDs
    zero_rows = user_level_interest[user_level_interest["avg_interest_embeddings"].apply(all_zeros)]
    zero_user_ids = zero_rows['USER_ID'].to_list()

    #keep only those who have the embeddings that are not all 0s
    new_filtered_df = matches[~matches["ACTIVE_USER_ID"].isin(zero_user_ids)]
    matches = new_filtered_df[~new_filtered_df["PASSIVE_USER_ID"].isin(zero_user_ids)]

    #count users
    matches = user_counts(matches, 'ACTIVE_USER_ID', 'PASSIVE_USER_ID', 'passive_user_count')
    matches = user_counts(matches, 'PASSIVE_USER_ID', 'ACTIVE_USER_ID', 'active_user_count')

    above_10 = matches[(matches['passive_user_count'] >= 10) & (matches['active_user_count'] >= 10)]

    #removing only positive and only negative samples
    filtered_df_matches = remove_one_sign_smaples(above_10, 'ACTIVE_USER_ID', 'MATCH_BINARY').reset_index()
    filtered_df_one_way = remove_one_sign_smaples(above_10, 'ACTIVE_USER_ID', 'ACTIVE_VOTE_RESULT_BINARY').reset_index()
    
    if target_var == 'bff':
        #save the datasets
        filtered_df_matches.to_feather("data/filtered_df_matches_bff")
        filtered_df_one_way.to_feather("data/filtered_df_one_way_bff")
    
    if target_var == 'dating':
        #save the datasets
        filtered_df_matches.to_feather("data/filtered_df_matches_dating")
        filtered_df_one_way.to_feather("data/filtered_df_one_way_dating")
        
    
    print("Saved the matches table")
        
        
    
    




