import pandas as pd
import pyarrow
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
import random 
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
import sys
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm


class CustomDataset(Dataset):
    """
    Custom PyTorch dataset for handling tabular data for user matching.

    Args:
        data_frame (pandas.DataFrame): The input data containing user profiles and binary match information.
        target_column (str): The name of the column containing binary match labels.

    Attributes:
        data (pandas.DataFrame): The input data frame.
        target_column (str): The name of the binary match label column.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index): Returns user IDs and binary match label for a specific index.

    """
    
    def __init__(self, data_frame, target_column):
        self.data = data_frame
        self.target_column = target_column

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id_1 = self.data.iloc[index]['ACTIVE_USER_ID']
        user_id_2 = self.data.iloc[index]['PASSIVE_USER_ID']
        binary_match = self.data.iloc[index][self.target_column]

        return user_id_1, user_id_2, binary_match
    
    

def prepare_dataloader(df, relevancy_column):
    """
    Prepare train, validation, and test datasets

    Inputs:
        df (pandas.DataFrame): The input DataFrame containing user data and relevancy information.
        relevancy_column (str): The name of the column indicating user relevancy or grouping.

    Output:
        train_df (pandas.DataFrame): The training dataset.
        validation_df (pandas.DataFrame): The validation dataset.
        test_df (pandas.DataFrame): The test dataset.
    """
    #get the unique number of active users
    unique_active_users = df[relevancy_column].unique()

    # Perform group-wise splitting
    train_queries, remaining_queries = train_test_split(unique_active_users, train_size=0.7)
    validation_queries, test_queries = train_test_split(remaining_queries, train_size=0.5)

    # Create the final train, validation, and test datasets
    train_df = df[df[relevancy_column].isin(train_queries)]
    validation_df = df[df[relevancy_column].isin(validation_queries)]
    test_df = df[df[relevancy_column].isin(test_queries)]
    
    return train_df, validation_df, test_df



def group_by_query(query_ids, predicted_scores, true_relevance_scores):
    """
    Group predicted scores and true relevance scores by query IDs.

    Inputs:
        query_ids (list): List of query IDs for grouping.
        predicted_scores (list): List of predicted scores for each query.
        true_relevance_scores (list): List of true relevance scores for each query.

    Output:
        grouped_predictions (dict): A dictionary with query IDs as keys and lists of predicted scores as values.
        grouped_relevance_scores (dict): A dictionary with query IDs as keys and lists of true relevance scores as values.
    """
    grouped_predictions = defaultdict(list)
    grouped_relevance_scores = defaultdict(list)

    for query_id, score, relevance in zip(query_ids, predicted_scores, true_relevance_scores):
        grouped_predictions[query_id.item()].append(score.item())
        grouped_relevance_scores[query_id.item()].append(relevance.item())

    return grouped_predictions, grouped_relevance_scores


def sort_documents_by_score(grouped_predictions, grouped_relevance_scores, document_ids):
    
    """
    Sort documents by predicted scores and organize true relevance scores accordingly.

    Inputs:
        grouped_predictions (dict): A dictionary with query IDs as keys and lists of predicted scores as values.
        grouped_relevance_scores (dict): A dictionary with query IDs as keys and lists of true relevance scores as values.
        document_ids (torch.Tensor): A tensor containing document IDs for the corresponding scores.

    Output:
        sorted_document_ids (dict): A dictionary with query IDs as keys and tensors of sorted document IDs.
        sorted_scores (dict): A dictionary with query IDs as keys and tensors of sorted predicted scores.
        true_relevance (dict): A dictionary with query IDs as keys and tensors of true relevance scores in the same order as the sorted documents.
    """
    
    sorted_document_ids = {}
    sorted_scores = {}
    true_relevance = {}

    for query_id, scores in grouped_predictions.items():
        relevance = grouped_relevance_scores[query_id]
        sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        sorted_document_ids[query_id] = document_ids.clone().detach()[sorted_indices]
        sorted_scores[query_id] = torch.tensor(scores)[sorted_indices]
        true_relevance[query_id] = torch.tensor(relevance)[sorted_indices]

    return sorted_document_ids, sorted_scores, true_relevance


def calculate_ndcg(true_relevance_scores, predicted_scores, k=None):
    
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG) score.

    Inputs:
        true_relevance_scores (torch.Tensor or list): True relevance scores for each item.
        predicted_scores (torch.Tensor or list): Predicted scores for each item.
        k (int, optional): The number of items to consider for NDCG calculation. If None, all items are considered.

    Output:
        ndcg (float): The computed NDCG score.
    """
    
    true_relevance_scores_list = true_relevance_scores.tolist()
    predicted_scores_list = predicted_scores.tolist()
    
    true_relevance_scores_list = np.asarray([true_relevance_scores_list])
    predicted_scores_list = np.asarray([predicted_scores_list])
    
    return ndcg_score(true_relevance_scores_list, predicted_scores_list, k=k)


def calculate_map(true_relevance_scores, predicted_scores):
    """
    Calculate the Mean Average Precision (MAP) score.
    Inputs:
        true_relevance_scores (array-like): True relevance scores for items in the ranked list.
        predicted_scores (array-like): Predicted scores for items in the ranked list.

    Output:
        map_score (float): The computed Mean Average Precision (MAP) score.
    """
    return average_precision_score(true_relevance_scores, predicted_scores)


def evaluate_model(sorted_document_ids, true_relevance, sorted_scores):
    """
    Evaluate a ranking model's performance using Mean Average Precision (mAP) and Mean Normalized Discounted Cumulative Gain (NDCG) scores.

    Inputs:
        sorted_document_ids (dict): A dictionary with query IDs as keys and tensors of sorted document IDs as values.
        true_relevance (dict): A dictionary with query IDs as keys and tensors of true relevance scores.
        sorted_scores (dict): A dictionary with query IDs as keys and tensors of predicted scores.

    Output:
        mean_mAP (float): The mean Average Precision (mAP) score across all queries.
        mean_NDCG (float): The mean Normalized Discounted Cumulative Gain (NDCG) score across queries.

    """
    
    mAP_scores = []
    NDCG_scores = []
    
    for query_id in sorted_document_ids.keys():
    
        true_relevance_scores_query = true_relevance[query_id]
        predicted_scores_query = sorted_scores[query_id]

        mAP_scores.append(calculate_map(true_relevance_scores_query, predicted_scores_query))
        
        if len(true_relevance_scores_query) > 1:
            NDCG_scores.append(calculate_ndcg(true_relevance_scores_query, predicted_scores_query))
        
    
    mean_mAP = torch.tensor(mAP_scores).mean().item()
    mean_NDCG = torch.tensor(NDCG_scores).mean().item()
    
    return mean_mAP, mean_NDCG


def random_ordering_baseline(matches, id_column):  
    """
    Generate a random ordering for user pairs in a dataset.

    Inputs:
        matches (pandas.DataFrame): The dataset containing user matches.
        id_column (str): The name of the column used for grouping user pairs (e.g., 'ACTIVE_USER_ID').

    Output:
        random_order_matches (pandas.DataFrame): The dataset with random user pair ordering.

    """
    
    #rank the user pairs in a random order
    random_order_matches = matches.groupby(id_column).apply(lambda x: x.sample(frac=1)).reset_index(drop=True)

    # Add a new column 'RANK' to show the ranking within each 'ACTIVE_USER_ID' group
    random_order_matches['RANK'] = random_order_matches.groupby(id_column).cumcount() + 1
    
    #Assign a score for ranking
    random_order_matches['SCORE'] =  1 / random_order_matches['RANK']
    
    return random_order_matches


def popularity_based_1w_baseline(input_data, votes_column, name_column):
    """
    This function groups the input data by 'PASSIVE_USER_ID', calculates the count of votes with value 1 and the total votes
    for each passive user, and then computes a score based on the ratio of votes with value 1 to total votes. This score
    represents the popularity of each passive user in the context of votes with value 1.

    Inputs:
        input_data (pandas.DataFrame): The input dataset containing user and vote information.
        votes_column (str): The name of the column containing vote values.
        name_column (str): The name of the column to store the calculated popularity score.

    Output:
        one_way_final (pandas.DataFrame): The dataset with the calculated popularity scores for passive users.
    """
    
    # Group by PASSIVE_USER_ID and calculate the count of ACTIVE_VOTE_RESULT_BINARY with value 1 and the total votes for each passive user
    one_way_df = input_data.groupby('PASSIVE_USER_ID').agg(
    total_votes=(votes_column, 'count'),
    votes_value_1=(votes_column, lambda x: (x == 1).sum())
    )
    
    #calculate the passive user's score
    one_way_df[name_column] = one_way_df['votes_value_1'] / one_way_df['total_votes']
    
    #mege the original dataset with the scores
    one_way_final = input_data.merge(one_way_df, on = 'PASSIVE_USER_ID', how = 'inner')
    
    return one_way_final


def evaluate_model_performance(dataset, target, score):
    
    """
    Evaluate the performance of a model using ranking metrics such as Mean Average Precision (mAP) and Mean Normalized Discounted Cumulative Gain (NDCG).

    Inputs:
        dataset (pandas.DataFrame): The dataset containing user pairs and relevant information.
        target (str): The name of the column containing the true relevance scores.
        score (str): The name of the column containing the model's predicted scores.

    Output:
        mAP (float): The computed Mean Average Precision (mAP) score.
        mean_NDCG (float): The computed Mean Normalized Discounted Cumulative Gain (NDCG) score.

    """
    
    user_1 = torch.tensor(dataset['ACTIVE_USER_ID'].to_list())
    user_2 = torch.tensor(dataset['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(dataset[target].to_list())
    predictions = torch.tensor(dataset[score].to_list())
    
    #evaluate the model 
    grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)
    
    #Get the evaluation metrics scores
    mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    return mAP, mean_NDCG


def calculate_cosine_similarity(q_v, d_v):
    '''
    A function that calculates the cosine similarity between 2 word embeddings
    
    Inputs: q_v - word embedding (np.array of size n)
            d_v - word embedding (np.array of size n)
    Output: 
            cos_sim - the cosine similarity between the 2 embeddings (int)
    '''
    cos_sim = np.dot(q_v, d_v) / (np.linalg.norm(q_v)*np.linalg.norm(d_v))
    
    return cos_sim


def get_sim_scores(user_id_1, user_id_2):
    """
    Calculate similarity scores between pairs of users based on their embeddings.

    Inputs:
        user_id_1 (torch.Tensor or list): A tensor or list of user IDs for the first set of users.
        user_id_2 (torch.Tensor or list): A tensor or list of user IDs for the second set of users.

    Output:
        final_scores (torch.Tensor): A tensor containing the calculated similarity scores.

    Example:
        similarity_scores = get_sim_scores(user_ids_1, user_ids_2)
    """
    
    scores = []
    
    for i in range(len(user_id_1)):
        
        user_1_embd = user_embedding_dict[user_id_1[i].item()]
        user_2_embd = user_embedding_dict[user_id_2[i].item()]
        
        #calculate the cosine similarity score
        sim_score = calculate_cosine_similarity(user_1_embd, user_2_embd)
        scores.append(sim_score)

    final_scores = torch.tensor(scores, dtype=torch.float32)
    
    return final_scores


class ShallowNN(nn.Module):
    
    """
    A simple Shallow Neural Network model.

    Args:
        layer_size (int): The number of neurons in the hidden layer.

    Attributes:
        layer1 (nn.Linear): The first linear layer responsible for transforming the input.
        layer2 (nn.Linear): The second linear layer for producing the final output.

    Methods:
        forward(x): Defines the forward pass of the model.
    """
    
    def __init__(self, layer_size):
        super(ShallowNN, self).__init__()
        self.layer1 = nn.Linear(1, layer_size)
        self.layer2 = nn.Linear(layer_size, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        
        return x
    
    
def make_prediction(shallow_model, val_loader, val_df):
    """
    Make predictions using a Shallow Neural Network model and a validation dataset.

    Args:
        shallow_model (ShallowNN): The trained Shallow Neural Network model for making predictions.
        val_loader (DataLoader): The data loader for the validation dataset.
        val_df (pandas.DataFrame): The validation dataset containing user pairs and binary match labels.

    Returns:
        avg_val_loss (float): The average validation loss.
        user_1 (torch.Tensor): User IDs from the first group in the validation dataset.
        user_2 (torch.Tensor): User IDs from the second group in the validation dataset.
        true_labels (torch.Tensor): True binary match labels for the user pairs.
        predictions (torch.Tensor): Model predictions for the user pairs.
    """
    
    true_labels = []
    user_1 = []
    user_2 = []
    predictions = []
    example_before = []

    
    shallow_model.eval()
    with torch.no_grad():

        val_loss = 0.0

        for i, (user_id_1, user_id_2, binary_match) in tqdm.tqdm(enumerate(val_loader), total=len(val_df)//batch_size+1):
            
            
            #get the similarity score for the user pair
            inputs = get_sim_scores(user_id_1, user_id_2)

            binary_match = binary_match.unsqueeze(1).float()
            inputs = inputs.unsqueeze(1)

            inputs, binary_match = inputs.to(device), binary_match.to(device)
            outputs = shallow_model(inputs)
            loss = criterion(outputs, binary_match)
            val_loss += loss.item()
            
            true_labels.append(binary_match.squeeze())
            user_1.append(user_id_1)
            user_2.append(user_id_2)
            predictions.append(outputs.squeeze())
            

        avg_val_loss = val_loss / len(val_loader)
    
        user_1 = torch.cat(user_1, dim=0)
        user_2 = torch.cat(user_2, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        predictions = torch.cat(predictions, dim=0)
    
    return avg_val_loss, user_1, user_2, true_labels, predictions


def train_model(train_loader, train_df, shallow_model):
    """
    Train a Shallow Neural Network model using a training dataset and a data loader.

    Args:
        train_loader (DataLoader): The data loader for the training dataset.
        train_df (pandas.DataFrame): The training dataset containing user pairs and binary match labels.
        shallow_model (ShallowNN): The Shallow Neural Network model to be trained.

    Returns:
        shallow_model (ShallowNN): The trained Shallow Neural Network model.
        avg_loss (float): The average training loss.
        user_1 (torch.Tensor): User IDs from the first group in the training dataset.
        user_2 (torch.Tensor): User IDs from the second group in the training dataset.
        true_labels (torch.Tensor): True binary match labels for the user pairs.
        predictions (torch.Tensor): Model predictions for the user pairs.
    """
    
    shallow_model.train()
    total_loss = 0.0
    
    true_labels = []
    user_1 = []
    user_2 = []
    predictions = []
    example_before = []
    
    for i, (user_id_1, user_id_2, binary_match) in tqdm.tqdm(enumerate(train_loader), total=len(train_df)//batch_size+1):
        
        #get the similarity score for the user pair
        inputs = get_sim_scores(user_id_1, user_id_2)
        binary_match = binary_match.unsqueeze(1).float()
        inputs = inputs.unsqueeze(1)
        
        inputs, binary_match = inputs.to(device), binary_match.to(device)
        
        
        optimizer.zero_grad()
        outputs = shallow_model(inputs)
        
        loss = criterion(outputs, binary_match)
        
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item()
        true_labels.append(binary_match.squeeze())
        user_1.append(user_id_1)
        user_2.append(user_id_2)
        predictions.append(outputs.squeeze())

        
        
    avg_loss = total_loss / len(train_loader)
    
    user_1 = torch.cat(user_1, dim=0)
    user_2 = torch.cat(user_2, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    
    return shallow_model, avg_loss, user_1, user_2, true_labels, predictions


def evaluate_model_performance(dataset, target, score):
    """
    Evaluate the performance of a model using ranking metrics such as Mean Average Precision (mAP) and Mean Normalized Discounted Cumulative Gain (NDCG).

    Args:
        dataset (pandas.DataFrame): The dataset containing user pairs and relevant information.
        target (str): The name of the column containing the true relevance scores.
        score (str): The name of the column containing the model's predicted scores.

    Returns:
        mAP (float): The computed Mean Average Precision (mAP) score.
        mean_NDCG (float): The computed Mean Normalized Discounted Cumulative Gain (NDCG) score.

    """
    
    user_1 = torch.tensor(dataset['ACTIVE_USER_ID'].to_list())
    user_2 = torch.tensor(dataset['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(dataset[target].to_list())
    predictions = torch.tensor(dataset[score].to_list())
    
    #evaluate the model 
    grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)
    
    #Get the evaluation metrics scores
    mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    return mAP, mean_NDCG


def simple_binary_similarity(interest_set_1, interest_set_2):
    '''
    This functions takes 2 lists and check if there is on overlap in at least one of the elements
    
    Inputs:
    *interest_set_1: the set of interest of user 1 (type:list)
    *interest_set_2: the set of interests of user 2 (type:list)
    
    Output: 
    *True or False (boolean)
    True is returned if there is at least one element in common between the 2 lists
    False if all the elements in the 2 sets are different
    '''
    set1 = set(interest_set_1)
    set2 = set(interest_set_2)
    
    if len(set1.intersection(set2)) > 0:
        return 1
    else:
        return 0
    
    
def binary_embedding_similarity(pair_cosine, interest_pairs):
    '''
    This function calculates the binary cosine similarity for a set of word embeddings
    If any pair of word embeddings have a cosine similarity which is larger than a certain 
    threshold, then we assign a similarity of 1, otherwise we assign a similiarity of 0
    
    Inputs: pair_cosine - the cosine similarity score of all the pairs of word embeddings (type: list)
            interest_pairs - a list with all the interest pairs (type: list)
    
    Output: 1 if at least one set of pairs is above a certain threshold
            0 otherwise
    '''
    
    for i in range(len(pair_cosine)): 
        
        if pair_cosine[i] >= 0.6: 
            
            return 1
    
    return 0


def pairwise_cosine_similarity(set_avg_embedding_1, set_avg_embedding_2, interest_set_1, interest_set_2):
    '''
    A function that takes in as inputs 2 sets of word embeddings and for each of the
    word embedding pairs from the 2 sets, it calculates the cosine similarity
    
    Inputs: set_avg_embedding_1 - a set of word embeddings (type: list)
            set_avg_embedding_1 = a set of word embeddings (type: list)
            
    Output: 
            pair_cosine - a list of the cosine similarity between each of the word embedding
            pairs from the 2 sets (type: list)
    
    '''
    
    pair_cosine = []
    pairs = []
    
    for i in range(len(set_avg_embedding_1)):
        
        for j in range(len(set_avg_embedding_2)):
            
            pair_cosine.append(calculate_cosine_similarity(set_avg_embedding_1[i], set_avg_embedding_2[j])) #calculate the cosine similarity between a pair of word embeddings
            pairs.append([interest_set_1[i],interest_set_2[j]])
    
    return pair_cosine, pairs



def simple_cosine_similarity(interest_set_1, interest_set_2):
    '''
    This funtion calculates the simple cosine similarity between 2 input sets of interests
    
    Inputs:
    *interest_set_1: the set of interest of user 1 (type:list)
    *interest_set_2: the set of interests of user 2 (type:list)
    
    Output:
    *cosine similarity score (type: int)
    '''
    
    set1 = set(interest_set_1)
    set2 = set(interest_set_2)
    
    l_u = len(interest_set_1) #number of interests of user u
    l_v = len(interest_set_2) #number of interests of user v
    
    l_uv = len(set1.intersection(set2)) #number of shared interests between user u and user v
    
    #calculate cosine similarity
    cos_sim = l_uv/(np.sqrt(l_u)*np.sqrt(l_v))
    
    return cos_sim


def simple_weighted_cosine_similarity(interest_set_1, interest_set_2, interest_dict):
    '''
    This function calcluates the weighted cosine similarity between 2 sets of interests where the
    weight of shared interests is defined as w = 1/log(N) where N is the frequency of occurance of the 
    interest
    
    Inputs:
    *interest_set_1: the set of interest of user 1 (type:list)
    *interest_set_2: the set of interests of user 2 (type:list)
    *interest_dict: a dictionary which contains the interest_id as the key, and the occurance
    frequency as the value
    
    Output:
    *weighted_cos_sim : weighted cosine similarity score (type: int)
    
    '''
    set1 = set(interest_set_1)
    set2 = set(interest_set_2)
    
    l_u = len(interest_set_1) #number of interests of user u
    l_v = len(interest_set_2) #number of interests of user v
    
    intersection = set1.intersection(set2)
    
    sum_w = 0
    
    for i in intersection:
        w = 1/np.log(interest_dict[i])
        sum_w += w
        
    
    weighted_cos_sim = sum_w / (np.sqrt(l_u)*np.sqrt(l_v))
    
    return weighted_cos_sim


def cosine_embeddings_similarity(interest_set_1, interest_set_2, pair_cosine, interest_pairs):
    '''
    This function calclutates the cosine similarity between 2 sets of interests based on the word embeddings
    The denominator is the product of the number of interests in each set
    The nominator is the number of shared interests between the 2 sets. Interests are considered to be the same if
    the cosine similarity between the interets' word embedding are above a certain threshold
    
    Inputs: interest_set_1: the set of interest of user 1 (type:list)
            interest_set_2: the set of interests of user 2 (type:list)
            pair_cosine - the cosine similarity score of all the pairs of word embeddings (type: list)
            interest_pairs - a list with all the interest pairs (type: list)
            
    Output: cos_sim - the cosine similarity score between the 2 sets of interests (type: int)
    
    '''
    
    #count number of shared interest 
    l_uv = 0
    
    for i in range(len(pair_cosine)):
        
        if pair_cosine[i] >= 0.6:
            l_uv += 1 #think about what is a good threshold
        
    
    l_u = len(interest_set_1) #number of interests of user u
    l_v = len(interest_set_2) #number of interests of user v
    
    
    #calculate cosine similarity
    cos_sim = l_uv/(len(interest_set_1) * len(interest_set_2))
    
    return cos_sim


def max_similarity(pair_cosine, interest_pairs):
    '''
    This function calcultes a similarity score between 2 sets of word embeddings by
    calcualting each pair's cosine similarity score and taking the maximum cosine similarity score
    from all the pairs as the similarity score betweent the 2 interest sets
    
    Inputs: pair_cosine - the cosine similarity score of all the pairs of word embeddings (type: list)
            interest_pairs - a list with all the interest pairs (type: list)
            
    
    Output: max_cosine - the maximum cosine similarity score of all the pairs of word
            embeddings (type: int)
    
    '''
    
    #find the maximum cosine similarity
    max_cosine = max(pair_cosine)
    
    return max_cosine


if __name__ == "__main__":
    
    #dating or bff
    mode = sys.argv[1]
    #one_way or matches 
    type_var = sys.argv[2]
    
    
    #load the matches
    file_path_matches = 'data/' + 'filtered_df_' + type_var + '_' + mode
    filtered_df_matches = pd.read_feather(file_path_matches)

    #load the interests
    file_path_interests = 'data/' + 'interests_table_' + mode
    interests = pd.read_feather(file_path_interests)

    #load the user-level interests
    file_path_user_level_interests = 'data/' + 'user_level_interest_' + mode
    user_interests = pd.read_feather(file_path_user_level_interests) 
    
    
    #load the interests table and initiate the hash maps for easier access of embeddings
    user_interests.set_index('USER_ID', inplace=True)
    user_embedding_dict = user_interests['avg_interest_embeddings'].to_dict()
    user_embedding_dict_2 = user_interests['interest_embedding'].to_dict()
    user_interests = user_interests.reset_index()
    
    
    if type_var == 'matches':
        target = 'MATCH_BINARY'
    else:
        target = 'ACTIVE_VOTE_RESULT_BINARY'
        
    #prepare the dataloader
    train_df, validation_df, test_df = prepare_dataloader(filtered_df_matches, 'ACTIVE_USER_ID')

    batch_size = 512

    #train dataset
    train_dataset = CustomDataset(train_df, target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #validation dataset
    val_dataset = CustomDataset(validation_df, target)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #test dataset
    test_dataset = CustomDataset(test_df, target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    #random ordering baseline model
    mean_mAP = []
    mean_NDCG_list = []

    for i in range(10):

        random_order_val = random_ordering_baseline(validation_df, 'ACTIVE_USER_ID')

        #let now evaluate this model
        mAP, mean_NDCG = evaluate_model_performance(random_order_val, target, 'SCORE')
        mean_mAP.append(mAP)
        mean_NDCG_list.append(mean_NDCG)

    #take the average
    avg_random_mAP = sum(mean_mAP)/len(mean_mAP)
    avg_random_NDCG = sum(mean_NDCG_list)/len(mean_NDCG_list)
    
    
    
    #popularity based baseline
    if type_var == 'one_way':

        popularity_one_way = popularity_based_1w_baseline(validation_df, 'ACTIVE_VOTE_RESULT_BINARY', 'popularity_score')
        one_way_final_sorted = popularity_one_way.groupby('ACTIVE_USER_ID').apply(lambda x: x.sort_values('popularity_score', ascending=False))
        one_way_final_sorted.reset_index(drop=True, inplace=True)

        #let now evaluate this model
        popularity_mAP, popularity_mean_NDCG = evaluate_model_performance(one_way_final_sorted, target, 'popularity_score')

    if type_var == 'matches':

        #perform popularity based ordering
        propensity_one_way = popularity_based_1w_baseline(validation_df, 'PASSIVE_VOTE_RESULT_BINARY', 'propensity_score')
        both_way = popularity_based_1w_baseline(propensity_one_way, 'ACTIVE_VOTE_RESULT_BINARY', 'popularity_score')

        #calculate the average score of propensity to vote and popularity
        both_way['total_score'] = (both_way['propensity_score'] + both_way['popularity_score']) / 2

        #sort by the score
        two_way_final_sorted = both_way.groupby('ACTIVE_USER_ID').apply(lambda x: x.sort_values('total_score', ascending=False))
        two_way_final_sorted.reset_index(drop=True, inplace=True)


        #let now evaluate this model
        popularity_mAP, popularity_mean_NDCG = evaluate_model_performance(two_way_final_sorted, target, 'total_score')
        
        
    
    #order based on the score itself
    user_id_1 = torch.tensor(validation_df['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(validation_df['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(validation_df[target].to_list())

    #get predictions
    predictions = get_sim_scores(user_id_1, user_id_2)

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    
    
    # Instantiate the model
    shallow_model = ShallowNN(layer_size = 8)

    # Define loss function and optimizer
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(shallow_model.parameters(), lr=0.001)

    num_epochs = 30
    
    
    #check ig GPU is available
    if torch.cuda.is_available():

        shallow_model.cuda()
        criterion.cuda()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shallow_model.to(device)
    
    
    for epoch in range(num_epochs):
        
        print("Here")
        #train the model
        shallow_model, avg_loss, user_1, user_2, true_labels, predictions = train_model(train_loader, train_df, shallow_model)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        #evaluate the train set
        grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
        sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)

        mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)

        print("Traing mAP is", mAP)
        print("Traing NDCG score is", mean_NDCG)


        #make a prediction
        avg_val_loss, user_1, user_2, true_labels, predictions = make_prediction(shallow_model, val_loader, validation_df)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")


        #evaluate the model 
        grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
        sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)

        #Get the evaluation metrics scores
        mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
        
    
    model_path = 'shallow_model_match_date.pth'
    torch.save(shallow_model,model_path)
    
    loaded_model = torch.load('shallow_model_one_way_date.pth')
    
    #make a prediction
    avg_val_loss, user_1, user_2, true_labels, predictions = make_prediction(loaded_model, test_loader, test_df)

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    #evaluate the model 
    grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)

    #Get the evaluation metrics scores
    mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    
    #Evaluate the models
    
    #load back the model and run it on the test dataset
    loaded_model = torch.load('shallow_model_one_way_date.pth')
    
    #make a prediction
    avg_val_loss, user_1, user_2, true_labels, predictions = make_prediction(loaded_model, test_loader, test_df)

    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

    #evaluate the model 
    grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)

    #Get the evaluation metrics scores
    mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    print("The Neural Network MAP Score is", mAP)
    print("The Nuerail Network mean_NDCG score is", mean_NDCG)
    
    
    #random ordering baseline model
    mean_mAP = []
    mean_NDCG_list = []

    for i in range(10):

        random_order_val = random_ordering_baseline(test_df, 'ACTIVE_USER_ID')

        #let now evaluate this model
        mAP, mean_NDCG = evaluate_model_performance(random_order_val, target, 'SCORE')
        mean_mAP.append(mAP)
        mean_NDCG_list.append(mean_NDCG)

    #take the average
    avg_random_mAP = sum(mean_mAP)/len(mean_mAP)
    avg_random_NDCG = sum(mean_NDCG_list)/len(mean_NDCG_list)
    
    print("The Random Ordering MAP Score is", avg_random_mAP)
    print("The Random Ordering mean_NDCG score is", avg_random_NDCG)
    
    
    #popularity based baseline
    if type_var == 'one_way':

        popularity_one_way = popularity_based_1w_baseline(test_df, 'ACTIVE_VOTE_RESULT_BINARY', 'popularity_score')
        one_way_final_sorted = popularity_one_way.groupby('ACTIVE_USER_ID').apply(lambda x: x.sort_values('popularity_score', ascending=False))
        one_way_final_sorted.reset_index(drop=True, inplace=True)

        #let now evaluate this model
        popularity_mAP, popularity_mean_NDCG = evaluate_model_performance(one_way_final_sorted, target, 'popularity_score')

    if type_var == 'matches':

        #perform popularity based ordering
        propensity_one_way = popularity_based_1w_baseline(test_df, 'PASSIVE_VOTE_RESULT_BINARY', 'propensity_score')
        both_way = popularity_based_1w_baseline(propensity_one_way, 'ACTIVE_VOTE_RESULT_BINARY', 'popularity_score')

        #calculate the average score of propensity to vote and popularity
        both_way['total_score'] = (both_way['propensity_score'] + both_way['popularity_score']) / 2

        #sort by the score
        two_way_final_sorted = both_way.groupby('ACTIVE_USER_ID').apply(lambda x: x.sort_values('total_score', ascending=False))
        two_way_final_sorted.reset_index(drop=True, inplace=True)


        #let now evaluate this model
        popularity_mAP, popularity_mean_NDCG = evaluate_model_performance(two_way_final_sorted, target, 'total_score')
        
    print("Popularity MAP Score is", popularity_mAP)
    print("Popularity mean_NDCG score is", popularity_mean_NDCG)
    
    
    
    #order based on the score itself
    user_id_1 = torch.tensor(test_df['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(test_df['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(test_df[target].to_list())

    #get predictions
    predictions = get_sim_scores(user_id_1, user_id_2)

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    print("Average Embedding MAP Score is", sim_order_mAP)
    print("Average Embedding mean_NDCG score is", sim_order_mean_NDCG)
    
    
    #Simple Binary Similarity
    #subset the dataset to only include the relevant measures
    subset_df_binary = test_df[["ACTIVE_USER_ID", "PASSIVE_USER_ID",target]].merge(user_interests[["USER_ID", "INTEREST_NAME"]], 
                                                                             how = 'inner', left_on = "ACTIVE_USER_ID",
                                                                             right_on = "USER_ID").rename(columns = {'INTEREST_NAME':'INTEREST_NAME_ACTIVE'}).drop(['USER_ID'], axis=1)

    subset_df_binary = subset_df_binary.merge(user_interests[["USER_ID", "INTEREST_NAME"]],
                                             how = 'inner', left_on = 'PASSIVE_USER_ID',
                                             right_on = "USER_ID").rename(columns = {'INTEREST_NAME':'INTEREST_NAME_PASSIVE'}).drop(['USER_ID'], axis=1)


    subset_df_binary['simple_binary_similarity'] = subset_df_binary.apply(lambda x: simple_binary_similarity(x.INTEREST_NAME_ACTIVE, x.INTEREST_NAME_PASSIVE), axis=1)

    #let's transform them into torch tensors so we can evaluate
    user_id_1 = torch.tensor(subset_df_binary['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(subset_df_binary['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(subset_df_binary[target].to_list())
    predictions = torch.tensor(subset_df_binary['simple_binary_similarity'].to_list())

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    print("Simple Binary Similarity MAP Score is", sim_order_mAP)
    print("Simple Binary Similarity mean_NDCG score is", sim_order_mean_NDCG)
    
    
    #Binary Threshold Similarity
    subset_binary_threshold = test_df[["ACTIVE_USER_ID", "PASSIVE_USER_ID",target]].merge(user_interests[["USER_ID", "INTEREST_NAME", "interest_embedding"]], 
                                                                             how = 'inner', left_on = "ACTIVE_USER_ID",
                                                                             right_on = "USER_ID").rename(columns = {'INTEREST_NAME':'INTEREST_NAME_ACTIVE',
                                                                                                                    'interest_embedding': 'interest_embedding_active'}).drop(['USER_ID'], axis=1)

    subset_binary_threshold = subset_binary_threshold.merge(user_interests[["USER_ID", "INTEREST_NAME", "interest_embedding"]],
                                             how = 'inner', left_on = 'PASSIVE_USER_ID',
                                             right_on = "USER_ID").rename(columns = {'INTEREST_NAME':'INTEREST_NAME_PASSIVE',
                                                                                    'interest_embedding': 'interest_embedding_passive'}).drop(['USER_ID'], axis=1)


    # Calculate the cosine similarity between all pairs of interests and save the combinations of each interest pair
    subset_binary_threshold[['pair_cosine', 'interest_pairs']] = subset_binary_threshold.apply(lambda x: pd.Series(pairwise_cosine_similarity(x['interest_embedding_active'], x['interest_embedding_passive'], x['INTEREST_NAME_ACTIVE'], x['INTEREST_NAME_PASSIVE'])), axis=1)

    #calculate the binary threshold embedding similarity
    subset_binary_threshold['binary_threshold_embedding_similarity'] = subset_binary_threshold.apply(lambda x: binary_embedding_similarity(x.pair_cosine, x.interest_pairs), axis = 1)

    #let's transform them into torch tensors so we can evaluate
    user_id_1 = torch.tensor(subset_binary_threshold['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(subset_binary_threshold['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(subset_binary_threshold[target].to_list())
    predictions = torch.tensor(subset_binary_threshold['binary_threshold_embedding_similarity'].to_list())

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    print("Binary Threshold Similarity MAP Score is", sim_order_mAP)
    print("Binary Threshold Similarity mean_NDCG score is", sim_order_mean_NDCG)
    
    
    #Simple Cosine Similarity
    subset_df_binary['simple_cosine_similarity'] = subset_df_binary.apply(lambda x: simple_cosine_similarity(x.INTEREST_NAME_ACTIVE, x.INTEREST_NAME_ACTIVE), axis=1)

    #let's transform them into torch tensors so we can evaluate
    user_id_1 = torch.tensor(subset_df_binary['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(subset_df_binary['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(subset_df_binary[target].to_list())
    predictions = torch.tensor(subset_df_binary['simple_cosine_similarity'].to_list())

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    print("Simple Cosine Similarity MAP Score is", sim_order_mAP)
    print("Simple Cosine Similarity mean_NDCG score is", sim_order_mean_NDCG)
    
    
    
    #Simple Weighted Cosine
    user_1 = subset_df_binary[['ACTIVE_USER_ID', 'INTEREST_NAME_ACTIVE']].rename(columns = {'ACTIVE_USER_ID':'UID',
                                                                                    'INTEREST_NAME_ACTIVE': 'INTEREST_NAME'}).drop_duplicates(subset=['UID'])
    user_2 = subset_df_binary[['PASSIVE_USER_ID', 'INTEREST_NAME_PASSIVE']].rename(columns = {'PASSIVE_USER_ID':'UID',
                                                                                    'INTEREST_NAME_PASSIVE': 'INTEREST_NAME'}).drop_duplicates(subset=['UID'])


    user_all = pd.concat([user_1,user_2])
    
    #create a dictionary for user interest frequency
    interest_count = {}

    # Iterate through the DataFrame and count interests
    for interests_i in user_all['INTEREST_NAME']:
        for interest in interests_i:
            if interest in interest_count:
                interest_count[interest] += 1
            else:
                interest_count[interest] = 1

                
                
    subset_df_binary['simple_weighted_cosine_similarity'] = subset_df_binary.apply(lambda x: simple_weighted_cosine_similarity(x.INTEREST_NAME_ACTIVE, x.INTEREST_NAME_PASSIVE,
                                                                                                                               interest_count), axis=1)


    #let's transform them into torch tensors so we can evaluate
    user_id_1 = torch.tensor(subset_df_binary['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(subset_df_binary['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(subset_df_binary[target].to_list())
    predictions = torch.tensor(subset_df_binary['simple_weighted_cosine_similarity'].to_list())

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    print("Wighted Cosine Similarity MAP Score is", sim_order_mAP)
    print("WWeighted Cosine Similarity mean_NDCG score is", sim_order_mean_NDCG)
    
    
    
    #Maximum Similarity
    
    subset_binary_threshold['max_similarity'] = subset_binary_threshold.apply(lambda x: max_similarity(x.pair_cosine, x.interest_pairs),
                                                                         axis = 1)


    subset_binary_threshold['max_similarity'] = subset_binary_threshold['max_similarity'].fillna(0)


    #let's transform them into torch tensors so we can evaluate
    user_id_1 = torch.tensor(subset_binary_threshold['ACTIVE_USER_ID'].to_list())
    user_id_2 = torch.tensor(subset_binary_threshold['PASSIVE_USER_ID'].to_list())
    true_labels = torch.tensor(subset_binary_threshold[target].to_list())
    predictions = torch.tensor(subset_binary_threshold['max_similarity'].to_list())

    #evaluate the model
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_labels)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    sim_order_mAP, sim_order_mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    
    print("Maximum Similarity MAP Score is", sim_order_mAP)
    print("Maximum Cosine Similarity mean_NDCG score is", sim_order_mean_NDCG)

    
    
    

