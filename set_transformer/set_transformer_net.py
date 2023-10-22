import torch
import torch.nn as nn
from modules import SAB, PMA
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
import sys


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
    A function that calculates the cosine similarity scores between corresponding user embeddings.

    Inputs:
        user_id_1 (list of torch.Tensor): A list of user IDs for the first set of users.
        user_id_2 (list of torch.Tensor): A list of user IDs for the second set of users. Both lists should have the same length.

    Output:
        final_scores (torch.Tensor): A tensor containing the cosine similarity scores between corresponding user embeddings.

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


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(SiameseNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)

    def forward_once(self, x):
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x1, x2):
        
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
    
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(target * torch.pow(euclidean_distance, 2) +
                                      (1 - target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
    
class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=300, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=300),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)
    
    

def get_embeddings(user_id_1, user_id_2, model):
    """
    Retrieve and process embeddings for two sets of users using a specified model.

    Inputs:
        user_id_1 (list of torch.Tensor): A list of user IDs for the first set of users.
        user_id_2 (list of torch.Tensor): A list of user IDs for the second set of users. Both lists should have the same length.
        model (torch.nn.Module): A PyTorch model used for processing the user embeddings.

    Output:
        output_1 (torch.Tensor): Processed embeddings for the first set of users.
        output_2 (torch.Tensor): Processed embeddings for the second set of users.
    """
    
    embeddings_user_1 = []
    embeddings_user_2 = []

    
    for i in range(len(user_id_1)):
        
        #get the set of embeddings for each user
        user_1_embd = user_embedding_dict_2[user_id_1[i].item()]
        user_2_embd = user_embedding_dict_2[user_id_2[i].item()]
        
        #convert the arrays to torch tensor and stack the interests for each user
        tensor_list_1 = [torch.tensor(arr) for arr in user_1_embd]
        stacked_tensor_1 = torch.stack(tensor_list_1)
        
        tensor_list_2 = [torch.tensor(arr) for arr in user_2_embd]
        stacked_tensor_2 = torch.stack(tensor_list_2)
        
        #append the interests embedding for each user
        embeddings_user_1.append(stacked_tensor_1)
        embeddings_user_2.append(stacked_tensor_2)
        
    
    #pad the tensors to the max dimension
    padded_tensors_1 = pad_tensors(embeddings_user_1)
    final_stacked_1 = torch.stack(padded_tensors_1)
    
    #print("final stacked dimensions", final_stacked_1.shape)
    
    padded_tensors_2 = pad_tensors(embeddings_user_2)
    final_stacked_2 = torch.stack(padded_tensors_2)
    
    final_stacked_1, final_stacked_2 = final_stacked_1.to(torch.float32), final_stacked_2.to(torch.float32)
    #pass them throug the model
    output_1 = model(final_stacked_1.to(device))
    output_2 = model(final_stacked_2.to(device))
    
    #print("output is", output_1)
    #print("reshape is", output_1.reshape(output_1.size(0), output_1.size(2)))
    
    output_1 = output_1.reshape(output_1.size(0), output_1.size(2))
    output_2 = output_2.reshape(output_2.size(0), output_2.size(2))
    
    
    return output_1, output_2 


def pad_tensors(tensor_list):
    
    
    padded_tensors = []
    
    max_size = max(tensor.size(0) for tensor in tensor_list)
    
    # Pad each tensor to the maximum size and store in the padded_tensors list
    for tensor in tensor_list:
        pad_size = max_size - tensor.size(0)
        padded_tensor = torch.nn.functional.pad(tensor, pad=(0, 0, 0, pad_size))  # pad along dimension 0
        padded_tensors.append(padded_tensor)

    return padded_tensors


def train_model(train_loader, train_df, current_model):
    """
    Train a Siamese network model using the provided data.

    Inputs:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        train_df (pandas.DataFrame): DataFrame containing training data.
        current_model (torch.nn.Module): Current Siamese network model to be trained.

    Output:
        current_model (torch.nn.Module): The trained Siamese network model.
        avg_loss (float): Average loss over the training data.
        user_1 (torch.Tensor): User IDs for the first set of users.
        user_2 (torch.Tensor): User IDs for the second set of users.
        true_labels (torch.Tensor): True binary match labels.
        predictions (torch.Tensor): Predicted cosine similarity scores.

    """
    
    siamese_net.train()
    total_loss = 0.0

    true_labels = []
    user_1 = []
    user_2 = []
    predictions = []
    example_before = []

    for i, (user_id_1, user_id_2, binary_match) in tqdm.tqdm(enumerate(train_loader), total=len(train_df)//batch_size+1):

            #get the user_1 and user_2 embeddings
            inputs_1, inputs_2 = get_embeddings(user_id_1, user_id_2, set_transformers_net)

            inputs_1, inputs_2, binary_match = inputs_1.to(device), inputs_2.to(device), binary_match.to(device)
            inputs_1, inputs_2 = inputs_1.to(torch.float32), inputs_2.to(torch.float32)

            optimizer.zero_grad()
            outputs1, outputs2 = current_model(inputs_1, inputs_2)

            #get the predictions
            cosine_similarity_score = F.cosine_similarity(outputs1, outputs2)
            predictions.append(cosine_similarity_score)

            loss = criterion(outputs1, outputs2, binary_match)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


            true_labels.append(binary_match)
            user_1.append(user_id_1)
            user_2.append(user_id_2)


    avg_loss = total_loss / len(train_loader)

    user_1 = torch.cat(user_1, dim=0)
    user_2 = torch.cat(user_2, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    predictions = torch.cat(predictions, dim=0)
    
    return current_model, avg_loss, user_1, user_2, true_labels, predictions


def get_predictions(siamese_net, data_loader, set_transformers_net):
    """
    Generate predictions using a Siamese network on a given data loader.

    Inputs:
        siamese_net (torch.nn.Module): Siamese network model used for generating predictions.
        data_loader (torch.utils.data.DataLoader): DataLoader for the data on which predictions are to be made.
        set_transformers_net (torch.nn.Module): Model for obtaining user embeddings.

    Output:
        user_id_1 (torch.Tensor): User IDs for the first set of users.
        user_id_2 (torch.Tensor): User IDs for the second set of users.
        true_relevance (torch.Tensor): True binary match labels.
        predictions (torch.Tensor): Predicted cosine similarity scores.
        
    """
    
    siamese_net.eval()
    predictions = []
    user_id_1 = []
    user_id_2 = []
    true_relevance = []
    total_loss = 0.0
    
    with torch.no_grad():
        
        for inputs1, inputs2, targets in data_loader:
            
            #get the user_1 and user_2 embeddings
            inputs_1, inputs_2 = get_embeddings(inputs1, inputs2, set_transformers_net)
            
            inputs_1, inputs_2, targets = inputs_1.to(device), inputs_2.to(device), targets.to(device)
            inputs_1, inputs_2 = inputs_1.to(torch.float32), inputs_2.to(torch.float32)
            
            outputs1, outputs2 = siamese_net(inputs_1, inputs_2)
            
            cosine_similarity_score = F.cosine_similarity(outputs1, outputs2)
            predictions.append(cosine_similarity_score)
            
            loss = criterion(outputs1, outputs2, targets)
            total_loss += loss.item()
            
            user_id_1.append(inputs1)
            user_id_2.append(inputs2)
            true_relevance.append(targets)
            
    
    user_id_1 = torch.cat(user_id_1, dim=0)
    user_id_2 = torch.cat(user_id_2, dim=0)
    true_relevance = torch.cat(true_relevance, dim=0)
    predictions = torch.cat(predictions, dim=0)
    
    avg_loss = total_loss / len(data_loader)
    
#     logger.current_logger().report_scalar(
#     "loss_epoch", "validation loss", iteration=epoch, value=avg_loss
#     )
            
    return user_id_1, user_id_2, true_relevance, predictions





if __name__ == "__main__":
    
    #dating or bff
    mode = sys.argv[1]
    #one_way or matches 
    type_var = sys.argv[2]
    
    
    #set the right file path
    current_directory = os.getcwd()
    grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
    
    #load the matches
    file_path_matches = grandparent_directory + '/bff_dating_dataset/' + 'data/' + 'filtered_df_' + type_var + '_' + mode
    filtered_df_matches = pd.read_feather(file_path_matches)

    #load the interests
    file_path_interests = grandparent_directory + '/bff_dating_dataset/' + 'data/' + 'interests_table_' + mode
    interests = pd.read_feather(file_path_interests)

    #load the user-level interests
    file_path_user_level_interests = grandparent_directory + '/bff_dating_dataset/' + 'data/' + 'user_level_interest_' + mode
    user_interests = pd.read_feather(file_path_user_level_interests)
    
    
    #load the interests table and initiate the hash maps for easier access of embeddings
    user_interests.set_index('USER_ID', inplace=True)
    user_embedding_dict = user_interests['avg_interest_embeddings'].to_dict()
    user_embedding_dict_2 = user_interests['interest_embedding'].to_dict()
    
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
    
    
    #define the embedding dimension
    embedding_dim = 300
    hidden_dim = 16
    margin = 0.5
    num_epochs = 30
    learning_rate = 0.001
    
    
    siamese_net = SiameseNetwork(embedding_dim, hidden_dim)
    criterion = ContrastiveLoss(margin)
    optimizer = optim.Adam(siamese_net.parameters(), lr=learning_rate)
    set_transformers_net = SmallSetTransformer()
    
    
    #set the device
    if torch.cuda.is_available():

        siamese_net.cuda()
        criterion.cuda()
        set_transformers_net.cuda()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siamese_net.to(device)
    set_transformers_net.to(device)
    
    
    for epoch in range(num_epochs):
    
        #train the model
        siamese_net, avg_loss, user_1, user_2, true_labels, predictions = train_model(train_loader, train_df, siamese_net)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

        #evaluate the train set
        grouped_predictions, grouped_relevance_scores = group_by_query(user_1, predictions, true_labels)
        sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_2)

        mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)

        print("Traing mAP is", mAP)
        print("Traing NDCG score is", mean_NDCG)

        #make a prediction
        user_id_1, user_id_2, true_relevance, predictions = get_predictions(siamese_net, val_loader, set_transformers_net)

        #evaluate the model 
        grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_relevance)
        sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

        #Get the evaluation metrics scores
        mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)


        print("Validation mAP score is", mAP)
        print("Validation mean NDCG score is", mean_NDCG)
        
        
    
    model_path = 'siamese_bff_set_matches_cosine.pth'
    torch.save(siamese_net.state_dict(),model_path)

    #test the model
    loaded_model = SiameseNetwork(embedding_dim, hidden_dim)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    
    #make a prediction
    user_id_1, user_id_2, true_relevance, predictions = get_predictions(loaded_model, val_loader, set_transformers_net)

    #evaluate the model 
    grouped_predictions, grouped_relevance_scores = group_by_query(user_id_1, predictions, true_relevance)
    sorted_document_ids, sorted_scores, true_relevance = sort_documents_by_score(grouped_predictions, grouped_relevance_scores, user_id_2)

    #Get the evaluation metrics scores
    mAP, mean_NDCG = evaluate_model(sorted_document_ids, true_relevance, sorted_scores)
    
    
    print("The evaluation mAP is", mAP)
    print("The evaluation mean_NDCG is", mean_NDCG)
