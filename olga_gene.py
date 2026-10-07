
path ='/content/drive/MyDrive/data/'


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GAE, GATConv

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import networkx as nx

import pickle
import random
from pathlib import Path
import time
import argparse
import warnings
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
import os



def ImportTrainData(disease, year):
  if year:
    train_data = pd.read_csv(path+f'main_data/train_data/{year}/train/{disease}.csv')
    test_data = pd.read_csv(path+f'main_data/train_data/{year}/test/{disease}.csv')
  else:
    train_data = pd.read_csv(path+f'main_data/train_data/{disease}.csv')

  return train_data, test_data


def set_index(data, edges_list):
  data = data.reset_index().set_index('ensembl')
  genes = data.index.to_numpy()
  gene_id_dict = {ensembl: idx for idx, ensembl in enumerate(genes)}

  # map ID's in node dataset
  myID = data.index.map(gene_id_dict).rename('myID')
  data.insert(loc=0, column='myID', value=myID)
  data = data.reset_index().set_index('myID')
   # map edge list
  edges_list.iloc[:, 0] = edges_list.iloc[:, 0].map(gene_id_dict)
  edges_list.iloc[:, 1] = edges_list.iloc[:, 1].map(gene_id_dict)
  edges_list = edges_list.dropna()
  return(data, edges_list)


def read_data(disease, year):

    if year:
      node_data = pd.read_csv(path+f'main_data/{year}/ppi_{year}_700_emb.csv', sep=",")
      edges_list = pd.read_csv(path+f"main_data/{year}/edge_list_{year}_700.edg", header=None, sep='\t')
      disease_train_data, disease_test_data = ImportTrainData(disease, year)
    else:
      node_data = pd.read_csv(path+f'main_data/full/ppi_full_emb.csv', sep=",")
      edges_list = pd.read_csv(path+f"main_data/{year}/edge_list_full_700.edg", header=None, sep='\t')
      disease_data = ImportTrainData(disease, '')


    node_data = node_data.rename(columns={'string_id': 'ensembl'})

    node_data['test'] = 1
    node_data['label'] = 0

    node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1 #train disease gene
    node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1 #test disease gene
    node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0
    data = node_data

    if year== 'full_700':
      data, edges_list  = set_index(data, edges_list)

      # scale edge features appropriately (they take values in the range 0-1000)
      if len(edges_list.columns) > 2:
          edge_feat_cols = edges_list.columns[2:].to_numpy()
          edges_list[edge_feat_cols] /= 1000

    edges_list.columns = ['gene1', 'gene2']


    return (data, edges_list)

"""
### create temporal graph with real edges
"""


def connect_to_train(graph, train_nodes, train_feats, isolated_nodes, label):
 added_edges = 0
 # k = mean(degree) and 3<k<15<len(train_feats)
 degrees = [deg for _, deg in graph.degree()]
 mean_deg = int(np.mean(degrees))
 k = min(max(3, mean_deg), min(15, len(train_nodes)))
 print(f"Using k={k} for KNN")
 knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
 knn.fit(train_feats)
 for node in isolated_nodes:
  node_feat = graph.nodes[node]['features'].reshape(1, -1)
  dists, indices = knn.kneighbors(node_feat)
  for idx in indices[0]:
    neighbor = train_nodes[idx]
    if not graph.has_edge(node, neighbor):
      graph.add_edge(node, neighbor)
      added_edges += 1
 print(f"{label}: connected {len(isolated_nodes)} isolated nodes with {added_edges} edges.")

def connect_isolated_nodes_to_train(graph):

 existing_nodes = list(graph.nodes(data=True))


 train_nodes = [n for n, d in existing_nodes if d.get('train') == 1]
 val_nodes = [n for n, d in existing_nodes if d.get('val') == 1]
 test_nodes = [n for n, d in existing_nodes if d.get('test') == 1]


 train_feats = np.array([graph.nodes[n]['features'] for n in train_nodes])

 isolated_train = [n for n in train_nodes if graph.degree[n] == 0]
 isolated_val = [n for n in val_nodes if graph.degree[n] == 0]
 isolated_test = [n for n in test_nodes if graph.degree[n] == 0]

 if isolated_train:
  connect_to_train(graph, train_nodes, train_feats, isolated_train, "TRAIN")

 if isolated_val:
  connect_to_train(graph, train_nodes, train_feats, isolated_val, "VAL")

 if isolated_test:
  connect_to_train(graph, train_nodes, train_feats, isolated_test, "TEST")


 total_isolates_after = list(nx.isolates(graph))
 if total_isolates_after:
  print(f" Still {len(total_isolates_after)} isolated nodes remaining (check manually):")
  print(total_isolates_after)
 else:
  print(" All isolated nodes are now connected.")

def create_graph_temporal(disease, seed = 81):

  data, edges_list = read_data(disease, 2019)
  train_nodes = data[(data['label'] == 1) & (data['test'] == 0)]
  pos_test_nodes = data[(data['label'] == 1) & (data['test'] == 1)]
  neg_test_nodes = data[(data['label'] == 0) & (data['test'] == 1)]

  train_data, pos_val_data = train_test_split(train_nodes, test_size=0.1, random_state=seed)
  neg_val_data, neg_test_data = train_test_split(neg_test_nodes, test_size=0.5, random_state=seed)
  val_data = pd.concat([pos_val_data, neg_val_data])

  test_data = pd.concat([pos_test_nodes, neg_test_data])

  all_gene_ids = pd.concat([train_data, val_data, test_data])['ensembl'].unique()
  gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(all_gene_ids)}

  graph = nx.Graph()

  for idx, row in train_data.iterrows():
      gene_id = row['ensembl']
      node_idx = gene_id_to_idx[gene_id]
      features = row.drop(['ensembl', 'label', 'test']).values.astype(np.float32)
      graph.add_node(node_idx,
                      gene_id=gene_id,
                      features=features,
                      label=row['label'],
                      train=1,
                      val=0,
                      test=0)


  for idx, row in val_data.iterrows():
    gene_id = row['ensembl']
    node_idx = gene_id_to_idx[gene_id]
    features = row.drop(['ensembl', 'label','test']).values.astype(np.float32)
    if node_idx not in graph.nodes:
       graph.add_node(node_idx,
                      gene_id=gene_id,
                      features=features,
                      label=row['label'],
                      train=0,
                      val=1,
                      test=0)

  for idx, row in test_data.iterrows():
    gene_id = row['ensembl']
    node_idx = gene_id_to_idx[gene_id]
    features = row.drop(['ensembl', 'label', 'test']).values.astype(np.float32)
    if node_idx not in graph.nodes:
      graph.add_node(node_idx,
                    gene_id=row['ensembl'],
                    features=features,
                    label=row['label'],
                    train=0,
                    val=0,
                    test=1)


  for gene1, gene2 in edges_list.itertuples(index=False, name=None):

    if gene1 in gene_id_to_idx and gene2 in gene_id_to_idx:

        u = gene_id_to_idx[gene1]
        v = gene_id_to_idx[gene2]
        if u in graph.nodes and v in graph.nodes:
            graph.add_edge(u, v)


  #connect val_pos to KNN train nodes
  connect_isolated_nodes_to_train(graph)


  print(f"Total nodes: {graph.number_of_nodes()}")
  print(f"Total edges: {graph.number_of_edges()}")

  train_count = sum(nx.get_node_attributes(graph, 'train').values())
  val_count = sum(nx.get_node_attributes(graph, 'val').values())
  test_count = sum(nx.get_node_attributes(graph, 'test').values())
  print(f"Train nodes: {train_count}, Val nodes: {val_count}, Test nodes: {test_count}")
  isolated_nodes = list(nx.isolates(graph))
  print(f"Isolated nodes: {len(isolated_nodes)}")


  gpickle_path = f"{disease}_graph_700.gpickle"
  with open(gpickle_path, 'wb') as f:
      pickle.dump(graph, f)

file_name = pd.read_csv(path+"main_data/train_data/2019/disease_summary.csv")

for row in file_name['disease_id']:

  create_graph_temporal(row)



"""### create knn temporal graph"""

def create_knn_gragh_temporal(disease, year, k, seed = 81, metric='euclidean'):

  data, edges_list = read_data(disease, year)
  train_nodes = data[(data['label'] == 1) & (data['test'] == 0)]
  pos_test_nodes = data[(data['label'] == 1) & (data['test'] == 1)]
  neg_test_nodes = data[(data['label'] == 0) & (data['test'] == 1)]

  train_data, pos_val_data = train_test_split(train_nodes, test_size=0.1, random_state=seed)
  neg_val_data, neg_test_data = train_test_split(neg_test_nodes, test_size=0.5, random_state=seed)
  val_data = pd.concat([pos_val_data, neg_val_data])

  test_data = pd.concat([pos_test_nodes, neg_test_data])

  all_gene_ids = pd.concat([train_data, val_data, test_data])['ensembl'].unique()
  gene_id_to_idx = {gene_id: idx for idx, gene_id in enumerate(all_gene_ids)}

  graph = nx.Graph()

  for idx, row in train_data.iterrows():
      gene_id = row['ensembl']
      node_idx = gene_id_to_idx[gene_id]
      features = row.drop(['ensembl', 'label', 'test']).values.astype(np.float32)
      graph.add_node(node_idx,
                      gene_id=gene_id,
                      features=features,
                      label=row['label'],
                      train=1,
                      val=0,
                      test=0)


  for idx, row in val_data.iterrows():
    gene_id = row['ensembl']
    node_idx = gene_id_to_idx[gene_id]
    features = row.drop(['ensembl', 'label','test']).values.astype(np.float32)
    if node_idx not in graph.nodes:
       graph.add_node(node_idx,
                      gene_id=gene_id,
                      features=features,
                      label=row['label'],
                      train=0,
                      val=1,
                      test=0)

  for idx, row in test_data.iterrows():
    gene_id = row['ensembl']
    node_idx = gene_id_to_idx[gene_id]
    features = row.drop(['ensembl', 'label', 'test']).values.astype(np.float32)
    if node_idx not in graph.nodes:
      graph.add_node(node_idx,
                    gene_id=row['ensembl'],
                    features=features,
                    label=row['label'],
                    train=0,
                    val=0,
                    test=1)

  # Build edges using kNN
  node_ids = list(graph.nodes())
  features = np.stack([graph.nodes[n]['features'] for n in node_ids])

  knn = NearestNeighbors(n_neighbors=k + 1, metric=metric)  # +1 to remove self-loop
  knn.fit(features)
  distances, indices = knn.kneighbors(features)

  for i, neighbors in enumerate(indices):
      src = node_ids[i]
      for j in neighbors:
          dst = node_ids[j]
          if src != dst and not graph.has_edge(src, dst):
              graph.add_edge(src, dst)


  print(f"k={k}")
  print(f"Total nodes: {graph.number_of_nodes()}")
  print(f"Total edges: {graph.number_of_edges()}")

  train_count = sum(nx.get_node_attributes(graph, 'train').values())
  val_count = sum(nx.get_node_attributes(graph, 'val').values())
  test_count = sum(nx.get_node_attributes(graph, 'test').values())
  print(f"Train nodes: {train_count}, Val nodes: {val_count}, Test nodes: {test_count}")

  val_nodes = [n for n, d in graph.nodes(data=True) if d['val'] == 1]
  test_nodes = [n for n, d in graph.nodes(data=True) if d['test'] == 1]



  return graph

year = '2017'
file_name = pd.read_csv(path+"main_data/train_data/2017/disease_summary.csv")

for row in file_name['disease_id']:

  for k in range(1,4):
      print(f'Processing {row} with k={k}')

      g = create_knn_gragh_temporal(row, year, k, metric='euclidean')

      path1 = path + 'models/OLGA/olga_knn_2017/' + row + '/k=' + str(k) + '/'
      os.makedirs(path1, exist_ok=True)
      name = f'{row}.gpickle'

      with open(path1 + name, 'wb') as f_out:
          pickle.dump(g, f_out)

train_nodes_in_graph = [n for n, d in g.nodes(data=True) if d.get('train') == 1]
print("Train nodes:", train_nodes_in_graph)


def train_test_split_OCL(df, seed=81, folds=10):

  kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

  df_int = df[df.label == 1]
  df_nint = df[df.label == 0]

  l_index_int = []

  for train_index, test_index in kf.split(df_int):
    df_train = df_int.iloc[train_index]
    df_test = df_int.iloc[test_index]

    df_int_train, df_int_val = train_test_split(df_train, test_size=0.1, random_state=seed)

    l_index_int.append([df_int_train.index, df_int_val.index, df_test.index])

  l_index_nint = []
  for i in range(folds):
    df_nint_val, df_nint_test = train_test_split(df_nint, test_size=0.5, random_state=i)
    l_index_nint.append([df_nint_val.index, df_nint_test.index])

  return l_index_int, l_index_nint

def generate_graph(df, k=10, metric='euclidean'):

    features = df.drop(columns=['ensembl', 'label']).values.astype(np.float32)

    adjacency = kneighbors_graph(features, k, mode='connectivity', include_self=False, metric=metric)

    graph = nx.Graph(adjacency)

    for i, row in df.iterrows():
        graph.nodes[i]['gene_id'] = row['ensembl']
        graph.nodes[i]['features'] = row.drop(['ensembl', 'label']).values.astype(np.float32)
        graph.nodes[i]['label'] = row['label']

    return graph

def create_knn_gragh():
  train_data_done = ['C16', 'D57', 'E10', 'F01', 'G10', 'H40', 'I10', 'J45', 'K44', 'L20', 'M32', 'N17']
  train_data = []
  file_name = pd.read_csv(path+"/main_data/train_data/file_name.csv")
  for f in file_name['disease_code']:
    train_data.append(f.split("_")[1])
  for row in train_data :
      if row in train_data_done:
        continue
      for k in [1, 2, 3]:
          print(f'Processing {row} with k={k}')

          data, _ = read_data(row)
          g = generate_graph(data, k=k, metric='euclidean')


          l_index_int, l_index_nint = train_test_split_OCL(data, folds=10)

          for f in range(10):

              for i in g.nodes():
                  if i in l_index_int[f][0]:
                      g.nodes[i]['train'] = 1
                      g.nodes[i]['val'] = 0
                      g.nodes[i]['test'] = 0
                  elif i in l_index_int[f][1] or i in l_index_nint[f][0]:
                      g.nodes[i]['train'] = 0
                      g.nodes[i]['val'] = 1
                      g.nodes[i]['test'] = 0
                  elif i in l_index_int[f][2] or i in l_index_nint[f][1]:
                      g.nodes[i]['train'] = 0
                      g.nodes[i]['val'] = 0
                      g.nodes[i]['test'] = 1


              path1 = path + 'main_data/olga_gragh/' + row + '/k=' + str(k) + '/'
              os.makedirs(path1, exist_ok=True)
              name = f'fold={f}.gpickle'
              with open(path1 + name, 'wb') as f_out:
                  pickle.dump(g, f_out)



class OLGA(nn.Module):
    def __init__(self, input_len, hidden_lens):
        super(OLGA, self).__init__()
        self.layer1 = GATConv(input_len, hidden_lens[0])
        self.layer2 = GATConv(hidden_lens[0], hidden_lens[1])

    def forward(self, x, edge_index):
        h1 = nn.Tanh()(self.layer1(x, edge_index)) #Tanh
        h2 = nn.Tanh()(self.layer2(h1, edge_index))
        return h2


def init_metrics():
    metrics = {
        '1': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        '-1': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'macro avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'weighted avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'accuracy': [],
        'time': []
    }
    return metrics

def save_values(metrics, values):
    for key in metrics.keys():
      if key == 'accuracy' or key == 'time':
        metrics[key].append(values[key])
      else:
        for key2 in metrics[key].keys():
          metrics[key][key2].append(values[key][key2])

def write_results(metrics, file_name, line_parameters, path, auc, train_data, r5, r10, r30):

    if not Path(path + 'results/OLGA/' + file_name).is_file():
        file_ = open(path + 'results/OLGA/' + file_name, 'w')
        string = 'file_name;Parameters'

        #--------------------------------------
        string += ';AUC;R5;R10;R30'
        #-------------------------------------

        string += '\n'
        file_.write(string)
        file_.close()


    file_ = open(path + 'results/OLGA/' + file_name, 'a')
    string = train_data
    string += ';' + line_parameters

    string += ';' + str(np.mean(auc)) + ';' + str(np.mean(r5)) + ';' + str(np.mean(r10)) + ';' + str(np.mean(r30))

    string += '\n'
    file_.write(string)
    file_.close()



def recall_at_top_k(scores_int, scores_out, top_percent):

    all_scores = np.concatenate([scores_int, scores_out])
    k = int(len(all_scores) * top_percent)
    top_k_threshold = np.sort(all_scores)[k]
    tp = np.sum(scores_int <= top_k_threshold)
    recall = tp / len(scores_int)
    return recall

def get_results(scores_int, interest_node, scores_out, outlier_node, disease):
  # print(disease)
  scores_int_df = pd.DataFrame(scores_int)
  scores_int_df.columns = ['scores' for col in scores_int_df.columns]
  scores_int_df['label'] = 1

  interest_node_df = pd.DataFrame(interest_node)
  interest_node_df.columns = ['gene_id' for col in interest_node_df.columns]
  interest_score = pd.concat([interest_node_df, scores_int_df], axis=1)

  scores_out_df = pd.DataFrame(scores_out)
  scores_out_df.columns = ['scores' for col in scores_out_df.columns]
  scores_out_df['label'] = 0

  outlier_node_df = pd.DataFrame(outlier_node)
  outlier_node_df.columns = ['gene_id' for col in outlier_node_df.columns]
  outlier_score = pd.concat([outlier_node_df, scores_out_df], axis=1)

  trial_score = pd.concat([interest_score, outlier_score], axis=0, ignore_index=True)
  # trial_score = trial_score.reset_index()
  results_df = pd.DataFrame({
        'gene_id': trial_score['gene_id'],
        'y_true': trial_score['label'],
        'score' : trial_score['scores']

         })
  results_df.to_csv(path+"/models/OLGA/scores/"+disease+"_scores_test2017.csv", index=False)



  trial_score_sorted = trial_score.sort_values(by=['scores'], ascending=True)  # , ascending=False
  trial_score_sorted.reset_index(inplace=True, drop=True)

  positive_data = trial_score_sorted.loc[trial_score_sorted['label'] == 1]
  negative_data = trial_score_sorted.loc[trial_score_sorted['label'] == 0]

  avg_rank = pd.DataFrame(positive_data.index).mean()
  AUC = 1 - (avg_rank[0] / trial_score_sorted.shape[0])

  recall_5 = recall_at_top_k(scores_int, scores_out, 0.05)
  recall_10 = recall_at_top_k(scores_int, scores_out, 0.10)
  recall_30 = recall_at_top_k(scores_int, scores_out, 0.30)


  return positive_data, negative_data, AUC, recall_5, recall_10, recall_30


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_emb = None
        self.best_radius = None
        self.best_center = None
        self.best_epoch = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, score, cur_loss, epoch, radius, center, embs):
        if (self.best_score is None) or (self.lowest_loss is None):
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.best_emb = embs
            self.best_radius = radius
            self.best_center = center
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.best_emb = embs
            self.best_radius = radius
            self.best_center = center

            self.counter = 0

        return self.early_stop, self.best_emb, self.best_radius, self.best_center, self.best_epoch

def One_Class_GNN_prediction(disease, center, radius, learned_representations, G, val_test, dic):
    # print(val_test)
    with torch.no_grad():
        count = 0
        for node in G.nodes:
            G.nodes[node]['embedding_aocgnn'] = learned_representations[count].cpu().numpy()
            count+=1

        interest = []
        outlier = []
        interest_node = []
        outlier_node = []
        for node in G.nodes:
            if G.nodes[node][val_test] == 1 and G.nodes[node]['label'] == 1:
                interest.append(G.nodes[node]['embedding_aocgnn'])
                interest_node.append(G.nodes[node]['gene_id'])
            elif G.nodes[node][val_test] == 1 and G.nodes[node]['label'] == 0:
                outlier.append(G.nodes[node]['embedding_aocgnn'])
                outlier_node.append(G.nodes[node]['gene_id'])

        dist_int = np.sum((interest - center.cpu().numpy()) ** 2, axis=1)

        scores_int = dist_int - radius.cpu().numpy() ** 2

        dist_out = np.sum((outlier - center.cpu().numpy()) ** 2, axis=1)

        scores_out = dist_out - radius.cpu().numpy() ** 2

        preds_interest = [1 if score < 0 else -1 for score in scores_int]
        preds_outliers = [-1 if score > 0 else 1 for score in scores_out]

        y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
        y_pred = list(preds_interest) + list(preds_outliers)
        #-----------------------------
        y_scores = list(scores_int) + list(scores_out)
        #---------------------------------Rank--------------------------------------

        positive_data, negative_data, AUC, recall_5, recall_10, recall_30 = get_results(scores_int, interest_node, scores_out, outlier_node, disease)

        if dic:
            return classification_report(y_true, y_pred, output_dict=dic), positive_data, negative_data, AUC, recall_5, recall_10, recall_30
        else:
            return print(AUC)

def anomaly_score(center, radius, learned_representations, mask):

    l_r_mask = torch.BoolTensor(mask)

    dist = torch.sum((learned_representations[l_r_mask] - center) ** 2, dim=1)

    scores = dist - radius ** 2

    return scores

def one_class_loss(center, radius, learned_representations, mask):

    scores = anomaly_score(center, radius, learned_representations, mask)

    loss = torch.mean(torch.where(scores > 0, scores + 1, torch.exp(scores)))

    return loss

def one_class_masking(G, train_val):

    train_mask = np.zeros(len(G.nodes), dtype='bool')
    unsup_mask = np.zeros(len(G.nodes), dtype='bool')

    normal_train_idx = []
    unsup_idx = []
    count = 0
    for node in G.nodes:
        if train_val:
            if G.nodes[node]['train'] == 1 or (G.nodes[node]['val'] == 1 and G.nodes[node]['label'] == 1):
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
            count += 1
        else:
            if G.nodes[node]['train'] == 1:
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
            count += 1


    train_mask[normal_train_idx] = 1
    unsup_mask[unsup_idx] = 1


    return train_mask, normal_train_idx, unsup_mask, unsup_idx



def train_olga(g, hidden1, hidden2, patience, lr, r, multi_task, epochs, disease):
    loss_ocl = 0
    recon_loss_unsup = 0
    embeddings, losses_ocl, losses_rec, accuracies, losses = [], [], [], [], []
    best_embeddings, best_radius, best_center = [], 0, []
    c = [0] * hidden2
    center = torch.Tensor(c)
    radius = torch.Tensor([r])

    mask, t_mask, mask_unsup, t_mask_unsup = one_class_masking(g, False)
    G = from_networkx(g)

    g_unsup = g.subgraph(t_mask_unsup)
    G_unsup = from_networkx(g_unsup)

    model_ocl = OLGA(len(G.features[0]), [hidden1, hidden2])
    model = GAE(model_ocl)
    stopper = EarlyStopping(patience)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs+1):
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        learned_representations = model.encode(G.features.float(), G.edge_index)


        if epoch < multi_task:
            loss = model.recon_loss(learned_representations, G.edge_index)
        else:
            loss_ocl = one_class_loss(center, radius, learned_representations, mask)

            recon_loss_unsup = model.recon_loss(learned_representations[mask_unsup], G_unsup.edge_index)

            loss = loss_ocl + recon_loss_unsup



        rep, positive_data, negative_data, AUC, r5, r10, r30 = One_Class_GNN_prediction(disease, center, radius, learned_representations, g, 'val', True)
        f1 = rep['macro avg']['f1-score']

        # Compute gradients
        loss.backward()

        # Tune parameters
        optimizer.step()

        stop, best_embeddings, best_radius, best_center, best_epoch = stopper.step(f1, loss, epoch, radius, center,
                                                                                  learned_representations)


        embeddings.append(learned_representations)
        losses_ocl.append(loss_ocl)
        losses_rec.append(recon_loss_unsup)
        losses.append(loss)
        accuracies.append(f1)

        if stop:
            break

        torch.cuda.empty_cache()


    dic_results, positive_data, negative_data, AUC, r5, r10, r30 = One_Class_GNN_prediction(disease, best_center, best_radius, best_embeddings, g, 'val', True)
    print(f'AUC={AUC}')

    return dic_results, positive_data, negative_data, AUC, r5, r10, r30

def train_parameters(l_graphs, patience, hidden1, hidden2, r, lr, file_name, pr, epochs, disease):
    multi_task = patience / 2
    l_param = str(hidden1) + '_' + str(hidden2) + '_' + str(r) + '_' + str(lr) + '_' + str(patience)
    metrics = init_metrics()
    auc = []
    g = l_graphs

    print(g)
    start = time.time()
    values, positive_data, negative_data, AUC, r5, r10, r30 = train_olga(g, hidden1, hidden2, patience, lr, r, multi_task,epochs, disease)
    end = time.time()
    time_ = end - start
    values['time'] = time_
    save_values(metrics, values)


    write_results(metrics, file_name, l_param, pr, AUC, disease, r5, r10, r30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OLGA')

    parser.add_argument("--k", type=str, default='k=1', help="k from graph modeling")

    parser.add_argument("--h1", type=int, default=48, help="neurons from first hidden layer")

    parser.add_argument("--h2", type=int, default=2, help="neurons from second hidden layer")

    parser.add_argument("--radius", type=float, default=0.5, help="hypershpere radius")

    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")

    parser.add_argument("--patience", type=int, default=300, help="patience for early stopping")

    parser.add_argument("--n-epochs", type=int, default=5000, help="training epochs")

    parser.add_argument("--dataset", type=str, default="fakenews", help="dataset") #food

    args_list = [
        "--k", "k=1",
        "--h1", "48",
        "--h2", "2",
        "--radius", "0.5",
        "--lr", "0.0001",
        "--patience", "300",
        "--n-epochs", "1000",
        "--dataset", "fakenews"
    ]
    args = parser.parse_args(args_list)

    # seeds
    seed = 81
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    disease = pd.read_csv(path+"main_data/train_data/2017/disease_summary.csv")
    result_file = 'results_knn20172.csv'
    l_graphs = []



    for row in disease['disease_id']:

      for k in range(1,4):
        path1 = path + f'models/OLGA/olga_knn_2017/{row}/k={k}/{row}.gpickle'
        with open(path1, 'rb') as f:
          train_graph = pickle.load(f)



        print(row)
        train_parameters(train_graph, args.patience, args.h1, args.h2, args.radius, args.lr, result_file, path, args.n_epochs, row)





learn_radius = False
learn_center = False
results = []

disease = pd.read_csv(path+"main_data/train_data/2019/disease_summary.csv")


seed = 81
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)


for row in disease['disease_id']:

  file = path + f'models/OLGA/olga_knn/{row}/k=2/{row}.gpickle'


  with open(file, 'rb') as f:
    g = pickle.load(f)

  mask, t_mask, mask_unsup, t_mask_unsup = one_class_masking(g, True)

  G = from_networkx(g)


  hidden1 = 48

  hidden2 = 2

  c = [0] * hidden2
  r = [0.35] #0.35
  c = torch.Tensor(c)
  r = torch.Tensor(r)

  model_ocl = OLGA(len(G.features[0]), [hidden1, hidden2])
  model = GAE(model_ocl)

  optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

  patience = 300

  stopper = EarlyStopping(patience)

  centers = []
  embeddings = []
  losses_ocl = []
  losses_rec = []
  accuracies = []
  radiuss = []
  losses = []


  best_embeddings, best_radius, best_center = [], 0, []
  # Training loop

  g_unsup = g.subgraph(t_mask_unsup)
  G_unsup = from_networkx(g_unsup)

  loss_ocl = 0
  recon_loss_unsup = 0

  radius = r
  center = c

  for epoch in range(1000):
      # Clear gradients
      optimizer.zero_grad()

      # Forward pass

      learned_representations = model.encode(G.features.float(), G.edge_index)

      if epoch < patience / 2:
          loss = model.recon_loss(learned_representations, G.edge_index)
      else:
          loss_ocl = one_class_loss(center, radius, learned_representations, mask)

          recon_loss_unsup = model.recon_loss(learned_representations[mask_unsup], G_unsup.edge_index)

          loss = loss_ocl + recon_loss_unsup

      rep, positive_data, negative_data, AUC, r5, r10, r30 = One_Class_GNN_prediction(row, center, radius, learned_representations, g, 'val', True)
      f1 = rep['macro avg']['f1-score']

      # Compute gradients
      loss.backward()

      # Tune parameters
      optimizer.step()

      print(f'Epoch {epoch:>3} | Loss: {loss:.5f} | F1: {f1*100:.2f}% | Loss_R: {recon_loss_unsup:.5f} |  Loss_O: '
            f''f'{loss_ocl:.5f} | R: {torch.abs(torch.mean(radius)):.2f}')

      stop, best_embeddings, best_radius, best_center, best_epoch = stopper.step(AUC, loss, epoch, radius, center,
                                                                                learned_representations)

      embeddings.append(learned_representations)
      losses_ocl.append(loss_ocl)
      losses_rec.append(recon_loss_unsup)
      losses.append(loss)
      accuracies.append(f1)
      centers.append(center)
      radiuss.append(radius)

      if stop:
          break

  rep, positive_data, negative_data, AUC, r5, r10, r30 = One_Class_GNN_prediction(row, best_center, best_radius, best_embeddings, g, 'test', True)
  f1 = rep['macro avg']['f1-score']
  print(f'row={row}')
  print(f'AUC={AUC}')
  print(r5)
  print(r10)
  print(r30)
  results.append({
      'Dataset': row,
      'AUC': AUC,
      'r5': r5,
      'r10': r10,
      'r30': r30,

    })


results_df = pd.DataFrame(results)
results_df.to_csv(path+f"/results/OLGA/results_2nn_test.csv", index=False)


