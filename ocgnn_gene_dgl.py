import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve,precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import dgl
from dgl import random as dr
from dgl.data import load_data, tu, register_data_args, CoraGraphDataset, DGLDataset
from dgl import DGLGraph
from dgl import DGLGraph, transforms
#from dgl import transforms as T

from dgl.nn.pytorch import GraphConv, edge_softmax, GATConv
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv, SAGEConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

# import argparse
import logging
import time
import os
import random
import argparse

os.environ["DGLBACKEND"] = "pytorch"
# os.environ["DGL_DISABLE_SPARSE"] = "1"
path ='/content/drive/MyDrive/data/'

def ImportTrainData(disease, year):
  if year:
    train_data = pd.read_csv(path+f'/main_data/train_data/{year}/train_2017/{disease}.csv')
    test_data = pd.read_csv(path+f'/main_data/train_data/{year}/test_2017/{disease}.csv')
  else:
    train_data = pd.read_csv(path+f'/main_data/train_data/{disease}.csv')

  return train_data, test_data


def prepare_data(data, edge_list):

    data = data.reset_index().set_index('ensembl')
    genes = data.index.to_numpy()
    gene_id_dict = {gene_id: idx for idx, gene_id in enumerate(genes)}
    # map ID's in node dataset
    myID = data.index.map(gene_id_dict).rename('myID')
    data.insert(loc=0, column='myID', value=myID)
    data = data.reset_index().set_index('myID')


    # map edge list
    edge_list = edge_list.copy(deep=True)
    edge_list.iloc[:, 0] = edge_list.iloc[:, 0].map(gene_id_dict)
    edge_list.iloc[:, 1] = edge_list.iloc[:, 1].map(gene_id_dict)
    edge_list = edge_list.dropna()
    # scale edge features appropriately (they take values in the range 0-1000)
    if len(edge_list.columns) > 2:
        edge_feat_cols = edge_list.columns[2:].to_numpy()
        edge_list[edge_feat_cols] /= 1000

    return data, edge_list

def read_data(disease, node_data, edges_list, year):

   disease_train_data, disease_test_data = ImportTrainData(disease, year)

   node_data['test'] = 1
   node_data['label'] = 0


   node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1 #train disease gene
   node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1 #test disease gene
   node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0

   data = node_data

   assert(data.index.duplicated().sum() == 0)

   labels = data['label'].to_numpy().astype('int32')

   node_dataset, edge = prepare_data(data, edges_list)
   return node_dataset, edge, labels


def get_train_val_test_masks_time(data, n_nodes, label_col, test_size=0.2, val_size=0.1, seed=81):

   train_nodes = data[(data['label'] == 1) & (data['test'] == 0)]
   pos_test_nodes = data[(data['label'] == 1) & (data['test'] == 1)]
   neg_test_nodes = data[(data['label'] == 0) & (data['test'] == 1)]

   train_data, pos_val_data = train_test_split(train_nodes, test_size=0.1, random_state=seed)
   neg_val_data, neg_test_data = train_test_split(neg_test_nodes, test_size=0.5, random_state=seed)

   val_data = pd.concat([pos_val_data, neg_val_data])

   test_data = pd.concat([pos_test_nodes, neg_test_data])

   myIDs_train = train_data.index.to_numpy()
   myIDs_val = val_data.index.to_numpy()
   myIDs_test = test_data.index.to_numpy()


  # NOTE: train-val-test split is shuffled and stratified

  # create masks
   train_mask = np.zeros(n_nodes, dtype=bool)
   train_mask[myIDs_train] = True
   train_mask = torch.Tensor(train_mask).type(torch.bool)

   val_mask = np.zeros(n_nodes, dtype=bool)
   val_mask[myIDs_val] = True
   val_mask = torch.Tensor(val_mask).type(torch.bool)

   test_mask = np.zeros(n_nodes, dtype=bool)
   test_mask[myIDs_test] = True
   test_mask = torch.Tensor(test_mask).type(torch.bool)

   return train_mask, val_mask, test_mask


class GeneDataset(DGLDataset):
    def __init__(self, dataset,node, edges,year):
        self.dataset = dataset
        self.year = year
        self.node = node
        self.edges = edges
        super().__init__(name="gene")

    def process(self):
        node_dataset, edge_list, labels = read_data(self.dataset,self.node,
                                                    self.edges,self.year)
        node_feat_cols = node_dataset.columns[2:-2]
        nodes_data = node_dataset
        edges_data = edge_list.astype(np.int64)


        node_features = torch.from_numpy(nodes_data[node_feat_cols].to_numpy())
        node_labels = torch.from_numpy(
            nodes_data["label"].astype("category").cat.codes.to_numpy()
        )

        edges_src = torch.from_numpy(edges_data[0].to_numpy())
        edges_dst = torch.from_numpy(edges_data[1].to_numpy())

        self.graph = dgl.graph(
            (edges_src, edges_dst), num_nodes=nodes_data.shape[0]
        )

        self.graph.ndata["feat"] = node_features
        self.graph.ndata["label"] = node_labels


        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = nodes_data.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.graph.ndata["train_mask"] = train_mask
        self.graph.ndata["val_mask"] = val_mask
        self.graph.ndata["test_mask"] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1



def data_loader(args, node_data, edges_list,year,test_size=0.2, val_size=0.1):
    """creates training-ready data in pytorch_geometric format.

    Args:
        node_dataset (DataFrame): node dataset.
        edge_list (DataFrame): edge list dataframe.
        labels (DataFrame): labels dataframe with trials of labels.
        label_col (str): name of column in labels dataframe.
        test_size (float, optional): test split size. Defaults to 0.2.
        val_size (float, optional): val split size. Defaults to 0.1.

    Returns:
        torch_geometric.data.Data: Data object with node features, edge list,
            edge attributes, and train-val-test masks.
    """
    label_col = 'label'
    node_dataset, edge_list, labels = read_data(args.dataset,node_data, edges_list,year)

    # restrict to data with labels
    node_data_labeled = node_dataset[node_dataset[label_col].notna()]
    n_nodes = len(node_dataset) # total number of nodes
    train_mask, val_mask, test_mask = get_train_val_test_masks_time(node_data_labeled, n_nodes, label_col, test_size=test_size, val_size=val_size)


    dataset = GeneDataset(args.dataset,node_data, edges_list, year)
    data = dataset[0]

    features = torch.FloatTensor(data.ndata['feat'].to(torch.float))
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)
    in_feats = features.shape[1]
    n_edges = data.num_edges()


    n_classes = 2

    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
              train_mask.sum().item(),
              val_mask.sum().item(),
              test_mask.sum().item()))


    if args.gpu < 0:
      cuda = False
    else:
      cuda = True
      torch.cuda.set_device(args.gpu)
      features = features.cuda()
      labels = labels.cuda()
      train_mask = train_mask.cuda()
      val_mask = val_mask.cuda()
      test_mask = test_mask.cuda()


    g = data
    # add self loop
    if args.self_loop:
        # g.remove_edges_from(nx.selfloop_edges(g))
        # g=transform(g)
        g = dgl.add_self_loop(g)
        # if args.module!='GraphSAGE':
        #   g.add_edges(zip(g.nodes(), g.nodes()))


    n_edges = g.number_of_edges()
    if args.norm:

        # normalization
        degs = g.in_degrees().float()
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 0
        if cuda:
            norm = norm.cuda()
        g.ndata['norm'] = norm.unsqueeze(1)

    datadict={'g':g,'features':features,'labels':labels,'train_mask':train_mask,
        'val_mask':val_mask,'test_mask': test_mask,'input_dim':in_feats,'n_classes':n_classes,'n_edges':n_edges, 'data': node_data_labeled}


    return datadict


class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g,inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, bias=False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden,  bias=False, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes,bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class GraphSAGE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type
                #  num_nodes=None
                 ):
        super(GraphSAGE, self).__init__()


        self.layers = nn.ModuleList()
        self.g = g

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, feat_drop=dropout, bias=True,activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, feat_drop=dropout, bias=True,activation=activation))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, feat_drop=dropout, bias=True, activation=None)) # activation None


    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
        return h

def init_model(args,input_dim, g):
    # create GCN model
    num_nodes = g.num_nodes()
    if args.module== 'GCN':
        model = GCN(None,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.module== 'GraphSAGE':
        model = GraphSAGE(None,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                args.n_layers,
                F.relu,
                args.dropout,
                aggregator_type='pool'
                # num_nodes=num_nodes
        ) #mean,pool,lstm,gcn There are big problems with using pool to do multi-graph learning.
    if args.module== 'GAT':
        model = GAT(None,
                args.n_layers,
                input_dim,
                args.n_hidden*2,
                args.n_hidden,
                heads=([8] * args.n_layers) + [1],
                activation=F.relu,
                feat_drop=args.dropout,
                attn_drop=args.dropout,
                negative_slope=0.2,
                residual=False)


    if args.gpu < 0:
        cuda = False
    else:
        cuda = True

    if cuda:
        model.cuda()

    print(f'Parameter number of {args.module} Net is: {count_parameters(model)}')

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def loss_anomaly_score(data_center,outputs,radius=0,mask= None):
    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)


    scores = dist - radius ** 2
    return dist,scores

def loss_function(nu,data_center,outputs,radius=0,mask=None):
    dist,scores=loss_anomaly_score(data_center,outputs,radius,mask)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def init_center(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=device)

    model.eval()

    with torch.no_grad():

        outputs= model(input_g,input_feat)

        # get the inputs of the batch

        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, acc,loss, model,epoch,path):
        score = acc
        cur_loss=loss
        if (self.best_score is None) or (self.lowest_loss is None):
        #if self.lowest_loss is None:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.save_checkpoint(acc,loss,model,path)
        #elif cur_loss > self.lowest_loss:
        elif (score < self.best_score) and (cur_loss > self.lowest_loss):
            self.counter += 1
            if self.counter >= 0.8*(self.patience):
                print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.save_checkpoint(acc,loss,model,path)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, acc,loss,model,path):
        '''Saves model when validation loss decrease.'''
        print('model saved. loss={:.4f} AUC={:.4f}'. format(loss,acc))
        torch.save(model.state_dict(), path)



GAE_mode='AX'

def ae_train(args,logger,data,model,path):

    checkpoints_path=path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    # initialize data center

    adj=data['g'].adjacency_matrix().to_dense().cuda()
    loss_fn = nn.MSELoss()


    dur = []
    model.train()

    #Create a matrix to store the resulting curves
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)

    for epoch in range(args.n_epochs):
        #model.train()
        #if epoch %5 == 0:
        t0 = time.time()
        # forward

        z,re_x,re_adj= model(data['g'],data['features'])

        loss=Recon_loss(re_x,re_adj,adj,data['features'],data['train_mask'],loss_fn,GAE_mode)

        #Save training loss
        arr_loss[epoch]=loss.item()
        #

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur=time.time() - t0

        auc,ap,val_loss=fixed_graph_evaluate(args,model,data,adj,data['val_mask'])
        #Save validation set AUC
        arr_valauc[epoch]=auc
        #Save validation set AUC
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                            auc,val_loss, data['n_edges'] / np.mean(dur) / 1000))
        if args.early_stop:
            if stopper.step(auc,val_loss.item(), model,epoch,checkpoints_path):
                break

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))

        #if epoch%100 == 0:

    auc,ap,_ = fixed_graph_evaluate(args,model,data,adj,data['test_mask'])
    test_dur=0
    #Save test set AUC
    arr_testauc[epoch]=auc
    #Save test set AUC
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur,auc,ap))
    #print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    #logger.info("Current epoch: {:d} Test AUROC {:.4f} | Test AUPRC {:.4f}".format(epoch,auc,ap))
    #logger.info(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
    #logger.info('\n')

    #np.savez('Dom3.npz',epoch=arr_epoch,loss=arr_loss,valauc=arr_valauc,testauc=arr_testauc)

    return model

def Recon_loss(re_x,re_adj,adj,x,mask,loss_fn,mode):
    #S_loss: structure loss A_loss: Attribute loss
    if mode=='A':
        return loss_fn(re_x[mask], x[mask])
    if mode=='X':
        return loss_fn(re_x[mask], x[mask])
    if mode=='AX':
        return 0.5*loss_fn(re_x[mask], x[mask]) + 0.5*loss_fn(re_adj[mask], adj[mask])

def ae_anomaly_score(re_x,re_adj,adj,x,mask,loss_fn,mode):
    if mode=='A':
        S_scores=F.mse_loss(re_adj[mask], adj[mask], reduction='none')
        return torch.mean(S_scores,1)
    if mode=='X':
        A_scores=F.mse_loss(re_x[mask], x[mask], reduction='none')
        return torch.mean(A_scores,1)
    if mode=='AX':
        A_scores=F.mse_loss(re_x[mask], x[mask], reduction='none')
        S_scores=F.mse_loss(re_adj[mask], adj[mask], reduction='none')
        return 0.5*torch.mean(A_scores,1)+0.5*torch.mean(S_scores,1)

def ae_fixed_graph_evaluate(args,model,data,adj,mask):
    loss_fn = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]

        loss_mask=mask.bool() & data['labels'].bool()

        #test_t0=time.time()
        z,re_x,re_adj= model(data['g'],data['features'])

        loss=Recon_loss(re_x,re_adj, adj, data['features'],loss_mask,loss_fn,GAE_mode)
        #test_dur = time.time()-test_t0
        #print("Test Time {:.4f}".format(test_dur))
        #print(recon[data['val_mask']].size())
        scores=ae_anomaly_score(re_x,re_adj, adj, data['features'],mask,loss_fn,GAE_mode)

        # A_scores=F.mse_loss(re_x[mask], data['features'][mask], reduction='none')
        # S_scores=F.mse_loss(re_adj[mask], adj[mask], reduction='none')
        # scores=torch.mean(A_scores,1)+torch.mean(S_scores,1)

        labels=labels.cpu().numpy()

        scores=scores.cpu().numpy()
        #pred=thresholding(scores,0)

        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)



    return auc,ap,loss


def tu_train(args, logger,dataset, model, val_dataset=None,path=None):
    '''
    training function
    '''
    checkpoints_path=path

    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    dataloader = dataset
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad,
    #                                     model.parameters()), lr=0.001)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    #early_stopping_logger = {"best_epoch": -1, "val_acc": -1}


    #data_center= init_center(args,input_g,input_feat, model)
    data_center= torch.zeros(args.n_hidden, device=f'cuda:{args.gpu}')
    radius=torch.tensor(0, device=f'cuda:{args.gpu}')# radius R initialized with 0 by default.
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.n_epochs):
        begin_time = time.time()
        # accum_correct = 0
        # total = 0
        print("EPOCH ###### {} ######".format(epoch))
        computation_time = 0.0
        for (batch_idx, (batch_graph, graph_labels)) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                #graph_labels = graph_labels.cuda()
            #print(batch_graph)
            train_mask=~batch_graph.ndata['node_labels'].bool().squeeze()
            model.zero_grad()
            compute_start = time.time()

            normlizing = nn.BatchNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            input_attr=normlizing(batch_graph.ndata['node_attr'])


            outputs = model(batch_graph,input_attr)

            loss,dist,score=loss_function(args.nu, data_center,outputs,radius,train_mask)

            loss.backward()
            batch_compute_time = time.time() - compute_start
            computation_time += batch_compute_time
            optimizer.step()

            print('RRR',radius.data)
            print("Epoch {:05d},loss {:.4f} with {}-th batch time(s) {:.4f}".format(
            epoch, loss.item(), batch_idx, computation_time))
        elapsed_time = time.time() - begin_time
        if val_dataset is not None:
            auc,ap,f1,acc,precision,recall,loss = multi_graph_evaluate(args,checkpoints_path, model, data_center,val_dataset,radius,'val')
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Val AUROC {:.4f} | Val F1 {:.4f} | Val ACC {:.4f} | ". format(
                epoch, elapsed_time, loss.item()*100000, auc,f1,acc))
            torch.cuda.empty_cache()
            if args.early_stop:
                if stopper.step(auc,loss.item()*100000, model,epoch,checkpoints_path):
                    print("best epoch is EPOCH {}, val_auc is {}%".format(stopper.best_epoch,
                                                        stopper.best_score))
                    break


    return model



def train(args,logger,data,model,path,year):
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)

    checkpoints_path=path


    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    # initialize data center

    input_feat = data['features']
    input_g = data['g']
    raw_data = data['data']
    # num_nodes = input_g.num_nodes()

    data_center= init_center(args,input_g,input_feat, model)

    radius=torch.tensor(0, device=device)# radius R initialized with 0 by default.


    #Create a matrix to store the resulting curves
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)

    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        #if epoch %5 == 0:
        t0 = time.time()
        # forward

        outputs= model(input_g,input_feat)

        loss,dist,_=loss_function(args.nu, data_center,outputs,radius,data['train_mask'])
        #Save training loss
        arr_loss[epoch]=loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch>= 3:
            dur.append(time.time() - t0)

        radius.data=torch.tensor(get_radius(dist, args.nu), device=device)


        auc,ap,f1,acc,precision,recall,val_loss, scores = fixed_graph_evaluate(args,checkpoints_path, model, data_center,data,radius,data['val_mask'])
        #Save validation set AUC
        arr_valauc[epoch]=auc
        #Save test set AUC
        print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100000,
                                            val_loss.item()*100000, auc, data['n_edges'] / np.mean(dur) / 1000))
        print(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        if args.early_stop:
            if stopper.step(auc,val_loss.item(), model,epoch,checkpoints_path):
                break

    if args.early_stop:
        print('loading model before testing.')
        model.load_state_dict(torch.load(checkpoints_path))


    auc,ap,f1,acc,precision,recall,loss, scores = fixed_graph_evaluate(args,checkpoints_path,model, data_center,data,radius,data['test_mask'])
    test_dur = 0
    #Save test set AUC
    arr_testauc[epoch]=auc
    #Save test set AUC
    print("Test Time {:.4f} | Test AUROC {:.4f} | Test AUPRC {:.4f}".format(test_dur,auc,ap))
    print(f'Test f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')


    (train_data, test_data, positive_data, negative_data,
     AUC, r5, r10, r30, r100) = get_results(args, raw_data, data, scores,year)

    #tsne_visualization(data, model)
    # return model

    return (train_data, test_data, positive_data, negative_data,
            AUC, auc, ap, f1, acc, precision, recall, r5, r10, r30,r100)



def compute_topk_metrics(df_sorted, percent, label_col = 'label'):
    """
    Compute recall, precision, and F1-score for the top-k percent of ranked samples.

    Parameters:
    - df: DataFrame with 'label' and 'score' columns.
    - percent: top-k percentage (e.g., 5, 10, 30).
    - label_col: name of the column containing true labels.
    - score_col: name of the column containing predicted scores.

    Returns:
    - dict with recall, precision, f1, tp, fp, fn, total_positives.
    """

    assert 0 < percent <= 100, "percent must be in (0, 100]"

    # df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    if percent!=100:
      top_k_count = max(1, int(round((percent / 100.0) * len(df_sorted))))
    else:
      top_k_count = percent
    top_k = df_sorted.iloc[:top_k_count]

    tp = (top_k[label_col] == 1).sum()
    fp = (top_k[label_col] == 0).sum()
    fn = (df_sorted[label_col] == 1).sum() - tp

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        'top_k_percent': percent,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_positives': (df_sorted[label_col] == 1).sum()
    }

def get_results(args, raw_data, data, scores, year):
    train_data = raw_data.iloc[np.where(data['train_mask'].numpy())[0]]
    test_data = raw_data.iloc[np.where(data['test_mask'].numpy())[0]]
    test_data.reset_index(inplace=True, drop=True)

    scores_df = pd.DataFrame(scores)
    scores_df.columns = ['scores' for col in scores_df.columns]

    trial_score = pd.concat([test_data,scores_df], axis=1)
    trial_score = trial_score.reset_index()

    trial_score_sorted = trial_score.sort_values(by=['scores']) #, ascending=False
    trial_score_sorted.reset_index(inplace=True, drop=True)

    positive_data = trial_score_sorted.loc[trial_score_sorted['label'] == 1]
    negative_data = trial_score_sorted.loc[trial_score_sorted['label'] == 0]
    # print(trial_score_sorted.loc[trial_score_sorted['label'] == 1])
    # print(trial_score_sorted.loc[trial_score_sorted['label'] == 0])


    results_df = pd.DataFrame({
        'gene_id': trial_score_sorted['ensembl'],
        'y_true': trial_score_sorted['label'],
        'score' : trial_score_sorted['scores']

    })
    results_df.to_csv(path+f"/models/OCGNN/scores/{args.dataset}_scores_{year}.csv", index=False)


    # anomaly_sampels = trial_score_sorted.loc[trial_score_sorted['label'] == 0]
    avg_rank = pd.DataFrame(positive_data.index).mean()
    AUC = 1-(avg_rank[0]/trial_score_sorted.shape[0])
    print(AUC)


    metrics5 = compute_topk_metrics(trial_score_sorted, percent=5)
    metrics10 = compute_topk_metrics(trial_score_sorted, percent=10)
    metrics30 = compute_topk_metrics(trial_score_sorted, percent=30)
    metrics100 = compute_topk_metrics(trial_score_sorted, percent=100)


    return (train_data, test_data, positive_data, negative_data, AUC,
            metrics5['recall'], metrics10['recall'],
            metrics30['recall'], metrics100['recall'])

def fixed_graph_evaluate(args,path,model, data_center,data,radius,mask):

    model.eval()
    with torch.no_grad():
        labels = data['labels'][mask]
        loss_mask = mask.bool() & data['labels'].bool()

        #test_t0 = time.time()
        outputs= model(data['g'],data['features'])


        #print(loss_mask.)
        _,scores = loss_anomaly_score(data_center,outputs,radius,mask)
        #test_dur = time.time()-test_t0
        loss,_,_=loss_function(args.nu,data_center,outputs,radius,loss_mask)
        #print("Test Time {:.4f}".format(test_dur))

        labels = labels.cpu().numpy()
        #dist=dist.cpu().numpy()
        scores = scores.cpu().numpy()

        threshold=0
        pred=thresholding(scores,threshold)

        auc=roc_auc_score(labels, scores)
        ap=average_precision_score(labels, scores)

        acc=accuracy_score(labels,pred)
        recall=recall_score(labels,pred)
        precision=precision_score(labels,pred)
        f1=f1_score(labels,pred)

        return auc,ap,f1,acc,precision,recall,loss, scores

def multi_graph_evaluate(args,path, model, data_center,dataloader,radius,mode='val'):
    '''
    evaluate function
    '''
    if mode=='test':
        print(f'model loaded.')
        model.load_state_dict(torch.load(path))
    model.eval()
    total_loss=0
    # pred_list=[]
    # labels_list=[]
    # scores_list=[]
    #correct_label = 0
    with torch.no_grad():
        for batch_idx, (batch_graph, graph_labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                for (key, value) in batch_graph.ndata.items():
                    batch_graph.ndata[key] = value.cuda()
                #graph_labels = graph_labels.cuda()

            # normlizing = nn.InstanceNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            # input_attr=normlizing(batch_graph.ndata['node_attr'].unsqueeze(1)).squeeze()

            normlizing = nn.BatchNorm1d(batch_graph.ndata['node_attr'].shape[1], affine=False).cuda()
            input_attr=normlizing(batch_graph.ndata['node_attr'])

            outputs = model(batch_graph,input_attr)

            labels = batch_graph.ndata['node_labels']
            #print(labels.size())
            loss_mask=~labels.bool().squeeze()
            #print(loss_mask.size())
            _,scores=loss_anomaly_score(data_center,outputs,radius,mask=None)
            #print(outputs[loss_mask].size())
            loss,_,_=loss_function(args.nu,data_center,outputs,radius,loss_mask)

            # loss,_,scores=loss_function(args.nu,data_center,outputs,radius,mask=None)
            labels=labels.cpu().numpy().astype('int8')
            #dist=dist.cpu().numpy()
            scores=scores.cpu().numpy()
            pred=thresholding(scores,0)
            #print('pred',pred[:30])
            # print(labels[:10])
            # print(scores[:10])

            total_loss+=loss
            if batch_idx==0:
                labels_vec=labels
                pred_vec=pred
                scores_vec=scores
            else:
                pred_vec=np.append(pred_vec,pred)
                labels_vec=np.concatenate((labels_vec,labels),axis=0)
                scores_vec=np.concatenate((scores_vec,scores),axis=0)

        total_loss/=(batch_idx+1)
        print('score std',scores_vec.std())
        print('score mean',scores_vec.mean())
        print('labels mean',labels_vec.mean())
        print('pred mean',pred_vec.mean())
        auc=roc_auc_score(labels_vec, scores_vec)
        ap=average_precision_score(labels_vec, scores_vec)

        acc=accuracy_score(labels_vec,pred_vec)
        recall=recall_score(labels_vec,pred_vec)
        precision=precision_score(labels_vec,pred_vec)
        f1=f1_score(labels_vec,pred_vec)

    return auc,ap,f1,acc,precision,recall,total_loss

def thresholding(recon_error,threshold):
    ano_pred=np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i]>threshold:
            ano_pred[i]=1
    return ano_pred

def baseline_evaluate(datadict,y_pred,y_score,val=True):

    if val==True:
        mask=datadict['val_mask']
    if val==False:
        mask=datadict['test_mask']

    auc=roc_auc_score(datadict['labels'][mask],y_score)
    ap=average_precision_score(datadict['labels'][mask],y_score)
    acc=accuracy_score(datadict['labels'][mask],y_pred)
    recall=recall_score(datadict['labels'][mask],y_pred)
    precision=precision_score(datadict['labels'][mask],y_pred)
    f1=f1_score(datadict['labels'][mask],y_pred)

    return auc,ap,f1,acc,precision,recall


def main(args, node_data, edges_list,year):
    if args.seed!=-1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        #torch.backends.cudnn.deterministic=True
        dr.seed(args.seed)

    checkpoints_path=f'{path}/models/OCGNN/checkpoints/{args.dataset}+OC-{args.module}+bestcheckpoint.pt'
    # print(checkpoints_path)
    logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    logger=logging.getLogger('OCGNN')


    data=data_loader(args, node_data, edges_list,year)
    input_dim = node_data.shape[1]-3 #128 #100 pubmed #128 ppi
    model=init_model(args,input_dim, data['g'])
    # model=init_model(args,data['input_dim'])

    if args.module != 'GAE':
        # model=train(args,logger,data,model,checkpoints_path)
        (positive, unlabled, positive_data, negative_data, AUC,
          Test_AUROC, Test_AUPRC, f1, acc, pre, recall,
          r5, r10, r30, r100) = train(args,logger,data,model,checkpoints_path,year)
        return (positive, unlabled, positive_data, negative_data, AUC,
                Test_AUROC, Test_AUPRC, f1, acc, pre, recall,
                r5, r10, r30,r100)

    else:
        model=ae_train(args,logger,data,model,checkpoints_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCGNN')
    register_data_args(parser)


    parser.add_argument("--dropout", type=float, default=0.5, #0.5
            help="dropout probability")
    parser.add_argument("--nu", type=float, default=0.3, #0.2, 0.3
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--seed", type=int, default=52,
            help="random seed, -1 means dont fix seed")
    parser.add_argument("--module", type=str, default='GraphSAGE',
            help="GCN/GAT/GIN/GraphSAGE/GAE")
    parser.add_argument('--n-worker', type=int, default=1,
            help='number of workers when dataloading')
    parser.add_argument('--batch-size', type=int, default=64,
            help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--normal-class", type=int, default=0,
            help="normal class")
    parser.add_argument("--n-epochs", type=int, default=1000,
            help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
            help="number of hidden gnn units")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of hidden gnn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, #5e-4
            help="Weight for L2 loss")
    parser.add_argument('--early-stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    parser.add_argument("--norm", action='store_true',
            help="graph normalization (default=False)")
    parser.set_defaults(self_loop=True)
    parser.set_defaults(norm=False)
    args, unknown = parser.parse_known_args()
    #args = parser.parse_args()

    if args.module=='GCN':
        args.self_loop=True
        args.norm=True
    if args.module=='GAE':
        args.lr=0.002
        args.dropout=0.
        args.weight_decay=0.
        # args.n_hidden=32
    #     args.self_loop=True
    # if args.module=='GraphSAGE':
    #     args.self_loop=True


    #fire.Fire(main(args))
    print(args)
    # args.dataset = 'luad'
    args.normal_class=0
    results = []
    year = 2017


    feature = 'ppi'

    file_name = pd.read_csv(path+f"/main_data/train_data/{year}/disease_summary.csv")

    ppi = pd.read_csv(path+f'main_data/{year}/ppi_{year}_700_emb.csv', sep=",")
    node_data = ppi


    edges_list = pd.read_csv(path+f"/main_data/{year}/edge_list_{year}_700.edg", header=None, sep='\t')

    node_data = node_data.rename(columns={'string_id': 'ensembl'})


    for row in file_name['disease_id']:

          print('#######################################')
          print(f'Processing dataset: {row}')

          args.dataset = row
          (positive, unlabled, positive_data, negative_data,
           AUC, Test_AUROC, Test_AUPRC, f1, acc, pre, recall,
           r5, r10, r30, r100) = main(args, node_data, edges_list, year)

          num_train = len(positive_data)
          num_test = len(negative_data)
          num_unlabled = len(unlabled)
          num_positive = len(positive)


          results.append({
              'Dataset': row,
              'Positive': num_positive,
              'unlabled': num_unlabled,
              'Train': num_train,
              'Test': num_test,
              'AUC': AUC,
              'R@5' : r5,
              'R@10': r10,
              'R@30': r30,
              'R@100': r100,

          })

          results_df = pd.DataFrame(results)
          results_df.to_csv(path+f"/results/OCGNN/{args.module}_{year}_early.csv", index=False)
    print(results_df)

   
