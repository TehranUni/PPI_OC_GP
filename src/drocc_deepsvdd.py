
path ='/content/drive/MyDrive/data/'

from __future__ import print_function
import os
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from collections import OrderedDict
import logging
import random


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.data[idx]), (self.labels[idx]), torch.tensor([0])

def ImportTrainData(diseas, year):
  if year:
    train_data = pd.read_csv(f'data/train_2017/{diseas}.csv')
    test_data = pd.read_csv(f'data/test_2017/{diseas}.csv')

  else:
    train_data = pd.read_csv(path+f'main_data/train_data/{diseas}.csv')

  return train_data, test_data

def read_data(disease, node_data, year):
  if year:
    disease_train_data, disease_test_data = ImportTrainData(disease, year)
  else:
    node_data = pd.read_csv(path+f'main_data/full/ppi_full_emb.csv', sep=",")
    disease_data = ImportTrainData(disease, '')

  node_data['test'] = 1
  node_data['label'] = 0


  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1
  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0
  node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1

  negative_data = node_data.loc[(node_data["label"] == 0) & (node_data["test"] == 1)]
  train_positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 0)]
  test_positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 1)]

  train_pos_nodes, val_pos_nodes = train_test_split(train_positive_data, test_size=0.2, random_state=42, shuffle=True)
  validation_neg_nodes, test_neg_nodes = train_test_split(negative_data, test_size=0.2, random_state=42, shuffle=True)
  train_neg_nodes, val_neg_nodes = train_test_split(validation_neg_nodes, test_size=0.2, random_state=42, shuffle=True)



  train_nodes = pd.concat([train_pos_nodes, train_neg_nodes])
  val_nodes = pd.concat([val_pos_nodes, val_neg_nodes])
  test_nodes = pd.concat([test_positive_data, test_neg_nodes])


  test_labels = test_nodes['label']
  train_labels = train_nodes['label']
  val_labels = val_nodes['label']


  return train_nodes, val_nodes, test_nodes, train_labels, val_labels, test_labels


class DROCCTrainer:
    """
    Trainer class that implements the DROCC algorithm proposed in
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device, year):
        """Initialize the DROCC Trainer class

        Parameters
        ----------
        model: Torch neural network object
        optimizer: Total number of epochs for training.
        lamda: Weight given to the adversarial loss
        radius: Radius of hypersphere to sample points from.
        gamma: Parameter to vary projection.
        device: torch.device object for device to use.
        """
        self.model = model
        self.optimizer = optimizer
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.device = device
        self.year = year

    def train(self, train_loader, val_loader, val_nodes, disease, learning_rate, lr_scheduler, total_epochs,
                only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50,
                metric='AUC'):
        """Trains the model on the given training dataset with periodic
        evaluation on the validation dataset.

        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial
                          generation of negative points.
        metric: Metric used for evaluation (AUC / F1).
        """
        best_auc = -np.inf
        best_model = None
        best_scores = []


        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        for epoch in range(total_epochs):
            #Make the weights trainable
            self.model.train()
            lr_scheduler(epoch, total_epochs, only_ce_epochs, learning_rate, self.optimizer)

            #Placeholder for the respective 2 loss values
            epoch_adv_loss = torch.tensor([0]).type(torch.float32).to(self.device)  #AdvLoss
            epoch_ce_loss = 0  #Cross entropy Loss

            batch_idx = -1
            for data, target, _ in train_loader:
                if data.shape[0] == 0:
                    continue
                batch_idx += 1
                data, target = data.to(self.device), target.to(self.device)
                # Data Processing
                data = data.to(torch.float)
                target = target.to(torch.float)
                target = torch.squeeze(target)

                self.optimizer.zero_grad()

                # Extract the logits for cross entropy loss
                logits = self.model(data)
                logits = torch.squeeze(logits, dim = 1)

                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==1).
                '''
                if  epoch >= only_ce_epochs:
                    data = data[target == 1]
                    #for nan Adv_loss
                    if data.shape[0] == 0:
                            loss = ce_loss
                    else:
                    #AdvLoss
                        adv_loss = self.one_class_adv_loss(data)
                        epoch_adv_loss += adv_loss
                        loss = ce_loss + adv_loss * self.lamda
                else:
                    # If only CE based training has to be done
                    loss = ce_loss

                # Backprop
                loss.backward()
                self.optimizer.step()

            epoch_ce_loss = epoch_ce_loss/(batch_idx + 1)  #Average CE Loss
            epoch_adv_loss = epoch_adv_loss/(batch_idx + 1) #Average AdvLoss

            AUC, test_scores = self.test(val_loader, metric, return_scores=True)#newww

            if AUC > best_auc:
                best_auc = AUC
                best_scores = test_scores
                best_model = copy.deepcopy(self.model)


            print('Epoch: {}, CE Loss: {}, AdvLoss: {}, {}: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item(),
                metric, AUC))

        if best_scores is None:
          print("Warning: rank_score is None. Setting it to an empty list.")
          best_scores = []


        self.model = copy.deepcopy(best_model)
        print('\nBest test {}: {}'.format(
            metric, best_auc
        ))


        rank_score_df = pd.DataFrame(best_scores, columns=['label', 'score'])
        val_nodes.reset_index(inplace=True, drop=True)
        val_nodes = val_nodes.drop(columns=['label', 'test'])
        trial_score = pd.concat([val_nodes,rank_score_df], axis=1)

        results_df = pd.DataFrame({
          'gene_id': trial_score['ensembl'],
          'y_true': trial_score['label'],
          'score' : trial_score['score']

        })
        results_df.to_csv(f"data/DROCC/scores/{disease}_scores_{self.year}.csv", index=False)

        positive_data, negative_data, AUC = self.display_sorted_data(best_scores)
        recall_5 = self.recall_at_top_k(best_scores, 5)
        recall_10 = self.recall_at_top_k(best_scores, 10)
        recall_30 = self.recall_at_top_k(best_scores, 30)
        recall_100 = self.recall_at_top_k(best_scores, 100)
        # print(recall_5)
        # print(recall_10)
        # print(recall_30)

        return positive_data, negative_data, AUC, recall_5, recall_10, recall_30, recall_100

    def test(self, test_loader, metric, return_scores=False):#newww
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        metric: Metric used for evaluation (AUC / F1).
        """
        self.model.eval()
        label_score = []
        batch_idx = -1


        for data, target, _ in test_loader:
            if data.shape[0] == 0:
              continue

            batch_idx += 1
            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            logits = torch.squeeze(logits, dim = 1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = logits

            label_score += list(zip(target.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))



        # Compute test score
        labels, scores = zip(*label_score)

        labels = np.array(labels)
        scores = np.array(scores)

        # print(scores)

        if metric == 'F1':
            # Evaluation based on https://openreview.net/forum?id=BJJLHbb0-
            thresh = np.percentile(scores, 20)
            y_pred = np.where(scores >= thresh, 1, 0)
            prec, recall, test_metric, _ = precision_recall_fscore_support(
                labels, y_pred, average="binary")
        # if metric == 'AUC':
        #     # test_metric = roc_auc_score(labels, scores)
        #     positive_data, negative_data, AUC = self.display_sorted_data(label_score)

        if return_scores:
            positive_data, negative_data, AUC = self.display_sorted_data(label_score)
            return AUC, label_score

        else:

          positive_data, negative_data, AUC = self.display_sorted_data(label_score)
          recall_5 = self.recall_at_top_k(label_score, 5)
          recall_10 = self.recall_at_top_k(label_score, 10)
          recall_30 = self.recall_at_top_k(label_score, 30)
          recall_100 = self.recall_at_top_k(label_score, 100)

          return (label_score, positive_data, negative_data, AUC,
                  recall_5, recall_10, recall_30, recall_100)



    def display_sorted_data(self, label_score):

        if not label_score:
          print(f"Warning: No data to sort in Test. Returning empty results.")
          return [], [], 0.0

        sorted_data = sorted(label_score, key=lambda x: x[1], reverse=True)
        sorted_df = pd.DataFrame(sorted_data, columns=['Label', 'Score']).reset_index(drop=True)

        positive_data = sorted_df[sorted_df['Label'] == 1]
        negative_data = sorted_df[sorted_df['Label'] == 0]

        avg_rank = pd.DataFrame(positive_data.index).mean()
        AUC = 1-(avg_rank[0]/sorted_df.shape[0])

        return positive_data.shape[0], negative_data.shape[0], AUC

    def recall_at_top_k(self, label_score, top_percent):

      label_score_sorted = sorted(label_score, key=lambda x: x[1], reverse=True)
      label_score_sorted = pd.DataFrame(label_score_sorted, columns=['Label', 'Score']).reset_index(drop=True)

      if top_percent!=100:
         k = max(1, int(len(label_score_sorted) * (top_percent / 100)))
      else:
          k = top_percent
      top_k = label_score_sorted.iloc[:k]

      total_positives = (label_score_sorted['Label'] == 1).sum()

      if total_positives == 0:
          return 0.0

      true_positives_in_top_k = (top_k['Label'] == 1).sum()
      recall = true_positives_in_top_k / total_positives

      return recall


    def one_class_adv_loss(self, x_train_data):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r))
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """

        if x_train_data.shape[0] == 0:
          return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points
        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data

        for step in range(self.ascent_num_steps):
            with torch.enable_grad():


                new_targets = torch.zeros(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)

                logits = self.model(x_adv_sampled)
                logits = torch.squeeze(logits, dim = 1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)
                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 10==0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2,
                                                dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius,
                                    self.gamma * self.radius).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))
                h = proj * h
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

        return adv_loss

    def save(self, path, disease):
      torch.save(self.model.state_dict(), f'data/DROCC/models/model_deep_{disease}_{self.year}.pt')

    def load(self, path, disease):
        self.model.load_state_dict(torch.load(f'data/DROCC/models/model_deep_{disease}_{self.year}.pt'))



class BaseNet(nn.Module):
    """Base class for all neural networks."""

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rep_dim = None  # representation dimensionality, i.e. dim of the last layer
        self.input_dim = None

    def forward(self, *input):
        """
        Forward pass logic
        :return: Network output
        """
        raise NotImplementedError

    def summary(self):
        """Network summary."""
        net_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in net_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class NNet(BaseNet):
    def __init__(self, size):
        super().__init__()

        self.rep_dim = 1
        self.model = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, 512),
            nn.ELU(),
            nn.LayerNorm(512),#BatchNorm1d
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


def svdd_adjust_learning_rate(epoch, total_epochs, only_ce_epochs, learning_rate, optimizer):
        """Adjust learning rate during training.

        Parameters
        ----------
        epoch: Current training epoch.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        learning_rate: Initial learning rate for training.
        """
        #We dont want to consider the only ce
        #based epochs for the lr scheduler
        epoch = epoch - only_ce_epochs
        drocc_epochs = total_epochs - only_ce_epochs
        # lr = learning_rate
        if epoch <= drocc_epochs:
            lr = learning_rate * 0.01
        if epoch <= 0.80 * drocc_epochs:
            lr = learning_rate * 0.1
        if epoch <= 0.40 * drocc_epochs:
            lr = learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return optimizer


def create_tensor(data):
  data_tensor = torch.Tensor(data.to_numpy()).type(torch.float32)
  return data_tensor

def create_data_loader(data, label, idx, shuffle=False):
  dataset = TensorDataset(data, label, idx)
  #optimization
  dataset_loader = DataLoader(dataset, batch_size=args.batch_size)
  return dataset_loader

def svdd_load_data(train_data, val_data, test_data, train_lab, val_lab, test_lab):


    train = create_tensor(train_data)
    train_label = create_tensor(train_lab)
    train_idx = create_tensor(train_data.index)
    train_loader = create_data_loader(train, train_label, train_idx,shuffle=True)

    val = create_tensor(val_data)
    val_label = create_tensor(val_lab)
    val_idx = create_tensor(val_data.index)
    val_loader = create_data_loader(val, val_label, val_idx)

    test = create_tensor(test_data)
    test_label = create_tensor(test_lab)
    test_idx = create_tensor(test_data.index)
    test_loader = create_data_loader(test, test_label, test_idx)



    return train_loader, val_loader, test_loader

"""### main"""

def svdd_main(disease, node_data, year):

    train_nodes, val_nodes, test_nodes, train_lab, val_lab, test_lab = read_data(disease,
                                                                                 node_data, year)
    train_data = train_nodes.drop(columns=['ensembl', 'label', 'test'])
    val_data = val_nodes.drop(columns=['ensembl', 'label', 'test'])
    test_data = test_nodes.drop(columns=['ensembl', 'label', 'test'])

    train_loader,  val_loader, test_loader = svdd_load_data(train_data, val_data, test_data, train_lab, val_lab, test_lab)


    model = NNet(train_data.shape[1]).to(device)
    model = nn.DataParallel(model)

    if args.optim == 1:
        optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr,
                                  momentum=args.mom)
        print("using SGD")
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr)
        print("using Adam")

    trainer = DROCCTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device, year)

    if args.eval == 0:
        # Training the model

        positive_data, negative_data, AUC, recall_5, recall_10, recall_30, recall_100 = trainer.train(train_loader, val_loader, val_nodes, disease, args.lr,  svdd_adjust_learning_rate, args.epochs,
            metric=args.metric, ascent_step_size=args.ascent_step_size, only_ce_epochs = args.only_ce_epochs)

        trainer.save(args.model_dir, disease )
        return train_data.shape[0], test_data.shape[0], positive_data, negative_data, AUC, recall_5, recall_10, recall_30, recall_100

    else:
        if os.path.exists(os.path.join(args.model_dir, f'model_deep_{disease}_{year}.pt')):
            trainer.load(args.model_dir, disease)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()
        (label_score, positive_data, negative_data, AUC,
         recall_5, recall_10, recall_30, recall_100) = trainer.test(test_loader, 'AUC')
        test_label_score_df = pd.DataFrame(label_score, columns=['label', 'score'])
        test_nodes.reset_index(inplace=True, drop=True)
        test_data = test_nodes.drop(columns=['label', 'test'])
        trial_score = pd.concat([test_data, test_label_score_df], axis=1)
        results_df = pd.DataFrame({
            'gene_id': trial_score['ensembl'],
            'y_true': trial_score['label'],
            'score': trial_score['score']

        })

        results_df.to_csv(f"data/DROCC/scores/{disease}_scores_{year}.csv", index=False)
        print('Test AUC: {}'.format(AUC))
        return (train_data.shape[0], test_data.shape[0], positive_data, negative_data, AUC,
                recall_5, recall_10, recall_30, recall_100)



if __name__ == '__main__':
    torch.set_printoptions(precision=5)

    parser = argparse.ArgumentParser(description='PyTorch Simple Training')
    parser.add_argument('--normal_class', type=int, default=0, metavar='N',
                    help='CIFAR10 normal class index')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('-oce,', '--only_ce_epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train with only CE loss')
    parser.add_argument('--ascent_num_steps', type=int, default=100, metavar='N',
                        help='Number of gradient ascent steps')
    parser.add_argument('--hd', type=int, default=128, metavar='N',
                        help='Num hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                        help='step size of gradient ascent')
    parser.add_argument('--mom', type=float, default=0.0, metavar='M',
                        help='momentum')
    parser.add_argument('--model_dir', default='log',
                        help='path where to save checkpoint')
    parser.add_argument('--one_class_adv', type=int, default=1, metavar='N',
                        help='adv loss to be used or not, 1:use 0:not use(only CE)')
    parser.add_argument('--radius', type=float, default=0.2, metavar='N',
                        help='radius corresponding to the definition of set N_i(r)')
    parser.add_argument('--lamda', type=float, default=1, metavar='N',
                        help='Weight to the adversarial loss')
    parser.add_argument('--reg', type=float, default=0, metavar='N',
                        help='weight reg')
    parser.add_argument('--eval', type=int, default=0, metavar='N',
                        help='whether to load a saved model and evaluate (0/1)')
    parser.add_argument('--optim', type=int, default=0, metavar='N',
                        help='0 : Adam 1: SGD')
    parser.add_argument('--gamma', type=float, default=2.0, metavar='N',
                        help='r to gamma * r projection for the set N_i(r)')
    parser.add_argument('-d', '--data_path', type=str, default='.')
    parser.add_argument('--metric', type=str, default='AUC')


    args_list = [
        "--normal_class", "1",
        "--batch_size", "256",
        "--epochs", "1000",
        "--only_ce_epochs", "15",
        "--ascent_num_steps", "100",
        "--hd", "128",
        "--lr", "0.001",
        "--ascent_step_size", "0.001",
        "--mom", "0.0",
        "--model_dir", 'data/DROCC/models/',
        "--one_class_adv", "1",
        "--radius", "17",
        "--lamda", "0.5",
        "--reg", "0",
        "--eval", "0",
        "--optim", "0",
        "--gamma", "2.0",
        "-d", 'data/DROCC/models/',
        "--metric", "AUC"
    ]


    args = parser. parse_args(args_list)

    # settings
    #Checkpoint store path
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    year = 2017
    feature = 'ppi'

    file_name = pd.read_csv(f"data/disease_summary_{year}.csv")
    ppi = pd.read_csv(f'data/embeddings/ppi_{year}_700_emb.csv', sep=",")
    node_data = ppi
    node_data = node_data.rename(columns={'string_id': 'ensembl'})

    results = []

    for row in file_name['disease_id']:

        #train and test
        print(f'Processing dataset: {row}')
        (train_size, test_size, num_positive_test, num_negative_test, AUC, recall_5,
         recall_10, recall_30, recall_100) = svdd_main(row, node_data, year)
        results.append({
            'Dataset': row,
            'Train Size': train_size,
            'Test Size': test_size,
            'Positive Test': num_positive_test,
            'Negative Test': num_negative_test,
            'AUC': AUC,
            'R@5' : recall_5,
            'R@10' : recall_10,
            'R@30' : recall_30,
            'R@100': recall_100,
        })


        results_df = pd.DataFrame(results)

        results_df.to_csv(f"data/DROCC/results/ppi_deepsvdd_{year}.csv", index=False)

