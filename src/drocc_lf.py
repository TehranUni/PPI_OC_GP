
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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_distances
from collections import OrderedDict
import logging


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
    train_data = pd.read_csv(f'../data/train_2017/{diseas}.csv')
    test_data = pd.read_csv(f'../data/test_2017/{diseas}.csv')
    close_neg = pd.read_csv(f'../data/close_neg/{diseas}.csv')

  else:
    train_data = pd.read_csv(path+f'main_data/train_data/{diseas}.csv')

  return train_data, test_data , close_neg


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
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.3),

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



def create_tensor_lf(data):
  data_tensor = torch.Tensor(data.to_numpy()).type(torch.float32)
  return data_tensor

def create_data_loader_lf(data, label, idx):
  dataset = TensorDataset(data, label, idx)
  dataset_loader = DataLoader(dataset, batch_size=args.batch_size)
  return dataset_loader

def lf_load_data(train_data, val_data, test_data, train_lab, val_lab, test_lab, close_neg_data, close_neg_lab):


    train = create_tensor_lf(train_data)
    train_label = create_tensor_lf(train_lab)
    train_idx = create_tensor_lf(train_data.index)
    train_loader = create_data_loader_lf(train, train_label, train_idx)

    val = create_tensor_lf(val_data)
    val_label = create_tensor_lf(val_lab)
    val_idx = create_tensor_lf(val_data.index)
    val_loader = create_data_loader_lf(val, val_label, val_idx)

    test = create_tensor_lf(test_data)
    test_label = create_tensor_lf(test_lab)
    test_idx = create_tensor_lf(test_data.index)
    test_loader = create_data_loader_lf(test, test_label, test_idx)
    if close_neg_data is not None and not close_neg_data.empty:
    # if close_neg_data:
      close_neg = create_tensor_lf(close_neg_data)
      close_neg_label = create_tensor_lf(close_neg_lab)
      close_neg_idx = create_tensor_lf(close_neg_data.index)
      close_neg_loader = create_data_loader_lf(close_neg, close_neg_label, close_neg_idx)

      return train_loader, val_loader, test_loader , close_neg_loader
    else:
      return train_loader, val_loader, test_loader

def read_data_close(disease, node_data, year):
  if year:
    disease_train_data, disease_test_data, disease_close_neg = ImportTrainData(disease, year)
  else:
    node_data = pd.read_csv(path+f'main_data/full/ppi_full_emb.csv', sep=",")
    disease_data = ImportTrainData(disease, '')

  node_data['test'] = 1
  node_data['label'] = 0


  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1
  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0
  node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1

  train_positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 0)]
  test_positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 1)]
  close_neg = node_data.loc[node_data['ensembl'].isin(disease_close_neg['ensembl'])]

   ## for disease that  have close neg
  if close_neg.shape[0] > 1000:
    close_neg_sample_size = 10 * train_positive_data.shape[0]
    sample_size = min(close_neg_sample_size, close_neg.shape[0])
    val_close_neg = close_neg.sample(n=sample_size, random_state=42)
    node_data.loc[node_data['ensembl'].isin(val_close_neg['ensembl']), 'test'] = 0
  else:
     node_data.loc[node_data['ensembl'].isin(close_neg['ensembl']),'test'] = 0

  close_neg_nodes = node_data.loc[(node_data["label"] == 0) & (node_data["test"] == 0)]
  negative_data = node_data.loc[(node_data["label"] == 0) & (node_data["test"] == 1)]


  train_pos_nodes, val_pos_nodes = train_test_split(train_positive_data, test_size=0.2, shuffle=True)
  test_neg_nodes, validation_neg_nodes = train_test_split(negative_data, test_size=0.2, shuffle=True)
  val_neg_nodes, train_neg_nodes  = train_test_split(validation_neg_nodes, test_size=0.2, shuffle=True)

  train_nodes = pd.concat([train_pos_nodes, train_neg_nodes])
  val_nodes = pd.concat([val_pos_nodes, val_neg_nodes])
  test_nodes = pd.concat([test_positive_data, test_neg_nodes])


  train_labels = train_nodes['label']
  val_labels = val_nodes['label']
  test_labels = test_nodes['label']
  close_neg_labels = close_neg_nodes['label']

  close_neg_data = close_neg_nodes.drop(columns=['ensembl', 'label', 'test'])


  return train_nodes, val_nodes, test_nodes, train_labels, val_labels, test_labels, close_neg_data, close_neg_labels

def read_data_lf(disease, node_data,year):
  if year:
    disease_train_data, disease_test_data, _ = ImportTrainData(disease, year)

  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1 #train disease gene
  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0
  node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1 #test disease gene

  negative_data = node_data.loc[(node_data["label"] == 0) & (node_data["test"] == 1)]
  train_positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 0)]
  test_positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 1)]


  train_pos_nodes, val_pos_nodes = train_test_split(train_positive_data, test_size=0.2, shuffle=True)
  test_neg_nodes, validation_neg_nodes = train_test_split(negative_data, test_size=0.2, shuffle=True)
  val_neg_nodes, train_neg_nodes = train_test_split(validation_neg_nodes, test_size=0.2, shuffle=True)

  train_nodes = pd.concat([train_pos_nodes, train_neg_nodes])
  val_nodes = pd.concat([val_pos_nodes, val_neg_nodes])
  test_nodes = pd.concat([test_positive_data, test_neg_nodes])


  train_labels = train_nodes['label']
  val_labels = val_nodes['label']
  test_labels = test_nodes['label']

  return train_nodes, val_nodes, test_nodes, train_labels, val_labels, test_labels


def recall_at_top_k(positive_scores, far_neg_scores, close_neg_scores, top_percent):
    all_scores = np.concatenate([positive_scores, far_neg_scores, close_neg_scores])
    if top_percent!=100:
        k = int(len(all_scores) * top_percent)
    else:
        k = top_percent
    top_k_threshold = np.sort(all_scores)[-k]
    tp = np.sum(positive_scores >= top_k_threshold)
    recall = tp / len(positive_scores)
    return recall

def cal_precision_recall(positive_scores, far_neg_scores, close_neg_scores, fpr):
    """
    Computes the precision and recall for the given false positive rate.
    """
    all_neg_scores = np.concatenate((far_neg_scores, close_neg_scores), axis = 0)
    num_neg = all_neg_scores.shape[0]
    idx = int((1-fpr) * num_neg)

    all_neg_scores.sort()
    thresh = all_neg_scores[idx]
    tp = np.sum(positive_scores > thresh)
    recall = tp/positive_scores.shape[0]
    fp = int(fpr * num_neg)
    precision = tp/(tp+fp)
    return precision, recall

def normalize_grads(grad):
    """
    Utility function to normalize the gradients.
    grad: (batch, -1)
    """
    # make sum equal to the size of second dim
    grad_norm = torch.sum(torch.abs(grad), dim=1)
    grad_norm = torch.unsqueeze(grad_norm, dim = 1)
    grad_norm = grad_norm.repeat(1, grad.shape[1])
    grad = grad/grad_norm * grad.shape[1]
    return grad

def compute_mahalanobis_distance(grad, diff, radius, device, gamma):
    """
    Compute the mahalanobis distance.
    grad: (batch,-1)
    diff: (batch,-1)
    """
    mhlnbs_dis = torch.sqrt(torch.sum(grad*diff**2, dim=1))
    #Categorize the batches based on mahalanobis distance
    #lamda = 1 : mahalanobis distance < radius
    #lamda = 2 : mahalanobis distance > gamma * radius
    lamda = torch.zeros((grad.shape[0],1))
    lamda[mhlnbs_dis < radius] = 1
    lamda[mhlnbs_dis > (gamma * radius)] = 2
    return lamda, mhlnbs_dis

def check_left_part1(lam, grad, diff, radius, device):
    #Part 1 condition value
    n1 = diff**2 * lam**2 * grad**2
    d1 = (1 + lam * grad)**2 + 1e-10
    term = n1/d1
    term_sum = torch.sum(term)
    return term_sum

def check_left_part2(nu, grad, diff, radius, device, gamma):
    #Part 2 condition value
    n1 = diff**2 * grad**2
    d1 = (nu + grad)**2 + 1e-10
    term = n1/d1
    term_sum = torch.sum(term)
    return term_sum

def check_right_part1(lam, grad, diff, radius, device):
    #Check if 'such that' condition is true in proposition 1 part 1
    n1 = grad
    d1 = (1 + lam * grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum > radius**2:
        return check_left_part1(lam, grad, diff, radius, device)
    else:
        return np.inf

def check_right_part2(nu, grad, diff, radius, device, gamma):
    #Check if 'such that' condition is true in proposition 1 part 2
    n1 = grad*nu**2
    d1 = (nu + grad)**2 + 1e-10
    term = diff**2 * n1/d1
    term_sum = torch.sum(term)
    if term_sum < (gamma*radius)**2:
        return check_left_part2(nu, grad, diff, radius, device, gamma)
    else:
        # return torch.tensor(float('inf'))
        return np.inf

def range_lamda_lower(grad):
    #Gridsearch range for lamda
    lam, _ = torch.max(grad, dim=1)
    eps, _ = torch.min(grad, dim=1)
    lam = -1 / lam + eps*0.0001
    return lam

def range_nu_upper(grad, mhlnbs_dis, radius, gamma):
    #Gridsearch range for nu
    alpha = (gamma*radius)/mhlnbs_dis
    max_sigma, _ = torch.max(grad, dim=1)
    nu = (alpha/(1-alpha))*max_sigma
    return nu

def optim_solver(grad, diff, radius, device, gamma=2):
    """
    Solver for the optimization problem presented in Proposition 1 in
    https://arxiv.org/abs/2002.12718
    """
    lamda, mhlnbs_dis = compute_mahalanobis_distance(grad, diff, radius, device, gamma)
    lamda_lower_limit = range_lamda_lower(grad).detach().cpu().numpy()
    nu_upper_limit = range_nu_upper(grad, mhlnbs_dis, radius, gamma).detach().cpu().numpy()

    #num of values of lamda and nu samples in the allowed range
    num_rand_samples = 40
    final_lamda =  torch.zeros((grad.shape[0],1))

    #Solve optim for each example in the batch
    for idx in range(lamda.shape[0]):
        #Optim corresponding to mahalanobis dis < radius
        if lamda[idx] == 1:
            min_left = np.inf
            best_lam = 0
            for k in range(num_rand_samples):
                val = np.random.uniform(low = lamda_lower_limit[idx], high = 0)
                left_val = check_right_part1(val, grad[idx], diff[idx], radius, device)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val

            final_lamda[idx] = best_lam

        #Optim corresponding to mahalanobis dis > gamma * radius
        elif lamda[idx] == 2:
            min_left = np.inf
            best_lam = np.inf
            for k in range(num_rand_samples):
                val = np.random.uniform(low = 0, high = nu_upper_limit[idx])
                left_val = check_right_part2(val, grad[idx], diff[idx], radius, device, gamma)
                if left_val < min_left:
                    min_left = left_val
                    best_lam = val

            final_lamda[idx] = 1.0/best_lam

        else:
            final_lamda[idx] = 0

    final_lamda = final_lamda.to(device)
    for j in range(diff.shape[0]):
        diff[j,:] = diff[j,:]/(1+final_lamda[j]*grad[j,:])

    return diff

def get_gradients(model, device, data, target):
    """
    Utility function to compute the gradients of the model on the
    given data.
    """
    total_train_pts = len(data)
    data = data.to(torch.float)
    target = target.to(torch.float)
    target = torch.squeeze(target)

    #Extract the logits for cross entropy loss
    data_copy = data
    data_copy = data_copy.detach().requires_grad_()
    # logits = model(data_copy)
    logits = model(data_copy)
    logits = torch.squeeze(logits, dim = 1)
    ce_loss = F.binary_cross_entropy_with_logits(logits, target)

    grad = torch.autograd.grad(ce_loss, data_copy)[0]

    return torch.abs(grad)


class DROCCLFTrainer:
    """
    Trainer class that implements the DROCC-LF algorithm proposed for
    one-class classification with limited negative data presented in
    https://arxiv.org/abs/2002.12718
    """

    def __init__(self, model, optimizer, lamda, radius, gamma, device, year):
        """Initialize the DROCC-LF Trainer class

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

    def train(self, disease, train_loader, val_loader, test_nodes, closeneg_val_loader, learning_rate, lr_scheduler, total_epochs,
                only_ce_epochs=50, ascent_step_size=0.001, ascent_num_steps=50):
        """Trains the model on the given training dataset with periodic
        evaluation on the validation dataset.

        Parameters
        ----------
        train_loader: Dataloader object for the training dataset.
        val_loader: Dataloader object for the validation dataset with far negatives.
        closeneg_val_loader: Dataloader object for the validation dataset with close negatives.
        learning_rate: Initial learning rate for training.
        total_epochs: Total number of epochs for training.
        only_ce_epochs: Number of epochs for initial pretraining.
        ascent_step_size: Step size for gradient ascent for adversarial
                          generation of negative points.
        ascent_num_steps: Number of gradient ascent steps for adversarial
                          generation of negative points.
        """
        best_recall_fpr03 = -np.inf
        best_precision_fpr03 = -np.inf
        best_recall_fpr05 = -np.inf
        best_precision_fpr05 = -np.inf
        best_AUC = -np.inf
        best_model = None

        best_pos_scores = []
        best_neg_scores = []
        best_close_scores = []

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

                target = target.view_as(logits)

                ce_loss = F.binary_cross_entropy_with_logits(logits, target)
                # Add to the epoch variable for printing average CE Loss
                epoch_ce_loss += ce_loss

                '''
                Adversarial Loss is calculated only for the positive data points (label==1).
                '''
                if  epoch >= only_ce_epochs:
                    data = data[target == 1]
                    if data.shape[0] == 0:
                        continue
                    target = torch.ones(data.shape[0]).to(self.device)
                    gradients = get_gradients(self.model, self.device, data, target)
                    # AdvLoss
                    adv_loss = self.one_class_adv_loss(data, gradients)
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

            _, pos_scores, far_neg_scores = self.test(val_loader,  get_auc=False)
            _, _, close_neg_scores= self.test(closeneg_val_loader, get_auc=False)


            label_score = [(1, s) for s in pos_scores] + \
                          [(0, s) for s in far_neg_scores] + \
                          [(0, s) for s in close_neg_scores]


            pos, neg, AUC = self.display_sorted_data(label_score)


            if AUC > best_AUC:
                best_AUC = AUC
                best_pos_scores = pos_scores
                best_neg_scores = far_neg_scores
                best_close_scores = close_neg_scores

                best_model = copy.deepcopy(self.model)

            print('Epoch: {}, CE Loss: {}, AdvLoss: {}'.format(
                epoch, epoch_ce_loss.item(), epoch_adv_loss.item()))
            print('AUC : {}'.format(AUC))

        recall_5 = recall_at_top_k(best_pos_scores, best_neg_scores, best_close_scores, 0.05)
        recall_10 = recall_at_top_k(best_pos_scores, best_neg_scores, best_close_scores, 0.1)
        recall_30 = recall_at_top_k(best_pos_scores, best_neg_scores, best_close_scores, 0.3)
        recall_100 = recall_at_top_k(best_pos_scores, best_neg_scores, best_close_scores, 100)

        best_label_score = [(1, s) for s in best_pos_scores] + \
                      [(0, s) for s in best_neg_scores] + \
                      [(0, s) for s in best_close_scores]


        label_score_df = pd.DataFrame(best_label_score, columns=['label', 'score'])
        label_score_df.to_csv(f"../results/DROCC_LF1/scores_train/{disease}_scoresLF1_{self.year}.csv", index=False)


        test_label_score = [(1, s) for s in best_pos_scores] + \
                           [(0, s) for s in best_neg_scores]

        test_label_score_df = pd.DataFrame(test_label_score, columns=['label', 'score'])
        test_nodes.reset_index(inplace=True, drop=True)
        test_data = test_nodes.drop(columns=['label', 'test'])
        trial_score = pd.concat([test_data, test_label_score_df], axis=1)
        results_df = pd.DataFrame({
            'gene_id': trial_score['ensembl'],
            'y_true': trial_score['label'],
            'score': trial_score['score']

        })
        results_df.to_csv(f"../results/DROCC_LF1/scores_train/{disease}_scores_dataLF1_{self.year}.csv", index=False)




        self.model = copy.deepcopy(best_model)


        print('\nBest test AUC : {}'.format(best_AUC))
        print('\nRecall@5% : {}'.format(recall_5))
        print('\nRecall@10% : {}'.format(recall_10))
        print('\nRecall@30% : {}'.format(recall_30))

        print(f'\npositives : {pos}')
        print(f'\nnegatives : {neg}')
        return best_AUC, recall_5, recall_10, recall_30, recall_100


    def test(self, test_loader, get_auc = True):
        """Evaluate the model on the given test dataset.

        Parameters
        ----------
        test_loader: Dataloader object for the test dataset.
        """
        label_score = []
        batch_idx = -1

        # with torch.no_grad():
        for data, target, _ in test_loader:

            batch_idx += 1

            data, target = data.to(self.device), target.to(self.device)
            data = data.to(torch.float)
            target = target.to(torch.float)
            target = torch.squeeze(target)

            logits = self.model(data)
            logits = torch.squeeze(logits, dim = 1)
            sigmoid_logits = torch.sigmoid(logits)
            scores = sigmoid_logits


            target_list = target.cpu().data.numpy().tolist()
            scores_list = scores.cpu().data.numpy().tolist()

            if isinstance(target_list, float):
                target_list = [target_list]

            if isinstance(scores_list, float):
                scores_list = [scores_list]

            label_score += list(zip(target_list, scores_list))


        labels, scores  = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        pos_scores = scores[labels==1]
        neg_scores = scores[labels==0]
        auc = -1

        if get_auc:
          auc = roc_auc_score(labels, scores)

        return auc, pos_scores, neg_scores

    def display_sorted_data(self, label_score):


      if not label_score:
        print(f"Warning: No data to sort in Test. Returning empty results.")
        return [], [], 0.0

      sorted_data = sorted(label_score, key=lambda x: x[1], reverse=True)
      sorted_df = pd.DataFrame(sorted_data, columns=['Label', 'Score']).reset_index(drop=True)
      sorted_df["rank"] = sorted_df["Score"].rank(method="average", ascending=False)

      positive_data = sorted_df[sorted_df['Label'] == 1]
      negative_data = sorted_df[sorted_df['Label'] == 0]


      avg_rank_pos = sorted_df.loc[sorted_df["Label"] == 1, "rank"].mean()
      AUC = 1-(avg_rank_pos/sorted_df.shape[0])

      return positive_data.shape[0], negative_data.shape[0], AUC

    def one_class_adv_loss(self, x_train_data, gradients):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r)
            classified as +ve (label=0). This is done by maximizing
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R
            (set N_i(r) with mahalanobis distance as a distance measure),
            by solving the optimization problem
        4) Pass the calculated adversarial points through the model,
            and calculate the CE loss wrt target class 0

        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        gradients: gradients of the model for the given data.
        """
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

            if (step + 1) % 5==0:
                # Project the normal points to the set N_i(r) based on mahalanobis distance
                h = x_adv_sampled - x_train_data
                h_flat = torch.reshape(h, (h.shape[0], -1))
                gradients_flat = torch.reshape(gradients, (gradients.shape[0], -1))
                #Normalize the gradients
                gradients_normalized = normalize_grads(gradients_flat)
                #Solve the non-convex 1D optimization
                h_flat = optim_solver(gradients_normalized, h_flat, self.radius, self.device, self.gamma)
                h = torch.reshape(h_flat, h.shape)
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere

        adv_pred = self.model(x_adv_sampled)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, (new_targets * 0))

        return adv_loss

    def save(self, path, disease):
        torch.save(self.model.state_dict(),os.path.join(f'model_LF1_{self.year}_{disease}.pt'))

    def load(self, path, disease):
        self.model.load_state_dict(torch.load(os.path.join(path, f'model_LF1_{self.year}_{disease}.pt')))


def get_close_negs(positive_embeddings, radius=0.5, num_negatives_per_gene=5):
  """
  Generate artificial close-negative samples from positive gene embeddings.

  Args:
      positive_embeddings (np.ndarray): array of shape (N, D), where N = number of genes, D = embedding size
      radius (float): radius of the hypersphere around each positive sample
      num_negatives_per_gene (int): how many negatives to generate per positive gene

  Returns:
      close_neg_data (np.ndarray): array of shape (N * num_negatives_per_gene, D)
  """
  N, D = positive_embeddings.shape
  close_neg_data = []

  for vec in positive_embeddings:
      for _ in range(num_negatives_per_gene):
          # Sample a random unit vector (direction)
          direction = np.random.randn(D)
          direction /= np.linalg.norm(direction)  # normalize

          # Move in that direction with given radius
          perturbed = vec + radius * direction
          close_neg_data.append(perturbed)

  close_neg_data = np.array(close_neg_data, dtype=np.float32)
  close_neg_labels = np.zeros((close_neg_data.shape[0]))


  return CustomDataset(close_neg_data, close_neg_labels), close_neg_data.shape[0]



def lf_main(disease, node_data, year):

    ### for read close neg  (LF1)
    train_nodes, val_nodes, test_nodes, train_lab, val_lab, test_lab, close_neg_data, close_neg_lab = read_data_close(disease, node_data, year)
    train_data = train_nodes.drop(columns=['ensembl', 'label', 'test'])
    val_data = val_nodes.drop(columns=['ensembl', 'label', 'test'])
    test_data = test_nodes.drop(columns=['ensembl', 'label', 'test'])
    train_loader, val_loader, test_loader, close_neg_loader = lf_load_data(train_data, val_data, test_data, train_lab, val_lab, test_lab, close_neg_data, close_neg_lab)

    ###for create close neg(LF2)
    # train_nodes, test_nodes, train_lab, test_lab = read_data_lf(disease, node_data,year)
    # train_data = train_nodes.drop(columns=['ensembl', 'label', 'test'])
    # test_data = test_nodes.drop(columns=['ensembl', 'label', 'test'])
    # train_loader, test_loader = lf_load_data(train_data, test_data, train_lab, test_lab, '', '')
    # positive_embeddings = train_data.to_numpy(dtype=np.float32)
    # close_neg_data = get_close_negs(positive_embeddings, radius=0.3, num_negatives_per_gene=3)
    # # print(close_neg_data)
    # close_neg_loader = DataLoader(close_neg_data, args.batch_size, shuffle=True)


    # train_nodes, val_nodes, test_nodes, train_lab, val_lab, test_lab = read_data_lf(disease,
    #                                                                                 node_data,year)
    # positive_embeddings = train_nodes.loc[(train_nodes["label"] == 1)]
    # train_data = train_nodes.drop(columns=['ensembl', 'label', 'test'])
    # test_data = test_nodes.drop(columns=['ensembl', 'label', 'test'])
    # val_data = val_nodes.drop(columns=['ensembl', 'label', 'test'])
    # positive_embeddings = positive_embeddings.drop(columns=['ensembl', 'label', 'test'])
    #
    # train_loader, val_loader, test_loader = lf_load_data(train_data, val_data, test_data, train_lab, val_lab, test_lab, '', '')
    # positive_embeddings = positive_embeddings.to_numpy(dtype=np.float32)
    # close_neg_data, close_neg = get_close_negs(positive_embeddings, radius=0.3, num_negatives_per_gene=3)
    # # print(close_neg_data)
    # close_neg_loader = DataLoader(close_neg_data, args.batch_size, shuffle=True)


    train_pos = train_nodes[train_nodes["label"] == 1].shape[0]
    train_neg = train_nodes[train_nodes["label"] == 0].shape[0]
    val_pos = val_nodes[val_nodes["label"] == 1].shape[0]
    val_neg = val_nodes[val_nodes["label"] == 0].shape[0]
    test_pos = test_nodes[test_nodes["label"] == 1].shape[0]
    test_neg = test_nodes[test_nodes["label"] == 0].shape[0]
    close_neg = close_neg_data.shape[0]

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

    trainer = DROCCLFTrainer(model, optimizer, args.lamda, args.radius, args.gamma, device, year)

    if args.eval==0:
        # Training the model

        AUC, recall_5, recall_10, recall_30, recall_100 = trainer.train(disease, train_loader, val_loader, val_nodes, close_neg_loader, args.lr, svdd_adjust_learning_rate, args.epochs,
            ascent_step_size=args.ascent_step_size, only_ce_epochs = args.only_ce_epochs)

        trainer.save(args.model_dir,disease)
        return AUC, recall_5, recall_10, recall_30, recall_100, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, close_neg

    else:
        if os.path.exists(os.path.join(args.model_dir, 'model_LF1_{year}_'+disease+'.pt')):
            trainer.load(args.model_dir, disease)
            print("Saved Model Loaded")
        else:
            print('Saved model not found. Cannot run evaluation.')
            exit()


        _, pos_scores, far_neg_scores  = trainer.test(test_loader, get_auc=False)

        test_label_score = [(1, s) for s in pos_scores] + \
                           [(0, s) for s in far_neg_scores]


        pos, neg, AUC = trainer.display_sorted_data(test_label_score)

        recall_5 = recall_at_top_k(pos_scores, far_neg_scores, '', 0.05)
        recall_10 = recall_at_top_k(pos_scores, far_neg_scores, '', 0.1)
        recall_30 = recall_at_top_k(pos_scores, far_neg_scores, '', 0.3)
        recall_100 = recall_at_top_k(pos_scores, far_neg_scores, '', 100)


        test_label_score_df = pd.DataFrame(test_label_score, columns=['label', 'score'])
        test_nodes.reset_index(inplace=True, drop=True)
        test_data = test_nodes.drop(columns=['label', 'test'])
        trial_score = pd.concat([test_data, test_label_score_df], axis=1)
        results_df = pd.DataFrame({
            'gene_id': trial_score['ensembl'],
            'y_true': trial_score['label'],
            'score': trial_score['score']

        })
        results_df.to_csv(f"../results/DROCC_LF1/scores/{disease}_scores_data_LF1_{year}.csv", index=False)


        print(f"Number of Positive Samples: {pos}")
        print(f"Number of Negative Samples: {neg}")
        print(f"AUC: {AUC}")
        return AUC, recall_5, recall_10, recall_30, recall_100, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, close_neg

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
    parser.add_argument('--ascent_num_steps', type=int, default=50, metavar='N',
                        help='Number of gradient ascent steps')
    parser.add_argument('--hd', type=int, default=128, metavar='N',
                        help='Num hidden nodes for LSTM model')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                        help='step size of gradient ascent')
    parser.add_argument('--mom', type=float, default=0.99, metavar='M',
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


    args_list = [
        "--normal_class", "1",
        "--batch_size", "128",
        "--epochs", "1000",
        "--only_ce_epochs", "50",
        "--ascent_num_steps", "50",
        "--hd", "128",
        "--lr", "0.1",
        "--ascent_step_size", "0.001",
        "--mom", "0.99",
        "--model_dir", path+'models/DROCC/LF/models',
        "--one_class_adv", "1",
        "--radius", "5",
        "--lamda", "1",
        "--reg", "0",
        "--eval", "0",
        "--optim", "0",
        "--gamma", "2.0",
        "-d", '../results/DROCC_LF1/models'
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
    results = []

    feature = 'ppi'

    file_name = pd.read_csv(f"../data/disease_summary_{year}.csv")

    ppi = pd.read_csv(f'../data/embeddings/ppi_{year}_700_emb.csv', sep=",")

    # node_datadata = ppi
    node_data = ppi.loc[ppi['string_id'].isin(fused_features['string_id'])].reset_index(drop=True)

    node_data = node_data.rename(columns={'string_id': 'ensembl'})


    for row in file_name['disease_id']:

        if row in ['ICD10_L20', 'ICD10_F90']:
            continue

        print(f'Processing dataset: {row}')
        AUC, recall_5, recall_10, recall_30,recall_100, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, close_neg = lf_main(row, node_data, year)


        results.append({
            'Dataset': row,
            'Train Pos': train_pos,
            'Train Neg': train_neg,
            'Val Pos': val_pos,
            'Val Neg': val_neg,
            'Test Pos': test_pos,
            'Test Neg': test_neg,
            'Close Neg': close_neg,
            'AUC': AUC,
            'R@5' : recall_5,
            'R@10' : recall_10,
            'R@30' : recall_30,
            'R@100' : recall_100,

        })


        results_df = pd.DataFrame(results)
        results_df.to_csv(f"../results/DROCC_LF1/LF1_{feature}_{year}_{args.eval}.csv", index=False)
