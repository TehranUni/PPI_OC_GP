import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

from torchvision.utils import make_grid

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import logging
import random
import time
import json



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

"""## base_trainer"""

class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, net: BaseNet,  dataset = '') -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self, net: BaseNet,  dataset = ''):
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass

"""# datasets

## main
"""

def ImportTrainData(file_name, year):

  if year:
    train_data = pd.read_csv(path+f'main_data/train_data/{year}/train/{file_name}.csv')
    test_data = pd.read_csv(path+f'main_data/train_data/{year}/test/{file_name}.csv')
  else:
    train_data = pd.read_csv(path+f'main_data/train_data/{file_name}.csv')

  return train_data, test_data

def read_data(dataset, node_data, year):
  if year:
    disease_train_data, disease_test_data = ImportTrainData(dataset, year)
  else:
    # node_data = pd.read_csv(path+f'main_data/full/ppi_full_emb.csv', sep=",")
    disease_data = ImportTrainData(dataset, '')


  node_data['test'] = 1
  node_data['label'] = 0


  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'label'] = 1 #train disease gene
  node_data.loc[node_data['ensembl'].isin(disease_train_data['ensembl']),'test'] = 0

  node_data.loc[node_data['ensembl'].isin(disease_test_data['ensembl']),'label'] = 1 #test disease gene

  positive_data = node_data.loc[(node_data["label"] == 1) & (node_data["test"] == 0)]
  test_data = node_data.loc[node_data["test"]==1]


  return positive_data, test_data

def create_tensor(data):
  data_tensor = torch.Tensor(data.to_numpy()).type(torch.float32)
  return data_tensor

def create_data_loader(data, label, idx):
  dataset = TensorDataset(data, label, idx)
  dataset_loader = DataLoader(dataset, batch_size=64)
  return dataset_loader

def load_dataset(data, node_data, year):

  train_interest_nodes, test_nodes  = read_data(data, node_data, year)


  y_test = test_nodes['label']
  y_train = train_interest_nodes['label']
  test_data = test_nodes.iloc[:, 1:-2] #ppi
  train_data = train_interest_nodes.iloc[:, 1:-2]


  #-----------------make tensor----------------------
  train = create_tensor(train_data)
  train_label = create_tensor(y_train)
  train_idx = create_tensor(train_data.index)
  train_loader = create_data_loader(train, train_label, train_idx)

  test = create_tensor(test_data)
  test_label = create_tensor(y_test)
  test_idx = create_tensor(test_data.index)
  test_loader = create_data_loader(test, test_label, test_idx)

  return train_loader, test_loader, train_interest_nodes, test_nodes, train_data.shape[1]



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class NNet(BaseNet):

    def __init__(self, size):
        super().__init__()

        self.rep_dim = 100

        self.model = nn.Sequential(
            nn.Linear(size, 150, bias=False),
            nn.ReLU(),
            nn.Linear(150, 100, bias=False),
            nn.Sigmoid()

        )

    def forward(self, x):
      return self.model(x)

class NNet_Autoencoder(BaseNet):
  def __init__(self, size):
    super().__init__()

    self.encoder = nn.Sequential(
            nn.Linear(size, 150, bias=False),
            nn.ReLU(),
            nn.Linear(150, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 70, bias=False),
            nn.ReLU(),
            nn.Linear(70, 50, bias=False),
            nn.ReLU(),
            nn.Linear(50, 100, bias=False),
            nn.ReLU()
        )

    self.decoder = nn.Sequential(
            nn.Linear(100, 50, bias=False),
            nn.ReLU(),
            nn.Linear(50, 70, bias=False),
            nn.ReLU(),
            nn.Linear(70, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 150, bias=False),
            nn.ReLU(),
            nn.Linear(150, size, bias=False),
            nn.ReLU()
        )



  def forward(self, x):
    # Encoding
    encoded = self.encoder(x)
    # Decoding
    decoded = self.decoder(encoded)


    return decoded

"""## main"""

def build_network(net_name, input_size):
    """Builds the neural network."""

    # implemented_networks = ('NNet', 'NNet_Autoencoder','CNN', 'CNN_Autoencoder')
    # assert net_name in implemented_networks

    net = None

    if net_name == 'NNet':
        net = NNet(input_size)

    return net

def build_autoencoder(net_name, input_size):
    """Builds the corresponding autoencoder network."""

    ae_net = None


    if net_name == 'NNet':
        ae_net = NNet_Autoencoder(input_size)


    return ae_net



class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 64, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, ae_net: BaseNet, dataset = ''):

        logger = logging.getLogger()

        train_loader = dataset

        # Set optimizer (Adam optimizer for now)


        # optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        #                        amsgrad=self.optimizer_name == 'amsgrad')
        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, betas=(0.9, 0.999))



        # Set learning rate scheduler
        # milestones=[30,80]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,self.lr_milestones], gamma=0.1)
        # Training
        # logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()

        for epoch in range(self.n_epochs):

            # scheduler.step()

            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()

            for data in train_loader:
                inputs, _ ,_ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()
            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time

            # logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
            #             .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        pretrain_time = time.time() - start_time
        # logger.info('Pretraining time: %.3f' % pretrain_time)
        # logger.info('Finished pretraining.')

        return ae_net

    def test(self, ae_net: BaseNet, dataset=''):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        # _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        test_loader = dataset

        # Testing
        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        # logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        test_time = time.time() - start_time
        logger.info('Autoencoder testing time: %.3f' % test_time)
        logger.info('Finished testing autoencoder.')


class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 2,
                 lr_milestones: tuple = (), batch_size: int = 64, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None



    def train(self, net: BaseNet, dataset = ''):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)


        train_loader = dataset

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,self.lr_milestones], gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        for epoch in range(self.n_epochs):

            # scheduler.step()
            # if epoch in self.lr_milestones:
            #     logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _  = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    scores = dist
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()



                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            scheduler.step()

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))



        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')


        return net

    def test(self, net: BaseNet, dataset = ''):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        test_loader = dataset

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)

                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist


                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))



        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _ , labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0

        c = torch.zeros(net.rep_dim, device=self.device)
        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _= data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)



class DeepSVDD(object):
    """A class for the Deep SVDD method.

    Attributes:
        objective: A string specifying the Deep SVDD objective (either 'one-class' or 'soft-boundary').
        nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
        R: Hypersphere radius R.
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network \phi.
        ae_net: The autoencoder network corresponding to \phi for network weights pretraining.
        trainer: DeepSVDDTrainer to train a Deep SVDD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SVDD network.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
    """

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        """Inits DeepSVDD with one of the two objectives and hyperparameter nu."""

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective
        assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu
        self.R = 0.0  # hypersphere radius R
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network \phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,

        }

    def set_network(self, net_name, input_size):
        """Builds the neural network \phi."""
        self.net_name = net_name
        self.input_size = input_size
        self.net = build_network(net_name, input_size)

    def train(self, dataset =  '' , optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 64, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SVDD model on the training data."""



        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(self.net, dataset)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset = '', device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SVDD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.nu,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(self.net, dataset)
        # Get results
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores


    def pretrain(self, train='', test='', optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 64, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SVDD network \phi via autoencoder."""



        self.ae_net = build_autoencoder(self.net_name, self.input_size)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)

        self.ae_net = self.ae_trainer.train(self.ae_net, train)
        self.ae_trainer.test(self.ae_net, test)
        self.init_network_weights_from_pretraining()


    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SVDD model to export_model."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        """Load Deep SVDD model from model_path."""

        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


def fp_calculate(labels):
  false_positive = -1
  last_one = labels.where(labels== 1).last_valid_index()
  if last_one:
    false_positive = (labels[-last_one:] < 1).sum()

  return false_positive

def thresholding(recon_error,threshold):
    ano_pred=np.zeros(recon_error.shape[0])
    for i in range(recon_error.shape[0]):
        if recon_error[i]>threshold:
            ano_pred[i]=1
    return ano_pred

def compute_topk_metrics(df_sorted, percent, label_col='label'):
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

    if percent == 100:
        top_k_count = percent
    else:
      top_k_count = max(1, int(round((percent / 100.0) * len(df_sorted))))

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

def get_results(trial_score_sorted, anomaly_sampels):
   avg_rank = pd.DataFrame(anomaly_sampels.index).mean()
   AUC = 1-(avg_rank[0]/trial_score_sorted.shape[0])
   return AUC


path = '/content/drive/My Drive/data/'


params = {'dataset_name' : 'ppi',
          'load_model' : None,
          'net_name' : 'NNet',
          'input_size': 128,
          'objective' : 'one-class',
          'nu' : 0.1,
          'device' : 'cuda',
          'seed' : 42,
          'optimizer_name' : 'adam',
          'batch_size' : 32,
          'lr' : 0.0001,
          'n_epochs' : 1000,
          'lr_milestone' : 50,
          'pretrain' : True,
          'weight_decay' : 1e-5,
          'ae_optimizer_name' : 'adam ',
          'ae_lr' :0.001,
          'ae_n_epochs':150,
          'ae_lr_milestone':10,
          'ae_batch_size':64,
          'ae_weight_decay':1e-6,
          'n_jobs_dataloader':0,
          'normal_class':1
          }




def main(data, node_data,year, params):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'


    train_loader, test_loader, train_data, test_data, input_size = load_dataset(data, node_data,year)
    params['input_size'] = input_size

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(params['objective'], params['nu'])
    deep_SVDD.set_network(params['net_name'], params['input_size'])

    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if params['load_model']:
      deep_SVDD.load_model(model_path=params['load_model'], load_ae=True)
      logger.info('Loading model from %s.' % params['load_model'])

    logger.info('Pretraining: %s' % params['pretrain'])

    if params['pretrain']:
      # Log pretraining details

        logger.info('Pretraining optimizer: %s' % params['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % params['ae_lr'])
        logger.info('Pretraining epochs: %d' % params['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (params['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % params['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % params['ae_weight_decay'])
        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(train_loader,
                          test_loader,
                          optimizer_name=params['ae_optimizer_name'],
                          lr=params['ae_lr'],
                          n_epochs=params['ae_n_epochs'],
                          lr_milestones=params['ae_lr_milestone'],
                          batch_size=params['ae_batch_size'],
                          weight_decay=params['ae_weight_decay'],
                          device=device,
                          n_jobs_dataloader=params['n_jobs_dataloader'])

    # Log training details

    logger.info('Training optimizer: %s' % params['optimizer_name'])
    logger.info('Training learning rate: %g' % params['lr'])
    logger.info('Training epochs: %d' % params['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (params['lr_milestone'],))
    logger.info('Training batch size: %d' % params['batch_size'])
    logger.info('Training weight decay: %g' % params['weight_decay'])

    # Train model on dataset
    deep_SVDD.train(train_loader,
                    optimizer_name=params['optimizer_name'],
                    lr=params['lr'],
                    n_epochs=params['n_epochs'],
                    lr_milestones=params['lr_milestone'],
                    batch_size=params['batch_size'],
                    weight_decay=params['weight_decay'],
                    device=device,
                    n_jobs_dataloader=params['n_jobs_dataloader'])

    # Test model

    deep_SVDD.test(test_loader, device=device, n_jobs_dataloader=params['n_jobs_dataloader'])

    # Plot most anomalous and most normal (within-class) test samples
    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score



    scores_df = pd.DataFrame(scores)
    scores_df.columns = ['scores' for col in scores_df.columns]
    test_data.reset_index(inplace=True, drop=True)


    trial_score = pd.concat([test_data,scores_df], axis=1)
    trial_score_sorted = trial_score.sort_values(by=['scores']) #, ascending=False
    trial_score_sorted.reset_index(inplace=True, drop=True)

    results_df = pd.DataFrame({
        'gene_id': trial_score_sorted['ensembl'],
        'y_true': trial_score_sorted['label'],
        'score' : trial_score_sorted['scores']

    })
    results_df.to_csv(path+f"models/DeepSVDD/scores/{data}_scores_{year}.csv", index=False)


    anomaly_sampels = trial_score_sorted.loc[trial_score_sorted['label'] == 1]
    unlabled_sampels = trial_score_sorted.loc[trial_score_sorted['label'] == 0]

    AUC = get_results(trial_score_sorted, anomaly_sampels)
    metrics5 = compute_topk_metrics(trial_score_sorted, percent=5)
    metrics10 = compute_topk_metrics(trial_score_sorted, percent=10)
    metrics30 = compute_topk_metrics(trial_score_sorted, percent=30)
    metrics100 = compute_topk_metrics(trial_score_sorted, percent=100)

    # Save results, model, and configuration
    deep_SVDD.save_results(export_json=path + f'models/DeepSVDD/results/results_{data}_{year}.json')
    deep_SVDD.save_model(export_model=path + f'models/DeepSVDD/model/model_{data}_{year}.tar')
    # Config.save_config(export_json=path + '/main_data/models/DeepSVDD/config/config1_'+data+'.json')

    return train_data.shape[0], test_data.shape[0], anomaly_sampels.shape[0], unlabled_sampels.shape[0], AUC, metrics5['recall'], metrics10['recall'], metrics30['recall'], metrics100['recall']

year = 2017
feature = 'ppi'

file_name = pd.read_csv(path+f"/main_data/train_data/{year}/disease_summary.csv")

ppi = pd.read_csv(path+f'main_data/{year}/ppi_{year}_700_emb.csv', sep=",")
node_data = ppi

node_data = node_data.rename(columns={'string_id': 'ensembl'})


results = []


for row in file_name['disease_id']:


  if params['seed'] != -1:
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    # logger.info('Set seed to %d.' % params['seed'])
  print(f'Processing dataset: {row}')

  train_size, test_size, num_positive_test, num_negative_test, AUC, r5, r10, r30, r100 = main(row, node_data,year, params)


  results.append({
            'Dataset': row,
            'Train Size': train_size,
            'Test Size': test_size,
            'Positive Test': num_positive_test,
            'Negative Test': num_negative_test,
            'AUC': AUC,
            'R@5' : r5,
            'R@10': r10,
            'R@30': r30,
            'R@100': r100

        })


results_df = pd.DataFrame(results)
results_df.to_csv(path+f"/results/DeepSVDD/{feature}_{year}_full_CV.csv", index=False)

print(results_df)

if __name__ == '__main__':
  main(data)