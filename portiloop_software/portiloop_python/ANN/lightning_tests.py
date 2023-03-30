from pathlib import Path
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from portiloop_software.portiloop_python.ANN.data.mass_data import MassDataset, MassRandomSampler, SingleSubjectDataset, SingleSubjectSampler, read_pretraining_dataset, read_spindle_trains_labels
from portiloop_software.portiloop_python.ANN.data.moda_data import RandomSampler, SignalDataset, ValidationSampler, get_class_idxs
from portiloop_software.portiloop_python.ANN.models.lstm import PortiloopNetwork
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from portiloop_software.portiloop_python.ANN.utils import LoggerWandb, get_configs


class LstmModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__() # init the parent class
        self.config = config
        self.model = PortiloopNetwork(config)
        self.loss = nn.BCELoss(reduction='none')

        # Initialize arrays to keep predictions for validation
        self.predictions = []
        self.targets = []

    def forward(self, x, h=None):
        return self.model(x, h)
    
    def training_step(self, batch, batch_idx):  
        # Load the batch from the dataloader
        x, _, _, y = batch
        y = y.float()

        # Forward pass
        h1 = torch.zeros((self.config['nb_rnn_layers'], self.config['batch_size'], self.config['hidden_size']), device=x.device)
        y_hat, h, _ = self.model(x, h1)
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y)
        loss = loss.mean()
        predictions = (y_hat >= 0.5)
        acc = (predictions == y).float().mean()
        self.log('train_loss', loss.item())
        self.log('train_acc', acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Load the batch from the dataloader
        x, _, _, y = batch
        x = x.float()
        y = y.float()

        # assert  in [0, 1], f"y should be 0 or 1, but is {y}"

        # Forward pass
        if batch_idx == 0:
            self.h1 = torch.zeros((self.config['nb_rnn_layers'], self.config['batch_size_validation'], self.config['hidden_size']), device=x.device)
        y_hat, self.h1, _ = self.model(x, self.h1)
        y_hat = y_hat.view(-1)
        loss = self.loss(y_hat, y).mean()

        predictions = (y_hat >= 0.5)
        # acc = (predictions == y).float().mean()
        # f1 = f1_score(y.cpu().numpy(), predictions.cpu().numpy())
        self.log('val_loss', loss.item())

        # Save the results
        self.predictions.append(predictions)
        self.targets.append(y)
        return loss
    
    def on_validation_epoch_end(self):
        predictions = torch.cat(self.predictions)
        targets = torch.cat(self.targets)
        acc = (predictions == targets).float().mean()
        f1 = f1_score(targets.cpu().numpy(), predictions.cpu().numpy())
        self.log('val_acc', acc)
        self.log('val_f1', f1)

        self.predictions.clear()
        self.targets.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr_adam'], weight_decay=self.config['adam_w'])
        return optimizer
    

class MODADataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.len_segment = config['len_segment_s'] * config['fe']
        self.filename = config['filename_classification_dataset']

        self.all_subject = pd.read_csv(Path(self.config['path_dataset']) / self.config['subject_list'], header=None, delim_whitespace=True).to_numpy()
        p1_subject = pd.read_csv(Path(self.config['path_dataset']) / self.config['subject_list_p1'], header=None, delim_whitespace=True).to_numpy()
        p2_subject = pd.read_csv(Path(self.config['path_dataset']) / self.config['subject_list_p2'], header=None, delim_whitespace=True).to_numpy()
        train_subject_p1, validation_subject_p1 = train_test_split(p1_subject, train_size=0.8, random_state=self.config['split_idx'])
        if config['test_set']:
            test_subject_p1, validation_subject_p1 = train_test_split(validation_subject_p1, train_size=0.5, random_state=config['split_idx'])
        
        train_subject_p2, validation_subject_p2 = train_test_split(p2_subject, train_size=0.8, random_state=self.config['split_idx'])
        if config['test_set']:
            test_subject_p2, validation_subject_p2 = train_test_split(validation_subject_p2, train_size=0.5, random_state=config['split_idx'])
        self.train_subject = np.array([s for s in self.all_subject if s[0] in train_subject_p1[:, 0] or s[0] in train_subject_p2[:, 0]]).squeeze()
        if config['test_set']:
            self.test_subject = np.array([s for s in self.all_subject if s[0] in test_subject_p1[:, 0] or s[0] in test_subject_p2[:, 0]]).squeeze()
        self.validation_subject = np.array(
            [s for s in self.all_subject if s[0] in validation_subject_p1[:, 0] or s[0] in validation_subject_p2[:, 0]]).squeeze()
        
        self.nb_segment_validation = len(np.hstack([range(int(s[1]), int(s[2])) for s in self.validation_subject]))
        self.batch_size_validation = len(list(range(0, (config['seq_stride'] // config['validation_network_stride']) * config['validation_network_stride'], config['validation_network_stride']))) * self.nb_segment_validation

    def train_dataloader(self):
        ds_train = SignalDataset(filename=self.filename,
                                 path=config['path_dataset'],
                                 window_size=config['window_size'],
                                 fe=config['fe'],
                                 seq_len=config['seq_len'],
                                 seq_stride=config['seq_stride'],
                                 list_subject=self.train_subject,
                                 len_segment=self.len_segment,
                                 threshold=config['threshold'])

        
        idx_true, idx_false = get_class_idxs(ds_train, config['distribution_mode'])
        samp_train = RandomSampler(idx_true=idx_true,
                                   idx_false=idx_false,
                                   batch_size=config['batch_size'],
                                   nb_batch=config['nb_batch_per_epoch'],
                                   distribution_mode=config['distribution_mode'])

        
        train_loader = DataLoader(ds_train,
                                  batch_size=config['batch_size'],
                                  sampler=samp_train,
                                  shuffle=False,
                                  num_workers=0,
                                  pin_memory=True)        
        
        return train_loader
    
    def val_dataloader(self):
        ds_validation = SignalDataset(filename=self.filename,
                                      path=config['path_dataset'],
                                      window_size=config['window_size'],
                                      fe=config['fe'],
                                      seq_len=1,
                                      seq_stride=1,  # just to be sure, fixed value
                                      list_subject=self.validation_subject,
                                      len_segment=self.len_segment,
                                      threshold=config['threshold'])
        
        samp_validation = ValidationSampler(ds_validation,
                                            seq_stride=config['seq_stride'],
                                            len_segment=self.len_segment,
                                            nb_segment=self.nb_segment_validation,
                                            network_stride=config['validation_network_stride'])
        validation_loader = DataLoader(ds_validation,
                                       batch_size=self.batch_size_validation,
                                       sampler=samp_validation,
                                       num_workers=0,
                                       pin_memory=True,
                                       shuffle=False)
        return validation_loader

class MASSDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # Read all the subjects available in the dataset
        self.labels = read_spindle_trains_labels(self.config['old_dataset']) 

        # Divide the subjects into train and test sets
        self.subjects = list(self.labels.keys())
        random.shuffle(self.subjects)

        # Read the pretraining dataset
        self.data = read_pretraining_dataset(self.config['MASS_dir'], patients_to_keep=self.subjects)

    def setup(self, stage=None):
        # called on every GPU
        # split train/val/test
        # make assignments here (val/train/test split)
        self.validation_subject = self.subjects[0]
        self.train_subjects = self.subjects[1:]

    def train_dataloader(self):
        # Create the train and test datasets
        train_dataset = MassDataset(self.train_subjects, self.data, self.labels, self.config)
        # Create the train dataloader
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=MassRandomSampler(train_dataset, self.config['batch_size'], nb_batch=self.config['nb_batch_per_epoch']),
            pin_memory=True,
            drop_last=True,
            num_workers=4
        )
        return train_dataloader

    def val_dataloader(self):
        test_dataset = SingleSubjectDataset(self.validation_subject, self.data, self.labels, self.config)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            sampler=SingleSubjectSampler(len(test_dataset), self.config['seq_stride']),
            pin_memory=True,
            drop_last=True,
        )
        return test_dataloader


if __name__ == "__main__":
    config = get_configs("Test", True, 0)
    experiment_name = "Test"
    wandb_project = f"full-dataset-public"
    wandb_group = f"default"
    logger = WandbLogger(
        project=wandb_project,
        name=experiment_name,
    )

    data = MODADataModule(config)
    config['batch_size_validation'] = data.batch_size_validation
    model = LstmModel(config)
    trainer = pl.Trainer(
        max_epochs=30,
        logger=logger)
    trainer.fit(model, data)

