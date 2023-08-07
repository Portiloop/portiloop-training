import os
import random
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Sampler

import wandb
from portiloop_software.portiloop_python.ANN.data.mass_data import (
    read_pretraining_dataset, read_sleep_staging_labels,
    read_spindle_trains_labels)
from portiloop_software.portiloop_python.ANN.data.sleepedf_data import get_sleepedf_loaders
from portiloop_software.portiloop_python.ANN.models.sleep_staging_models import (
    CNNBlock, TSNConv, TransformerEncoderWithCLS)
from portiloop_software.portiloop_python.ANN.models.test_dsn import DeepSleepNet, TinySleepNet
from portiloop_software.portiloop_python.ANN.utils import (get_configs,
                                                           set_seeds)
from pytorch_lightning.callbacks import ModelCheckpoint


class SleepStageDataset(Dataset):
    def __init__(self, subjects, data, labels, seq_len, window_size, freq):
        '''
        This class takes in a list of subject, a path to the MASS directory 
        and reads the files associated with the given subjects as well as the sleep stage annotations
        '''
        super().__init__()

        self.seq_len = seq_len
        self.window_size = window_size

        # Get the sleep stage labels
        self.full_signal = []
        self.full_labels = []

        self.subject_list = []
        for subject in subjects:
            if subject not in data.keys():
                print(
                    f"Subject {subject} not found in the pretraining dataset")
                continue

            # Get the signal for the given subject
            signal = torch.tensor(data[subject]['signal'], dtype=torch.float)

            # Get all the labels for the given subject
            label = torch.tensor([SleepStageDataset.get_labels().index(
                lab) for lab in labels[subject]]).type(torch.uint8)

            # Repeat the labels freq times to match the signal using a pytorch function
            label = torch.tensor(label).repeat_interleave(freq)

            # Add some '?' padding at the end to make sure the length of signal and label match
            missing = len(signal) - len(label)
            label = torch.cat([label, torch.full(
                (missing, ), SleepStageDataset.get_labels().index('?')).type(torch.uint8)])

            # Make sure that the signal and the labels are the same length
            assert len(signal) == len(label)

            # Add to full signal and full label
            self.full_labels.append(label)
            self.full_signal.append(signal)
            del data[subject], signal, label

        self.full_signal = torch.cat(self.full_signal)
        self.full_labels = torch.cat(self.full_labels)

        # Make a dictionary of all the indexes of each class label
        self.label_indexes = {}
        for label in self.full_labels.unique():
            self.label_indexes[int(label)] = (
                self.full_labels == label).nonzero(as_tuple=False).reshape(-1)

        total_samp_labels = sum([len(self.label_indexes[i]) for i in range(5)])
        self.sampleable_weights = [
            len(self.label_indexes[i]) / total_samp_labels for i in range(5)]

    @staticmethod
    def get_labels():
        return ['1', '2', '3', 'R', 'W', '?']

    def __getitem__(self, index):
        # Get data and label at the given index
        signal = self.full_signal[index -
                                  (self.seq_len * self.window_size):index]
        signal = signal.unfold(0, self.window_size, self.window_size)
        signal = signal.unsqueeze(1)
        label = self.full_labels[index]

        return signal, label.type(torch.LongTensor)

    def __len__(self):
        return len(self.full_signal)


class SleepStagingModel(pl.LightningModule):
    def __init__(self, config, weights):
        super().__init__()

        # encoder = CNNBlock(1, config['embedding_size'])
        # encoder = TSNConv(config['freq'])

        # self.transformer = TransformerEncoderWithCLS(
        #     encoder,
        #     config['embedding_size'],
        #     config['num_heads'],
        #     config['num_layers'],
        #     5,
        #     dropout=config['dropout'],
        #     cls=config['cls'])

        self.transformer = TinySleepNet(fs=config['freq'])

        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.validation_outputs = []
        self.validation_labels = []
        self.config = config

    def forward(self, x):
        return self.transformer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss.mean(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        accuracy = y_hat.argmax(dim=1).eq(y).sum().float() / y.size(0)
        self.log("train_acc", accuracy, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        prediction = y_hat.argmax(dim=1)
        accuracy = prediction.eq(y).sum().float() / y.size(0)
        self.log("val_acc", accuracy, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.validation_outputs.append(prediction.cpu())
        self.validation_labels.append(y.cpu())
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.stack(self.validation_outputs).flatten()
        y_true = torch.stack(self.validation_labels).flatten()

        # conf_mat = confusion_matrix(y_true, y_pred)
        class_report = classification_report(
            y_true, y_pred, target_names=SleepStageDataset.get_labels()[:-1], labels=[0, 1, 2, 3, 4], output_dict=True)

        self.log('f1', class_report['macro avg']['f1-score'], logger=True)
        self.log(
            'precision', class_report['macro avg']['precision'], logger=True)
        self.log('recall', class_report['macro avg']['recall'], logger=True)

        self.validation_outputs.clear()
        self.validation_labels.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config['lr'], betas=(0.9, 0.999), weight_decay=1e-3)


if __name__ == "__main__":
    # Wandb stuff
    os.environ['WANDB_API_KEY'] = "a74040bb77f7705257c1c8d5dc482e06b874c5ce"
    seed = 42
    set_seeds(seed)
    project_name = "sleep_staging_portiloop"

    # Get the config
    config = {
        'batch_size': 64,
        'freq': 100,
        'inception': [16, 8, 16, 16, 32, 16],
        'lr': 1e-3,
        'num_heads': 8,
        'num_layers': 1,
        'noise_std': 0.1,
        'dropout': 0.1,
        'cls': False,
        'window_size': 30 * 100,
        'seq_len': 50,
    }

    config['embedding_size'] = 128

    # Load the data
    unfiltered_mass = "/project/portiloop_transformer/transformiloop/dataset/MASS_preds/"
    path_dataset = "/project/portiloop-training/portiloop_software/dataset"

    # ss_labels = read_sleep_staging_labels(path_dataset)
    # # Divide subjects between test and validation
    # max_subjects = -1
    # subjects = list(ss_labels.keys()) if max_subjects == \
    #     -1 else list(ss_labels.keys())[:max_subjects]

    # random.shuffle(subjects)
    # cutoff = int(len(subjects) * 0.8)
    # test_subjects = subjects[:cutoff]
    # val_subjects = subjects[cutoff:]

    # data = read_pretraining_dataset(unfiltered_mass, patients_to_keep=subjects)

    # dataset = SleepStageDataset(
    #     test_subjects, data, ss_labels, config['seq_len'], config['window_size'], config['freq'])
    # test_dataset = SleepStageDataset(
    #     val_subjects, data, ss_labels, config['seq_len'], config['window_size'], config['freq'])
    # sampler = SSValidationSampler(
    #     dataset, 1000, config['batch_size'])
    # test_sampler = SSValidationSampler(
    #     test_dataset, 1000, config['batch_size'])

    # loader = DataLoader(
    #     dataset=dataset,
    #     batch_size=config['batch_size'],
    #     sampler=sampler,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=config['batch_size'],
    #     sampler=test_sampler,
    #     num_workers=0,
    #     pin_memory=True,
    #     drop_last=True
    # )

    print("Loading data...")
    train_loader, test_loader, loss_weights = get_sleepedf_loaders(82, config)
    print("done...")

    # Add a timestamps to the name
    experiment_name = f"TSN_again_{int(time.time())}"

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(),
        filename=f'{experiment_name}' + '-{epoch:02d}-{f1:.2f}',
        save_top_k=5,
        verbose=True,
        monitor='f1',
        mode='max',
    )

    wandb_logger = WandbLogger(
        project=project_name, config=config, id=experiment_name)
    model = SleepStagingModel(config, loss_weights)
    trainer = pl.Trainer(max_epochs=100, accelerator='gpu',
                         logger=wandb_logger, callbacks=[checkpoint_callback])  # , fast_dev_run=10

    trainer.fit(model, train_loader, test_loader)
