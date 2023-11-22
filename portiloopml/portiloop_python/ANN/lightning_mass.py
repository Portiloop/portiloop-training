import argparse
import os
# from pathlib import Path
import time
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import torch
# from torchinfo import summary
from torch import optim, utils
# import wandb
from portiloopml.portiloop_python.ANN.data.mass_data_new import CombinedDataLoader, MassDataset, MassSampler, SubjectLoader
from pytorch_lightning.loggers import WandbLogger
# import plotly.figure_factory as ff
from sklearn.metrics import classification_report


from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork
from portiloopml.portiloop_python.ANN.utils import get_configs, set_seeds
from pytorch_lightning.callbacks import ModelCheckpoint


class MassLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Define your model architecture here
        self.model = PortiloopNetwork(config)
        self.spindle_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.staging_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.ss_validation_preds = torch.tensor([])
        self.ss_validation_labels = torch.tensor([])

        self.spindle_validation_preds = torch.tensor([])
        self.spindle_validation_labels = torch.tensor([])

        self.save_hyperparameters()

    def forward(self, x, h):
        # Define the forward pass of your model here
        out_spindles, out_sleep_stages, h, embeddings = self.model(x, h)
        return out_spindles, out_sleep_stages, h, embeddings

    def training_step(self, batch, batch_idx):
        # Define the training step here
        batch_ss, batch_spindles = batch

        ######### Do the sleep staging first #########
        vector = batch_ss[0].to(self.device)

        _, out_ss, _, _ = self(vector, None)
        ss_label = batch_ss[1]['sleep_stage'].to(self.device)
        ss_loss = self.staging_criterion(out_ss, ss_label)

        # Get the accuracy of sleep staging
        _, ss_pred = torch.max(out_ss, dim=1)
        ss_acc = torch.sum(ss_pred == ss_label).float() / ss_label.shape[0]

        self.log('train_ss_loss', ss_loss)
        self.log('train_ss_acc', ss_acc)

        ######### Do the spindle detection #########
        vector = batch_spindles[0].to(self.device)

        out_spindles, _, _, _ = self(vector, None)
        out_spindles = out_spindles.squeeze(-1)
        spindle_label = batch_spindles[1]['spindle_label'].to(self.device)
        spindle_loss = self.spindle_criterion(out_spindles, spindle_label)

        # Get the accuracy of spindle detection with threshold 0.5
        out_spindles = torch.sigmoid(out_spindles)
        spindle_pred = (out_spindles > 0.5).float()
        spindle_acc = torch.sum(spindle_pred == spindle_label).float() / \
            spindle_label.shape[0]

        self.log('train_spindle_loss', spindle_loss)
        self.log('train_spindle_acc', spindle_acc)

        loss = spindle_loss + ss_loss

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        batch_ss, batch_spindles = batch

        ######### Do the staging first #########
        vector = batch_ss[0].to(self.device)

        _, out_ss, _, _ = self(vector, None)
        ss_label = batch_ss[1]['sleep_stage'].to(self.device)
        ss_loss = self.staging_criterion(out_ss, ss_label)

        # Get the accuracy of sleep staging
        _, ss_pred = torch.max(out_ss, dim=1)

        self.ss_validation_preds = torch.cat(
            (self.ss_validation_preds, ss_pred.cpu()), dim=0)
        self.ss_validation_labels = torch.cat(
            (self.ss_validation_labels, ss_label.cpu()), dim=0)

        self.log('val_ss_loss', ss_loss)

        ######### Do the spindle detection #########

        vector = batch_spindles[0].to(self.device)

        out_spindles, _, _, _ = self(vector, None)
        out_spindles = out_spindles.squeeze(-1)
        spindle_label = batch_spindles[1]['spindle_label'].to(self.device)
        spindle_loss = self.spindle_criterion(out_spindles, spindle_label)

        # Get the accuracy of spindle detection with threshold 0.5
        out_spindles = torch.sigmoid(out_spindles)
        spindle_pred = (out_spindles > 0.5).float()

        self.spindle_validation_preds = torch.cat(
            (self.spindle_validation_preds, spindle_pred.cpu()), dim=0)
        self.spindle_validation_labels = torch.cat(
            (self.spindle_validation_labels, spindle_label.cpu()), dim=0)

        self.log('val_spindle_loss', spindle_loss)

        loss = ss_loss + spindle_loss
        self.log('val_loss', loss)

        return loss

    def on_validation_epoch_end(self):
        '''
        Compute all the metrics for the validation epoch
        '''
        # Compute the metrics for sleep staging using sklearn classification report
        report_ss = classification_report(
            self.ss_validation_labels,
            self.ss_validation_preds,
            output_dict=True,
        )

        # Get the confusion matrix
        cm = confusion_matrix(
            self.ss_validation_labels,
            self.ss_validation_preds,
        )

        # Get the accuracy of sleep staging
        ss_acc = accuracy_score(
            self.ss_validation_labels,
            self.ss_validation_preds,
        )

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=MassDataset.get_ss_labels()[:-1])
        disp.plot()

        # Log the figure
        self.logger.experiment.log(
            {
                'confusion_matrix_staging': plt,
            },
            commit=False,
        )

        # Log all metrics:
        self.log('val_ss_acc', ss_acc)
        self.log('val_ss_f1', report_ss['macro avg']['f1-score'])
        self.log('val_ss_recall', report_ss['macro avg']['recall'])
        self.log('val_ss_precision', report_ss['macro avg']['precision'])

        # Compute the metrics for spindle detection using sklearn classification report
        report_spindles = classification_report(
            self.spindle_validation_labels,
            self.spindle_validation_preds,
            output_dict=True,
        )

        # Get the confusion matrix
        cm = confusion_matrix(
            self.spindle_validation_labels,
            self.spindle_validation_preds,
        )

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['non-spindle', 'spindle'])

        disp.plot()

        # Log the figure
        self.logger.experiment.log(
            {
                'confusion_matrix_spindle': plt,
            },
            commit=False,
        )

        # Get the accuracy of spindle detection
        spindle_acc = accuracy_score(
            self.spindle_validation_labels,
            self.spindle_validation_preds,
        )

        # Log all metrics:
        self.log('val_spindle_acc', spindle_acc)
        self.log('val_spindle_f1', report_spindles['1.0']['f1-score'])
        self.log('val_spindle_recall', report_spindles['1.0']['recall'])
        self.log('val_spindle_precision', report_spindles['1.0']['precision'])

        # Log the metrics for the whole model
        self.log('val_acc', (ss_acc + spindle_acc) / 2)
        self.log('val_f1', (report_ss['macro avg']['f1-score'] +
                            report_spindles['1.0']['f1-score']) / 2)

        # Reset the validation predictions and labels
        self.ss_validation_preds = torch.tensor([])
        self.ss_validation_labels = torch.tensor([])
        self.spindle_validation_preds = torch.tensor([])
        self.spindle_validation_labels = torch.tensor([])

    def configure_optimizers(self):
        # Define your optimizer(s) and learning rate scheduler(s) here
        optimizer = optim.AdamW(self.parameters(), betas=(
            0.9, 0.99), lr=self.config['lr'], weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, patience=20, factor=0.5, verbose=True, mode='min')
        # scheduler_info = {
        #     'scheduler': scheduler,
        #     'monitor': 'val_loss',  # Metric to monitor for LR scheduling
        #     'interval': 'epoch',  # Adjust LR on epoch end
        #     'frequency': 1  # Check val_loss every epoch
        # }

        return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for random number generator')
    parser.add_argument('--experiment_name', type=str,
                        required=True, help='Name of the experiment')
    parser.add_argument('--num_train_subjects', type=int,
                        required=True, help='Number of subjects for training')
    parser.add_argument('--num_val_subjects', type=int,
                        required=True, help='Number of subjects for validation')

    args = parser.parse_args()

    # Use the command-line arguments
    seed = args.seed
    experiment_name = f"{args.experiment_name}_{str(time.time()).split('.')[0]}"
    num_train_subjects = args.num_train_subjects
    num_val_subjects = args.num_val_subjects
    ################ Model Config ##################
    if seed > 0:
        set_seeds(seed)

    config = get_configs(experiment_name, True, seed)
    config['hidden_size'] = 64
    config['nb_rnn_layers'] = 3
    config['lr'] = 1e-4
    config['epoch_length'] = 1000
    config['validation_batch_size'] = 512

    model = MassLightning(config)
    # input_size = (
    #     config['batch_size'],
    #     config['seq_len'],
    #     1,
    #     config['window_size'])

    # x = torch.zeros(input_size)
    # h = torch.zeros(config['nb_rnn_layers'],
    #                 config['batch_size'], config['hidden_size'])
    # out_spindles, out_sleep_stages, h, embeddings = model(x, h)

    ############### DATA STUFF ##################
    config['num_subjects_train'] = num_train_subjects
    config['num_subjects_val'] = num_val_subjects
    subject_loader = SubjectLoader(
        '/project/MASS/mass_spindles_dataset/subject_info.csv')
    train_subjects = subject_loader.select_random_subjects(
        num_subjects=config['num_subjects_train'], seed=seed)
    val_subjects = subject_loader.select_random_subjects(
        num_subjects=config['num_subjects_val'], seed=seed)

    train_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=train_subjects,
        window_size=config['window_size'],
        seq_len=config['seq_len'],
        seq_stride=config['seq_stride'],
        use_filtered=False,
        sampleable='both')

    val_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=val_subjects,
        window_size=config['window_size'],
        seq_len=config['seq_len'],
        seq_stride=config['seq_stride'],
        use_filtered=False,
        sampleable='both')

    train_staging_sampler = MassSampler(
        train_dataset, option='staging_eq', num_samples=config['epoch_length'] * config['batch_size'])
    val_staging_sampler = MassSampler(
        val_dataset, option='staging_all', num_samples=config['epoch_length'] * config['validation_batch_size'])

    train_staging_loader = utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_staging_sampler)

    val_staging_loader = utils.data.DataLoader(
        val_dataset, batch_size=config['validation_batch_size'], sampler=val_staging_sampler)

    train_spindle_sampler = MassSampler(
        train_dataset, option='spindles', num_samples=config['epoch_length'] * config['batch_size'])
    val_spindle_sampler = MassSampler(
        val_dataset, option='random', num_samples=config['epoch_length'] * config['validation_batch_size'])

    train_spindle_loader = utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_spindle_sampler)
    val_spindle_loader = utils.data.DataLoader(
        val_dataset, batch_size=config['validation_batch_size'], sampler=val_spindle_sampler)

    train_loader = CombinedDataLoader(
        train_staging_loader, train_spindle_loader)
    val_loader = CombinedDataLoader(val_staging_loader, val_spindle_loader)
    config['subjects_train'] = train_subjects
    config['subjects_val'] = val_subjects

    ############### Logger ##################
    os.environ['WANDB_API_KEY'] = "a74040bb77f7705257c1c8d5dc482e06b874c5ce"
    # Add a timestamps to the name
    project_name = "dual_model"
    group = 'Training'
    wandb_logger = WandbLogger(
        project=project_name,
        group=group,
        config=config,
        id=experiment_name,
        log_model="all")

    wandb_logger.watch(model, log="all")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        verbose=True,
        monitor='val_f1',
        mode='max',
    )

    ############### Trainer ##################
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator='gpu',
        fast_dev_run=10,
        logger=wandb_logger,
    )

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
