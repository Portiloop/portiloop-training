import os
import time
import pytorch_lightning as pl
import torch
from torchinfo import summary
from torch import optim, nn, utils, Tensor
from portiloopml.portiloop_python.ANN.data.mass_data_new import MassDataset, MassSampler, SubjectLoader
from pytorch_lightning.loggers import WandbLogger


from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork
from portiloopml.portiloop_python.ANN.utils import get_configs, set_seeds
from pytorch_lightning.callbacks import ModelCheckpoint


class MassLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Define your model architecture here
        self.model = PortiloopNetwork(config)
        self.spindle_criterion = torch.nn.BCELoss()
        self.staging_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x, h):
        # Define the forward pass of your model here
        out_spindles, out_sleep_stages, h, embeddings = self.model(x, h)
        return out_spindles, out_sleep_stages, h, embeddings

    def training_step(self, batch, batch_idx):
        # Define the training step here
        batch_ss = batch
        vector = batch_ss[0].to(self.device)

        # out_spindles, _, _, _ = self(batch_spindle, None)
        # spindle_loss = self.spindle_criterion(out_spindles, batch_spindle)

        _, out_ss, _, _ = self(vector, None)
        ss_label = batch_ss[1]['sleep_stage'].to(self.device)
        ss_loss = self.staging_criterion(out_ss, ss_label)

        # Get the accuracy of sleep staging
        _, ss_pred = torch.max(out_ss, dim=1)
        ss_acc = torch.sum(ss_pred == ss_label).float() / ss_label.shape[0]

        self.log('train_ss_loss', ss_loss)
        self.log('train_ss_acc', ss_acc)

        loss = ss_loss

        return loss

    def validation_step(self, batch, batch_idx):
        # Define the validation step here
        batch_ss = batch
        vector = batch_ss[0].to(self.device)

        # out_spindles, _, _, _ = self(batch_spindle, None)
        # spindle_loss = self.spindle_criterion(out_spindles, batch_spindle)

        _, out_ss, _, _ = self(vector, None)
        ss_label = batch_ss[1]['sleep_stage'].to(self.device)
        ss_loss = self.staging_criterion(out_ss, ss_label)

        # Get the accuracy of sleep staging
        _, ss_pred = torch.max(out_ss, dim=1)
        ss_acc = torch.sum(ss_pred == ss_label).float() / ss_label.shape[0]

        self.log('val_ss_acc', ss_acc)
        self.log('val_ss_loss', ss_loss)

        loss = ss_loss

        return loss

    def configure_optimizers(self):
        # Define your optimizer(s) and learning rate scheduler(s) here
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


if __name__ == "__main__":
    # Ask for the experiment name
    experiment_name = input("Enter the experiment name: ")
    experiment_name = f"{experiment_name}_{time.time()}"

    ################ Model Config ##################
    seed = 42
    set_seeds(seed)
    config = get_configs(experiment_name, True, seed)
    config['hidden_size'] = 64
    config['nb_rnn_layers'] = 3

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
    subject_loader = SubjectLoader(
        '/project/MASS/mass_spindles_dataset/subject_info.csv')
    train_subjects = subject_loader.select_random_subjects(
        num_subjects=31, seed=42)
    val_subjects = subject_loader.select_random_subjects(
        num_subjects=4, seed=42)

    train_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=train_subjects,
        window_size=config['window_size'],
        seq_len=config['seq_len'],
        seq_stride=config['seq_stride'],
        use_filtered=False,
        sampleable='staging')

    val_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=val_subjects,
        window_size=config['window_size'],
        seq_len=config['seq_len'],
        seq_stride=config['seq_stride'],
        use_filtered=False,
        sampleable='staging')

    train_sampler = MassSampler(train_dataset, option='staging_eq')
    val_sampler = MassSampler(val_dataset, option='staging_all')

    train_loader = utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler)

    val_loader = utils.data.DataLoader(
        val_dataset, batch_size=1024, sampler=val_sampler)

    ############### Logger ##################
    os.environ['WANDB_API_KEY'] = "a74040bb77f7705257c1c8d5dc482e06b874c5ce"
    # Add a timestamps to the name
    project_name = "dual_model"
    wandb_logger = WandbLogger(
        project=project_name, config=config, id=experiment_name, log_model="all")

    checkpoint_callback = ModelCheckpoint(
        # dirpath=os.getcwd(),
        # filename=f'{experiment_name}' + '-{epoch:02d}-{f1:.2f}',
        save_top_k=5,
        verbose=True,
        monitor='f1',
        mode='max',
    )

    ############### Trainer ##################
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        fast_dev_run=10,
        logger=wandb_logger,
    )

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
