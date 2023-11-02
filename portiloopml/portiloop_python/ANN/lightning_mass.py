import pytorch_lightning as pl
import torch
from torchinfo import summary

from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork
from portiloopml.portiloop_python.ANN.utils import get_configs, set_seeds

class MyLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Define your model architecture here
        self.model = PortiloopNetwork(config)
        self.spindle_criterion = torch.nn.BCELoss()

    def forward(self, x, h):
        # Define the forward pass of your model here
        out_spindles, out_sleep_stages, h, embeddings = self.model(x, h)
        return out_spindles, out_sleep_stages, h, embeddings

    def training_step(self, batch, batch_idx):
        # Define the training step here
        batch_spindle, batch_ss = batch

        out_spindles, _, _, _ = self(batch_spindle, None)
        spindle_loss = self.spindle_criterion(out_spindles, batch_spindle)

        _, out_ss, _, _ = self(batch_ss, None)
        ss_loss = self.spindle_criterion(out_ss, batch_ss)

        loss = ...
        # TODO
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # Define the validation step here
        val_loss = ...
        return {'val_loss': val_loss}

    def test_step(self, batch, batch_idx):
        # Define the test step here
        test_loss = ...
        return {'test_loss': test_loss}

    def configure_optimizers(self):
        # Define your optimizer(s) and learning rate scheduler(s) here
        optimizer = ...
        lr_scheduler = ...
        return [optimizer], [lr_scheduler]
    
if __name__ == "__main__":
    experiment_name = "DEBUG"
    seed = 42
    set_seeds(seed)
    config = get_configs(experiment_name, True, seed)
    config['hidden_size'] = 64
    config['nb_rnn_layers'] = 3

    model = MyLightningModule(config)
    input_size = (
        config['batch_size'], 
        config['seq_len'], 
        1,
        config['window_size'])
    
    x = torch.zeros(input_size)
    h = torch.zeros(config['nb_rnn_layers'], config['batch_size'], config['hidden_size'])
    out_spindles, out_sleep_stages, h, embeddings = model(x, h)
    print(out_spindles.shape)
    print(out_sleep_stages.shape)
    print(h.shape)
    print(embeddings.shape)


