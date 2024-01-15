import argparse
import os
# from pathlib import Path
import time
from matplotlib import pyplot as plt
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
import torch
# from torchinfo import summary
from torch import optim, utils
# import wandb
from portiloopml.portiloop_python.ANN.data.mass_data_new import CombinedDataLoader, MassConsecutiveSampler, MassDataset, MassRandomSampler, SubjectLoader
from pytorch_lightning.loggers import WandbLogger
# import plotly.figure_factory as ff
from sklearn.metrics import classification_report
from transformers import ViTImageProcessor, ViTModel
import numpy as np
import pywt
from torchvision.transforms.functional import to_pil_image
import torch.nn as nn


from portiloopml.portiloop_python.ANN.models.lstm import PortiloopNetwork
from portiloopml.portiloop_python.ANN.utils import get_configs, set_seeds
from pytorch_lightning.callbacks import ModelCheckpoint

from portiloopml.portiloop_python.ANN.wamsley_utils import binary_f1_score, get_spindle_onsets


class MassLightning(pl.LightningModule):
    def __init__(self, config, train_choice='both'):
        super().__init__()
        self.config = config
        # Define your model architecture here
        self.model = PortiloopNetwork(config)
        self.spindle_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.staging_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.ss_val_preds = torch.tensor([])
        self.ss_val_labels = torch.tensor([])
        self.spindle_val_preds = torch.tensor([])
        self.spindle_val_labels = torch.tensor([])

        self.ss_testing_preds = torch.tensor([])
        self.ss_testing_labels = torch.tensor([])
        self.spindle_testing_preds = torch.tensor([])
        self.spindle_testing_labels = torch.tensor([])
        self.testing_embeddings = torch.tensor([])

        assert train_choice in ['both', 'spindles', 'staging']
        self.train_choice = train_choice

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

        alpha = self.config['alpha']
        if self.train_choice == 'both':
            loss = alpha * spindle_loss + (1 - alpha) * ss_loss
        elif self.train_choice == 'spindles':
            loss = spindle_loss
        elif self.train_choice == 'staging':
            loss = ss_loss

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        vector = batch[0].to(self.device)
        label_ss = batch[1]['sleep_stage'].to(self.device)
        label_spindles = batch[1]['spindle_label'].to(self.device)
        if hasattr(self, 'val_h'):
            if self.val_h is not None:
                h = self.val_h.to(self.device)
            else:
                h = None
        else:
            h = None

        out_spindles, out_sleep_stages, h, _ = self(vector, h)

        # Convert the outputs and labels to the CPU and detach them from the computation graph
        out_spindles = out_spindles.cpu().detach()
        out_spindles = out_spindles.squeeze(-1)
        out_sleep_stages = out_sleep_stages.cpu().detach()
        label_ss = label_ss.cpu().detach()
        label_spindles = label_spindles.cpu().detach()

        # Get the validation losses
        # remove all columns where the label is 5 (unknown)
        mask = label_ss != 5
        label_ss_4loss = label_ss[mask]
        out_sleep_stages_4loss = out_sleep_stages[mask]
        ss_loss = self.staging_criterion(
            out_sleep_stages_4loss, label_ss_4loss)
        spindle_loss = self.spindle_criterion(out_spindles, label_spindles)

        # Apply softmax and sigmoid to get probabilities
        out_spindles = torch.sigmoid(out_spindles)
        out_sleep_stages = torch.softmax(out_sleep_stages, dim=1)

        self.val_h = h

       # Append the outputs and labels to the class variables
        self.ss_val_preds = torch.cat(
            (self.ss_val_preds, out_sleep_stages.unsqueeze(0)))
        self.ss_val_labels = torch.cat(
            (self.ss_val_labels, label_ss.unsqueeze(0)))
        self.spindle_val_preds = torch.cat(
            (self.spindle_val_preds, out_spindles.unsqueeze(0)))
        self.spindle_val_labels = torch.cat(
            (self.spindle_val_labels, label_spindles.unsqueeze(0)))

        # Log all losses:
        self.log('val_spindle_loss', spindle_loss)
        self.log('val_ss_loss', ss_loss)

        loss = ss_loss + spindle_loss
        self.log('val_loss', loss)

        return loss

    def on_validation_epoch_end(self):
        '''
        Compute all the metrics for the validation epoch
        '''
        # Define what to do at the end of a test epoch
        # We get the probabilities of sleep stage of each batch by averaging the last n batches
        n = min(5, self.ss_val_preds.shape[0])
        averaged_ss_preds = self.ss_val_preds.unfold(0, n, 1).sum(dim=3)
        averaged_ss_preds = torch.softmax(averaged_ss_preds, dim=2)
        averaged_ss_preds = averaged_ss_preds.argmax(dim=2)

        # Now we can flatten the predictions and labels
        averaged_ss_preds = averaged_ss_preds.flatten(start_dim=0, end_dim=1)
        ss_labels = self.ss_val_labels[n-1:].flatten(
            start_dim=0, end_dim=1)
        ss_preds = self.ss_val_preds[n-1:].flatten(
            start_dim=0, end_dim=1)
        ss_preds = torch.argmax(ss_preds, dim=1)

        # We remove all indexes where the label is 5 (unknown)
        mask = ss_labels != 5
        ss_labels = ss_labels[mask]
        ss_preds = ss_preds[mask]
        averaged_ss_preds = averaged_ss_preds[mask]

        # Compute the metrics for sleep staging using sklearn classification report
        report_ss = classification_report(
            ss_labels,
            ss_preds,
            output_dict=True,
        )
        report_avg_ss = classification_report(
            ss_labels,
            averaged_ss_preds,
            output_dict=True,
        )

        # Get the confusion matrix
        cm = confusion_matrix(
            ss_labels,
            ss_preds,
            labels=[0, 1, 2, 3, 4],
        )

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=MassDataset.get_ss_labels()[:-1])
        disp.plot()

        # Log the figure
        self.logger.experiment.log(
            {
                'val_cm_staging': plt,
            },
            commit=False,
        )

        # Create a matplotlib figure for the confusion matrix fro average method
        cm = confusion_matrix(
            ss_labels,
            averaged_ss_preds,
            labels=[0, 1, 2, 3, 4],
        )

        # Log the figure
        self.logger.experiment.log(
            {
                'val_cm_staging_avg': plt,
            },
            commit=False,
        )

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=MassDataset.get_ss_labels()[:-1])
        disp.plot()

        # Log all metrics:
        self.log('val_ss_acc', report_ss['accuracy'])
        self.log('val_ss_f1', report_ss['macro avg']['f1-score'])
        self.log('val_ss_recall', report_ss['macro avg']['recall'])
        self.log('val_ss_precision', report_ss['macro avg']['precision'])

        self.log('val_avg_ss_acc', report_avg_ss['accuracy'])
        self.log('val_avg_ss_f1', report_avg_ss['macro avg']['f1-score'])
        self.log('val_avg_ss_recall', report_avg_ss['macro avg']['recall'])
        self.log('val_avg_ss_precision',
                 report_avg_ss['macro avg']['precision'])

        # Flatten all spindle predictions and labels
        spindle_labels = self.spindle_val_labels.T.flatten(
            start_dim=0, end_dim=1)
        spindle_preds = self.spindle_val_preds.T.flatten(
            start_dim=0, end_dim=1) >= 0.5

        # Add zeros between each prediction to account for the seq_stride
        # Helper method do that cleanly
        def add_stride_back(preds, seq_stride=42):
            total_points = n * len(preds)
            out = torch.zeros(total_points, dtype=preds.dtype)
            # Get the indexes where we need to put the predictions
            idx = torch.arange(0, total_points, n)
            # Put the predictions at the right indexes
            out.index_copy_(0, idx, preds)

            return out

        spindle_preds = add_stride_back(spindle_preds)
        spindle_labels = add_stride_back(spindle_labels)

        # Get all the spindle onsets
        spinlde_onsets_labels = get_spindle_onsets(spindle_labels)
        spindle_onsets_preds = get_spindle_onsets(spindle_preds)

        # Compute the metrics for spindle detection using out binary f1 score
        spindle_precision, spindle_recall, spindle_f1, tp, fp, fn, _ = binary_f1_score(
            spinlde_onsets_labels, spindle_onsets_preds)
        tn = len(spindle_labels) - tp - fp - fn
        cm = np.array([[tn, fp], [fn, tp]])

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['non-spindle', 'spindle'])
        disp.plot()

        # Log the figure
        self.logger.experiment.log(
            {
                'val_cm_spindle': plt,
            },
            commit=False,
        )

        # Log all metrics:
        self.log('val_spindle_acc', (tp + tn) / (tp + tn + fp + fn))
        self.log('val_spindle_f1', spindle_f1)
        self.log('val_spindle_recall', spindle_recall)
        self.log('val_spindle_precision', spindle_precision)

        self.ss_val_preds = torch.tensor([])
        self.ss_val_labels = torch.tensor([])
        self.spindle_val_preds = torch.tensor([])
        self.spindle_val_labels = torch.tensor([])

    def test_step(self, batch, batch_idx):
        # Define the test step here
        vector = batch[0].to(self.device)
        label_ss = batch[1]['sleep_stage'].to(self.device)
        label_spindles = batch[1]['spindle_label'].to(self.device)
        if hasattr(self, 'testing_h'):
            if self.testing_h is not None:
                h = self.testing_h.to(self.device)
            else:
                h = None
        else:
            h = None

        out_spindles, out_sleep_stages, h, embeddings = self(vector, h)

        # Convert the outputs and labels to the CPU and detach them from the computation graph
        out_spindles = out_spindles.cpu().detach()
        out_spindles = out_spindles.squeeze(-1)
        out_spindles = torch.sigmoid(out_spindles)
        out_sleep_stages = out_sleep_stages.cpu().detach()
        # Apply softmax to get probabilities
        out_sleep_stages = torch.softmax(out_sleep_stages, dim=1)
        embeddings = embeddings.cpu().detach()
        label_ss = label_ss.cpu().detach()
        label_spindles = label_spindles.cpu().detach()

        # Append the outputs and labels to the class variables
        self.ss_testing_preds = torch.cat(
            (self.ss_testing_preds, out_sleep_stages.unsqueeze(0)))
        self.ss_testing_labels = torch.cat(
            (self.ss_testing_labels, label_ss.unsqueeze(0)))
        self.spindle_testing_preds = torch.cat(
            (self.spindle_testing_preds, out_spindles.unsqueeze(0)))
        self.spindle_testing_labels = torch.cat(
            (self.spindle_testing_labels, label_spindles.unsqueeze(0)))
        self.testing_embeddings = torch.cat(
            (self.testing_embeddings, embeddings.unsqueeze(0)))

        self.testing_h = h

    def on_test_epoch_end(self):
        # Define what to do at the end of a test epoch
        # We get the probabilities of sleep stage of each batch by averaging the last n batches
        n = 5
        averaged_ss_preds = self.ss_testing_preds.unfold(0, n, 1).sum(dim=3)
        averaged_ss_preds = torch.softmax(averaged_ss_preds, dim=2)
        averaged_ss_preds = averaged_ss_preds.argmax(dim=2)

        # Now we can flatten the predictions and labels
        averaged_ss_preds = averaged_ss_preds.flatten(start_dim=0, end_dim=1)
        ss_labels = self.ss_testing_labels[n-1:].flatten(
            start_dim=0, end_dim=1)
        ss_preds = self.ss_testing_preds[n-1:].flatten(
            start_dim=0, end_dim=1)
        ss_preds = torch.argmax(ss_preds, dim=1)

        # We remove all indexes where the label is 5 (unknown)
        mask = ss_labels != 5
        ss_labels = ss_labels[mask]
        ss_preds = ss_preds[mask]
        averaged_ss_preds = averaged_ss_preds[mask]

        # Compute the metrics for sleep staging using sklearn classification report
        report_ss = classification_report(
            ss_labels,
            ss_preds,
            output_dict=True,
        )
        report_avg_ss = classification_report(
            ss_labels,
            averaged_ss_preds,
            output_dict=True,
        )

        # Get the confusion matrix
        cm = confusion_matrix(
            ss_labels,
            ss_preds,
        )

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=MassDataset.get_ss_labels()[:-1])
        disp.plot()

        # Log the figure
        self.logger.experiment.log(
            {
                'test_cm_staging': plt,
            },
            commit=False,
        )

        # Create a matplotlib figure for the confusion matrix fro average method
        cm = confusion_matrix(
            ss_labels,
            averaged_ss_preds,
        )

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=MassDataset.get_ss_labels()[:-1])
        disp.plot()

        # Log all metrics:
        self.log('test_ss_acc', report_ss['accuracy'])
        self.log('test_ss_f1', report_ss['macro avg']['f1-score'])
        self.log('test_ss_recall', report_ss['macro avg']['recall'])
        self.log('test_ss_precision', report_ss['macro avg']['precision'])

        self.log('test_avg_ss_acc', report_avg_ss['accuracy'])
        self.log('test_avg_ss_f1', report_avg_ss['macro avg']['f1-score'])
        self.log('test_avg_ss_recall', report_avg_ss['macro avg']['recall'])
        self.log('test_avg_ss_precision',
                 report_avg_ss['macro avg']['precision'])

        # Flatten all spindle predictions and labels
        spindle_labels = self.spindle_testing_labels.T.flatten(
            start_dim=0, end_dim=1)
        spindle_preds = self.spindle_testing_preds.T.flatten(
            start_dim=0, end_dim=1) >= 0.5

        # Add zeros between each prediction to account for the seq_stride
        # Helper method do that cleanly
        def add_stride_back(preds, seq_stride=42):
            total_points = n * len(preds)
            out = torch.zeros(total_points, dtype=preds.dtype)
            # Get the indexes where we need to put the predictions
            idx = torch.arange(0, total_points, n)
            # Put the predictions at the right indexes
            out.index_copy_(0, idx, preds)

            return out

        spindle_preds = add_stride_back(spindle_preds)
        spindle_labels = add_stride_back(spindle_labels)

        # Get all the spindle onsets
        spinlde_onsets_labels = get_spindle_onsets(spindle_labels)
        spindle_onsets_preds = get_spindle_onsets(spindle_preds)

        # Compute the metrics for spindle detection using out binary f1 score
        spindle_precision, spindle_recall, spindle_f1, tp, fp, fn, _ = binary_f1_score(
            spinlde_onsets_labels, spindle_onsets_preds)
        tn = len(spindle_labels) - tp - fp - fn
        cm = np.array([[tn, fp], [fn, tp]])

        # Create a matplotlib figure for the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['non-spindle', 'spindle'])
        disp.plot()

        # Log the figure
        self.logger.experiment.log(
            {
                'test_cm_spindle': plt,
            },
            commit=False,
        )

        # Log all metrics:
        self.log('test_spindle_acc', (tp + tn) / (tp + tn + fp + fn))
        self.log('test_spindle_f1', spindle_f1)
        self.log('test_spindle_recall', spindle_recall)
        self.log('test_spindle_precision', spindle_precision)

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


class MassLightningViT(MassLightning):
    def __init__(self, config, train_choice='both'):
        super().__init__(config, train_choice='both')
        self.model = None
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.classifier_ss = nn.Linear(768, 5)
        self.classifier_spindles = nn.Linear(768, 1)
        self.processor = ViTImageProcessor.from_pretrained(
            'google/vit-base-patch16-224')

        self.wavelet = 'morl'
        fs = 250  # Sampling frequency in hz
        freq_num = 224  # Number of frequencies
        frequencies = [i for i in range(1, freq_num + 1)]
        frequencies_norm = np.array(frequencies) / fs  # normalize
        self.scales = pywt.frequency2scale(self.wavelet, frequencies_norm)

    def freeze_up_to(self, layer):
        '''
        Freeze all the layers up to the layer specified
        If the layer is -1, freeze all the layers
        '''
        if layer == -1:
            layer = 100
        for name, param in model.named_parameters():
            if name.split('.')[0] == 'encoder' and int(name.split('.')[2]) < layer:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, x, h):
        # Define the forward pass of your model here

        # Convert the tensor to image format (Wavelet Transform)
        x = self.tensor2img(x)

        # Apply the ViT model
        out = self.vit(x)

        # Extract the embeddings from the ViT model
        embeddings = out.last_hidden_state[:, 0, :]

        # Apply the classifiers
        out_spindles = self.classifier_spindles(embeddings)
        out_sleep_stages = self.classifier_ss(embeddings)

        # Keep this to be consistent with LSTM code
        h = None

        return out_spindles, out_sleep_stages, h, embeddings

    def tensor2img(self, tensors):
        inputs = []
        for tensor in tensors:
            # Apply the inverse wavelet transform
            tensor_wt, _ = pywt.cwt(
                tensor[0, 0, :].cpu().numpy(), self.scales, self.wavelet)
            # Add an axis for the channels
            tensor_wt = np.expand_dims(tensor_wt, axis=-1)
            # Standardize the data to be between 0 and 255
            tensor_wt = (((tensor_wt - tensor_wt.min()) /
                          tensor_wt.max()) * 255).astype(np.uint8)
            # Convert to RGB
            tensor_wt = np.repeat(tensor_wt, 3, axis=-1)
            # Convert to PIL image
            image = to_pil_image(tensor_wt, mode='RGB')

            # Go through the image processor:
            inputs.append(self.processor(
                image, return_tensors='pt')['pixel_values'].to(self.device))

        # Stack all the inputs in one tensor
        inputs = torch.cat(inputs, dim=0)

        return inputs


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
    config['hidden_size'] = 256
    config['nb_rnn_layers'] = 8
    config['lr'] = 1e-5
    config['epoch_length'] = 100000
    config['validation_batch_size'] = 32
    config['segment_len'] = 10000
    config['train_choice'] = 'both'  # One of "both", "spindles", "staging"
    config['use_filtered'] = False
    config['alpha'] = 0.1
    config['useViT'] = True

    if config['useViT']:
        config['batch_size'] = 16
        config['window_size'] = 224
        config['seq_len'] = 1
        config['seq_stride'] = 224
        model = MassLightningViT(config, train_choice=config['train_choice'])
        model.freeze_up_to(-1)
    else:
        model = MassLightning(config, train_choice=config['train_choice'])

    ############### DATA STUFF ##################
    config['num_subjects_train'] = num_train_subjects
    config['num_subjects_val'] = num_val_subjects
    subject_loader = SubjectLoader(
        '/project/MASS/mass_spindles_dataset/subject_info.csv')
    train_subjects = subject_loader.select_random_subjects(
        num_subjects=config['num_subjects_train'], seed=seed)
    val_subjects = subject_loader.select_random_subjects(
        num_subjects=config['num_subjects_val'], seed=seed, exclude=train_subjects)
    test_subjects = subject_loader.select_random_subjects(
        num_subjects=config['num_subjects_val'], seed=seed, exclude=train_subjects + val_subjects)

    train_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=train_subjects,
        window_size=config['window_size'],
        seq_len=config['seq_len'],
        seq_stride=config['seq_stride'],
        use_filtered=config['use_filtered'],
        sampleable='both')

    val_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=val_subjects,
        window_size=config['window_size'],
        seq_len=1,
        seq_stride=config['seq_stride'],
        use_filtered=config['use_filtered'],
        sampleable='both')

    test_dataset = MassDataset(
        '/project/MASS/mass_spindles_dataset',
        subjects=test_subjects,
        window_size=config['window_size'],
        seq_len=1,
        seq_stride=config['seq_stride'],
        use_filtered=config['use_filtered'],
        sampleable='both')

    # Training Combined Dataloader DataLoaders
    train_staging_sampler = MassRandomSampler(
        train_dataset, option='staging_all', num_samples=config['epoch_length'] * config['batch_size'])
    train_staging_loader = utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_staging_sampler)
    train_spindle_sampler = MassRandomSampler(
        train_dataset, option='spindles', num_samples=config['epoch_length'] * config['batch_size'])
    train_spindle_loader = utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_spindle_sampler)
    train_loader = CombinedDataLoader(
        train_staging_loader, train_spindle_loader)

    # Validation DataLoader
    val_sampler = MassConsecutiveSampler(
        val_dataset,
        config['seq_stride'],
        config['segment_len'],
        max_batch_size=config['validation_batch_size'],
    )
    real_batch_size = val_sampler.get_batch_size()
    val_loader = utils.data.DataLoader(
        val_dataset,
        batch_size=real_batch_size,
        sampler=val_sampler)

    # Test DataLoader
    test_sampler = MassConsecutiveSampler(
        test_dataset,
        config['seq_stride'],
        config['segment_len'],
        max_batch_size=config['validation_batch_size'],
    )
    real_batch_size = test_sampler.get_batch_size()
    test_loader = utils.data.DataLoader(
        test_dataset,
        batch_size=real_batch_size,
        sampler=test_sampler)

    config['subjects_train'] = train_subjects
    config['subjects_val'] = val_subjects
    config['subjects_test'] = test_subjects

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
        max_epochs=100,
        accelerator='gpu',
        fast_dev_run=10,
        logger=wandb_logger,
    )

    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # trainer.test(
    #     model, dataloaders=test_loader, verbose=True)
