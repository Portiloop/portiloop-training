import os
from pathlib import Path
import torch
import wandb

from portiloopml.portiloop_python.ANN.lightning_mass import MassLightning
import pytorch_lightning as pl
from torch import optim, utils
from portiloopml.portiloop_python.ANN.utils import get_configs, set_seeds
from portiloopml.portiloop_python.ANN.data.mass_data_new import (
    CombinedDataLoader, MassConsecutiveSampler, MassDataset, MassRandomSampler,
    SubjectLoader)


def load_model(checkpoint_ref, project, group, run_id):

    # download checkpoint locally (if not already cached)
    run = wandb.init(
        project=project,
        group=group,
        name=run_id,)
    artifact = run.use_artifact(checkpoint_ref, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    model = MassLightning.load_from_checkpoint(
        Path(artifact_dir) / "model.ckpt")
    return model, run


def load_model_mass(new_run_name, run_id, group_name=None):
    # Log in with our wandb id
    os.environ['WANDB_API_KEY'] = "a74040bb77f7705257c1c8d5dc482e06b874c5ce"

    # Get checkpoint reference
    user = "milosobral"
    project = "dual_model"
    artifact_name = "best"
    group = "Adapt_cc_1" if group_name is None else group_name
    run_id_val = new_run_name
    checkpoint_ref = f"{user}/{project}/model-{run_id}:{artifact_name}"

    # Load model
    model, run = load_model(checkpoint_ref, project, group, run_id_val)
    return model, run


if __name__ == "__main__":

    run_id_new = 'both_cc_limited_ss_44055'
    run_id_old = "both_cc_smallLR_1706210166"

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # model_new, _ = load_model_mass("Validating", run_id_new)
    # print(f"MODEL NEW: {count_parameters(model_new)}")

    model_old, _ = load_model_mass("Validating", run_id_old)
    print(f"MODEL OLD: {count_parameters(model_old)}")

    # seed = 42
    # experiment_name = "mass_val"

    # config = get_configs(experiment_name, True, seed)
    # config['hidden_size'] = 256
    # config['nb_rnn_layers'] = 8
    # config['lr'] = 1e-4
    # config['adamw_weight_decay'] = 0.001
    # config['epoch_length'] = -1
    # config['validation_batch_size'] = 512
    # config['segment_len'] = 1000
    # config['train_choice'] = 'spindles'  # One of "both", "spindles", "staging"
    # config['use_filtered'] = False
    # config['alpha'] = 0.1
    # config['useViT'] = False
    # config['dropout'] = 0.5
    # config['batch_size'] = 64
    # config['window_size'] = 54
    # config['seq_stride'] = 42
    # config['seq_len'] = 50
    # config['num_subjects_val'] = 2

    # dataset_path = '/project/MASS/mass_spindles_dataset/'

    # model, _ = load_model_mass("Validating")

    # subject_loader = SubjectLoader(
    #     os.path.join(dataset_path, 'subject_info.csv'))

    # val_subjects = subject_loader.select_subjects_age(
    #     min_age=0,
    #     max_age=40,
    #     num_subjects=config['num_subjects_val'] // 2,
    #     seed=seed)
    # val_subjects += subject_loader.select_subjects_age(
    #     min_age=40,
    #     max_age=100,
    #     num_subjects=config['num_subjects_val'] // 2,
    #     seed=seed,
    #     exclude=val_subjects)

    # val_dataset = MassDataset(
    #     dataset_path,
    #     subjects=val_subjects,
    #     window_size=config['window_size'],
    #     seq_len=1,
    #     seq_stride=config['seq_stride'],
    #     use_filtered=config['use_filtered'],
    #     sampleable='both')

    # val_sampler = MassConsecutiveSampler(
    #     val_dataset,
    #     config['seq_stride'],
    #     config['segment_len'],
    #     max_batch_size=config['validation_batch_size'],
    # )
    # real_batch_size = val_sampler.get_batch_size()
    # val_loader = utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=real_batch_size,
    #     sampler=val_sampler)

    # trainer = pl.Trainer(
    #     max_epochs=10,
    #     accelerator='gpu',
    #     # fast_dev_run=10,
    # )

    # trainer.validate(
    #     model,
    #     val_loader
    # )
    # print("Validation done")
