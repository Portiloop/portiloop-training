"""
Model Loading Utility for MassLightning using Weights & Biases (W&B)

This script provides functionality to download and load a trained MassLightning model
from the W&B artifact store.
"""

import os
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run

from portiloopml.portiloop_python.ANN.lightning_mass import MassLightning


def load_model(checkpoint_ref:str, project:str, group:str, run_id:str)->tuple[MassLightning, Run]:
    """
    Downloads a model artifact from Weights & Biases and loads it as a MassLightning model.

    Args:
        checkpoint_ref (str): Full artifact reference (e.g., 'user/project/model-run_id:best').
        project (str): The W&B project name.
        group (str): The W&B group name.
        run_id (str): The name to assign to the local W&B run used for downloading.

    Returns:
        Tuple[MassLightning, wandb.Run]: A tuple containing the loaded model and the associated W&B run object.
    """

    # download checkpoint locally (if not already cached)
    wandb.login(key=os.getenv('WANDB_API_KEY'))
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


def load_model_mass(new_run_name:str, run_id:str, group_name:str=None)->tuple[MassLightning,Run]:
    """
    Loads a trained MassLightning model from Weights & Biases using a given run ID.

    Args:
        new_run_name (str): The name to assign to the new run instance.
        run_id (str): The identifier of the run to load the model from.
        group_name (str, optional): The name of the group to associate with the run. Defaults to "Adapt_cc_1".

    Returns:
        tuple[MassLightning, Run]: A tuple containing the loaded model and the associated wandb Run object.
    """

    # Get checkpoint reference
    user = os.getenv('WANDB_USERNAME')
    project = os.getenv('WANDB_PROJECT')
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

    model_old, _ = load_model_mass("Validating", run_id_old)
    print(f"MODEL OLD: {count_parameters(model_old)}")
