import os
from pathlib import Path
import wandb

from portiloopml.portiloop_python.ANN.lightning_mass import MassLightning


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


def validate_model(dataloader, model, device):
    '''
    Run through the validation set and return the metrics
    '''
    # Run through the validation set
    for batch in dataloader:
        # Get the data
        vector = batch[0].to(device)


if __name__ == "__main__":
    # Log in with our wandb id
    os.environ['WANDB_API_KEY'] = "a74040bb77f7705257c1c8d5dc482e06b874c5ce"

    # Get checkpoint reference
    user = "milosobral"
    project = "dual_model"
    run_id = "testing_cc_1700617530"
    artifact_name = "best"
    group = "Validation"
    run_id_val = run_id + "_val"
    checkpoint_ref = f"{user}/{project}/model-{run_id}:{artifact_name}"

    # Load model
    model, run = load_model(checkpoint_ref, project, group, run_id_val)
