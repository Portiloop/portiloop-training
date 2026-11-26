#!/usr/bin/env python3
"""
Train your own variant of the Portiloop spindle detector.

Companion script of the tutorial "Training a new model" (docs/tutorials/05-training.html).
It imports the training module, copies the configuration of the published model,
changes a few hyperparameters and calls `run()`, which is exactly what the
command-line entry point of portiloop_detector_training.py does, but with every
choice visible and editable.

Run from the repository root, after `pip install -e .` and after placing the dataset
files in portiloop_software/dataset (tutorial 02):

    # 1. Check the configuration only (no dataset, no GPU, no W&B needed):
    python docs/tools/train_custom.py --check

    # 2. Real training on the first GPU, with W&B logging (or WANDB_MODE=offline):
    python docs/tools/train_custom.py --name my_variant_h10 --hidden-size 10

    # 3. A quick CPU smoke test (a few minutes, the model will not be good):
    WANDB_MODE=offline python docs/tools/train_custom.py --device cpu --epochs 2 --batches-per-epoch 20

Everything written by the training (checkpoints and .onnx files) goes to the dataset
folder (portiloop_software/dataset by default, see --dataset-dir).
"""
import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # also works without `pip install -e .`

from portiloop_software.portiloop_python.ANN import portiloop_detector_training as pdt  # noqa: E402

# Thresholds used by the command-line entry point (the `if __name__ == "__main__"` block).
# Labels strictly greater than the threshold count as "spindle".
CLI_THRESHOLDS = {"p1": 0.2, "p2": 0.35, "full": 0.2}


def set_phase(phase="full", test_set=True, threshold=None):
    """Set the module-level globals that the command line would set.

    When portiloop_detector_training is IMPORTED (as here) rather than run as a script,
    its globals come from the `else:` branch at the bottom of the file:
    PHASE = 'full', TEST_SET = True, THRESHOLD = 0.5, and dataset file names built for
    phase 'full'. The command line uses THRESHOLD = 0.2 for phase 'full'.
    For the classification dataset (labels exactly 0 or 1) any threshold between 0 and 1
    gives the same labels, so this only matters for regression (graded labels).
    """
    pdt.PHASE = phase
    pdt.TEST_SET = test_set
    pdt.THRESHOLD = CLI_THRESHOLDS[phase] if threshold is None else threshold
    pdt.filename_classification_dataset = f"dataset_classification_{phase}_big_250_matlab_standardized_envelope_pf.txt"
    pdt.filename_regression_dataset = f"dataset_regression_{phase}_big_250_matlab_standardized_envelope_pf.txt"
    pdt.subject_list = f"subject_sequence_{phase}_big.txt"


def make_config(args):
    """Start from the published model and change only what we want to study."""
    config = pdt.get_final_model_config_dict(index=0, split_i=args.split)
    config["experiment_name"] = args.name
    config["hidden_size"] = args.hidden_size        # size of the GRU memory (published: 7)
    config["nb_channel"] = args.nb_channel          # convolution filters per layer (published: 31)
    config["device_train"] = args.device            # 'cuda', 'cuda:1', or 'cpu'
    config["device_val"] = args.device
    config["nb_epoch_max"] = args.epochs            # published: 150
    config["nb_batch_per_epoch"] = args.batches_per_epoch  # published: 1000
    config["lr_adam"] = args.lr                     # published: 0.0005
    config["classification"] = not args.regression
    config["distribution_mode"] = 1 if args.regression else 0  # regression requires 1 (no class balancing)
    config["reg_balancing"] = args.reg_balancing    # regression only: 'none' (published), 'lds' or 'sr'
    return config


def describe(config):
    """Print what the configuration means in samples and seconds, and test one forward pass."""
    fe = config["fe"]
    window = int(config["window_size_s"] * fe)
    stride = int(config["seq_stride_s"] * fe)
    net = pdt.PortiloopNetwork(config)
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.zeros(2, config["seq_len"], window)          # (batch, sequence, samples)
    h = torch.zeros(config["nb_rnn_layers"], 2, config["hidden_size"])
    y, _, _, _ = net(x, x, torch.zeros(2), h, h)
    print(f"experiment name      : {config['experiment_name']}")
    print(f"window               : {window} samples = {window / fe * 1000:.0f} ms")
    print(f"sequence             : {config['seq_len']} windows, {stride} samples ({stride / fe:.3f} s) apart"
          f" -> {config['seq_len'] * stride / fe:.1f} s of context")
    print(f"trainable parameters : {n_params}")
    print(f"dummy forward pass   : output shape {tuple(y.shape)}, values in [0, 1]: {bool(((y >= 0) & (y <= 1)).all())}")
    print(f"train / val device   : {config['device_train']} / {config['device_val']}"
          f" (CUDA available here: {torch.cuda.is_available()})")
    print(f"module globals       : PHASE={pdt.PHASE} TEST_SET={pdt.TEST_SET} THRESHOLD={pdt.THRESHOLD}")
    print(f"dataset folder       : {pdt.path_dataset}")
    name = pdt.filename_classification_dataset if config["classification"] else pdt.filename_regression_dataset
    for f in [name, pdt.subject_list, pdt.subject_list_p1, pdt.subject_list_p2]:
        status = "found" if (Path(pdt.path_dataset) / f).is_file() else "MISSING"
        print(f"  {status:8s} {f}")
    return n_params


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="my_variant", help="experiment name = checkpoint file name (default: my_variant)")
    p.add_argument("--hidden-size", type=int, default=7)
    p.add_argument("--nb-channel", type=int, default=31)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batches-per-epoch", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", type=int, default=0, help="random_state of the subject split (default 0)")
    p.add_argument("--phase", default="full", choices=["p1", "p2", "full"])
    p.add_argument("--no-test-set", action="store_true", help="use the 20%% held-out subjects for validation only")
    p.add_argument("--regression", action="store_true", help="train on graded labels with an MSE loss")
    p.add_argument("--reg-balancing", default="none", choices=["none", "lds", "sr"], help="loss reweighting in regression (default: none)")
    p.add_argument("--dataset-dir", default=None, help="folder with the dataset files; checkpoints are written here too")
    p.add_argument("--project", default=None, help="W&B project (default: <phase>-dataset-public, as the CLI)")
    p.add_argument("--unique", action="store_true", help="append a timestamp to the name so nothing is overwritten")
    p.add_argument("--check", action="store_true", help="only validate the configuration, do not train")
    args = p.parse_args()

    set_phase(args.phase, test_set=not args.no_test_set)
    if args.dataset_dir:
        pdt.path_dataset = Path(args.dataset_dir).resolve()
    config = make_config(args)
    describe(config)
    if args.check:
        return

    project = args.project or f"{args.phase}-dataset-public"
    t0 = time.time()
    best_loss, best_f1, best_epoch_es = pdt.run(config, wandb_project=project, save_model=True, unique_name=args.unique)
    print(f"finished in {time.time() - t0:.0f} s")
    print(f"best validation F1 = {best_f1:.4f} (validation loss of that model = {best_loss:.4f})")
    print(f"epoch with the lowest smoothed validation loss = {best_epoch_es}")
    print(f"checkpoints and .onnx files are in {pdt.path_dataset}")


if __name__ == "__main__":
    main()
