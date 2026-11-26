"""
Export a trained PortiloopNetwork checkpoint to a JSON file that the tutorial
website runs directly in the browser (see docs/assets/js/portiloop.js, PortiloopModel).

The JSON also embeds a reference test vector (inputs and the outputs produced by
the real PyTorch PortiloopNetwork) so the JavaScript port can be checked numerically:

    python3 docs/tools/export_model_json.py
    node docs/tools/check_model_js.mjs

Only the 1-input model (RNN=True, envelope_input=False, power_features_input=False)
is supported, which is the configuration of the model published in the Portiloop paper.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The training module imports sklearn and wandb at import time; neither is needed to
# run a forward pass, so we provide empty stand-ins when they are not installed.
for name in ("sklearn", "sklearn.model_selection", "wandb"):
    try:
        __import__(name)
    except ImportError:
        mod = types.ModuleType(name)
        mod.train_test_split = None
        sys.modules[name] = mod

from portiloop_software.portiloop_python.ANN.portiloop_detector_training import (  # noqa: E402
    PortiloopNetwork,
    get_final_model_config_dict,
)

CKPT = ROOT / "portiloop_software" / "experiments" / "pareto_search_15_35_v4_0"
OUT = ROOT / "docs" / "assets" / "models" / "portiloop_spindle_v4_0.json"


def main():
    cfg = get_final_model_config_dict()
    net = PortiloopNetwork(cfg)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()

    weights = {k: {"shape": list(v.shape), "data": [round(float(x), 7) for x in v.flatten().tolist()]}
               for k, v in net.state_dict().items()}

    # Reference test: a stream of 12 consecutive windows fed one at a time,
    # carrying the GRU hidden state, exactly as the Portiloop does online.
    window = int(cfg["window_size_s"] * cfg["fe"])
    rng = np.random.default_rng(0)
    t = np.arange(window * 12) / cfg["fe"]
    sig = rng.normal(0, 1, t.size) + 2.5 * np.sin(2 * np.pi * 13 * t) * np.exp(-((t - 1.5) ** 2) / 0.2)
    windows = sig.reshape(12, window).astype(np.float32)
    h = torch.zeros(cfg["nb_rnn_layers"], 1, cfg["hidden_size"])
    outputs, hiddens = [], []
    with torch.no_grad():
        for w in windows:
            x = torch.tensor(w).view(1, 1, window)
            y, h, _, _ = net(x, None, None, h, None)
            outputs.append(float(y.item()))
            hiddens.append(h.flatten().tolist())

    keep = ["window_size_s", "fe", "seq_stride_s", "seq_len", "nb_conv_layers", "nb_channel",
            "kernel_conv", "kernel_pool", "stride_conv", "stride_pool", "dilation_conv",
            "dilation_pool", "hidden_size", "nb_rnn_layers", "nb_out", "time_in_past"]
    payload = {
        "name": "pareto_search_15_35_v4_0",
        "description": "Portiloop 1-input spindle detector (Valenchon et al., 2022), PyTorch checkpoint exported to JSON.",
        "epoch": int(ckpt.get("epoch", -1)),
        "n_parameters": int(sum(p.numel() for p in net.parameters())),
        "config": {k: cfg[k] for k in keep},
        "weights": weights,
        "reference_test": {
            "windows": windows.round(7).tolist(),
            "outputs": outputs,
            "hidden_states": hiddens,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB), {payload['n_parameters']} parameters")
    print("reference outputs:", [round(o, 4) for o in outputs])


if __name__ == "__main__":
    main()
