"""
ONNX export utilities for Portiloop models (Sleep Spindle detector, TinySleepNet, NanoSleepNet-TCN).

Produces single-step stateful ONNX models (opset 11, 4D NCHW spatial convolutions and explicit
recurrent cells) suitable for both ONNX Runtime (CPU/GPU) and the D-Robotics RDK X5 BPU
toolchain (`hb_mapper makertbin`, march='bayes-e').
"""
import argparse
from pathlib import Path
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _StaticPadW(nn.Module):
    """
    Static width padding on 4D NCHW tensors `(1, C, 1, W)` lowered to a single ONNX `Concat`
    with constant zero buffers (avoids dynamic `ConstantOfShape` / `Slice` in ONNX opset 11).
    """
    def __init__(self, channels: int, left: int, right: int):
        super().__init__()
        self.left = left
        self.right = right
        if left > 0:
            self.register_buffer("pad_l", torch.zeros(1, channels, 1, left, dtype=torch.float32))
        if right > 0:
            self.register_buffer("pad_r", torch.zeros(1, channels, 1, right, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.left > 0:
            parts.append(self.pad_l)
        parts.append(x)
        if self.right > 0:
            parts.append(self.pad_r)
        if len(parts) == 1:
            return x
        return torch.cat(parts, dim=3)


class _UnrolledGRUCell(nn.Module):
    """
    Single-step GRU cell lowered to explicit Linear + Sigmoid/Tanh + elementwise ops
    so that BPU compilers (Horizon bayes-e `hb_mapper`) do not encounter an ONNX Scan/GRU op.
    Matches `torch.nn.GRU` layer l mathematically to float32 precision.
    """
    def __init__(self, weight_ih: torch.Tensor, weight_hh: torch.Tensor,
                 bias_ih: torch.Tensor, bias_hh: torch.Tensor):
        super().__init__()
        hidden_size = weight_hh.shape[1]
        in_features = weight_ih.shape[1]
        self.hidden_size = hidden_size
        self.ih = nn.Linear(in_features, 3 * hidden_size, bias=True)
        self.hh = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        with torch.no_grad():
            self.ih.weight.copy_(weight_ih)
            self.ih.bias.copy_(bias_ih)
            self.hh.weight.copy_(weight_hh)
            self.hh.bias.copy_(bias_hh)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gi = self.ih(x)
        gh = self.hh(h)
        i_r, i_z, i_n = torch.split(gi, self.hidden_size, dim=-1)
        h_r, h_z, h_n = torch.split(gh, self.hidden_size, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h


class _ConvPool2dBlock(nn.Module):
    """
    1D ConvPoolModule lowered to 2D Conv2d((1, k)) + ReLU + MaxPool2d((1, k)) on 4D NCHW tensors
    for native BPU hardware acceleration.
    """
    def __init__(self, conv1d: nn.Conv1d, pool1d: nn.MaxPool1d):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=conv1d.in_channels,
            out_channels=conv1d.out_channels,
            kernel_size=(1, conv1d.kernel_size[0]),
            stride=(1, conv1d.stride[0]),
            padding=(0, conv1d.padding[0]),
            dilation=(1, conv1d.dilation[0]),
            bias=conv1d.bias is not None,
        )
        with torch.no_grad():
            self.conv.weight.copy_(conv1d.weight.unsqueeze(2))
            if conv1d.bias is not None:
                self.conv.bias.copy_(conv1d.bias)
        self.pool = nn.MaxPool2d(
            kernel_size=(1, pool1d.kernel_size),
            stride=(1, pool1d.stride),
            padding=(0, pool1d.padding),
            dilation=(1, pool1d.dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.conv(x)))


class PortiloopSpindleOnnxStep(nn.Module):
    """
    Single-step stateful ONNX wrapper for `PortiloopNetwork` (1-input sleep spindle detector).

    Inputs:
        x: (1, 1, 1, window_size) float32 (default window_size=54)
        h: (1, hidden_size) float32 (default hidden_size=7, or (1, nb_rnn_layers * hidden_size))
    Outputs:
        y: (1, 1) float32 probability in [0, 1]
        h_out: (1, hidden_size) float32 updated hidden state
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        blocks = [_ConvPool2dBlock(net.first_layer_input1.conv, net.first_layer_input1.pool)]
        for layer in net.seq_input1:
            blocks.append(_ConvPool2dBlock(layer.conv, layer.pool))
        self.cnn = nn.Sequential(*blocks)

        gru = net.gru_input1
        self.num_layers = gru.num_layers
        self.hidden_size = gru.hidden_size
        self.cells = nn.ModuleList([
            _UnrolledGRUCell(
                getattr(gru, f"weight_ih_l{l}"),
                getattr(gru, f"weight_hh_l{l}"),
                getattr(gru, f"bias_ih_l{l}"),
                getattr(gru, f"bias_hh_l{l}"),
            )
            for l in range(self.num_layers)
        ])
        self.fc = nn.Linear(net.fc.in_features, net.fc.out_features)
        with torch.no_grad():
            self.fc.weight.copy_(net.fc.weight)
            self.fc.bias.copy_(net.fc.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        feat = self.cnn(x)
        flat = feat.reshape(1, -1)
        if self.num_layers == 1:
            h_next = self.cells[0](flat, h)
            out_feat = h_next
        else:
            h_slices = torch.split(h, self.hidden_size, dim=-1)
            next_slices = []
            cur = flat
            for cell, h_l in zip(self.cells, h_slices):
                cur = cell(cur, h_l)
                next_slices.append(cur)
            h_next = torch.cat(next_slices, dim=-1)
            out_feat = cur
        y = torch.sigmoid(self.fc(out_feat))
        return y, h_next


class TinySleepNetOnnxStep(nn.Module):
    """
    Single-step stateful TinySleepNet (CNN + peephole LSTM) using 4D NCHW convolutions.
    Can be initialized directly from `tinysleepnet_float.tflite`.

    Inputs:
        x: (1, 1, 1, 3000) float32 epoch (30 s at 100 Hz)
        c: (1, 128) float32 LSTM cell state
        h: (1, 128) float32 LSTM hidden state
    Outputs:
        logits: (1, 5) float32 stage logits (W, N1, N2, N3, REM)
        c_out: (1, 128) float32 next cell state
        h_out: (1, 128) float32 next hidden state
    """
    def __init__(self, tflite_path=None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1, 50), stride=(1, 6), padding=(0, 22), bias=True)
        self.pad2_1 = _StaticPadW(128, 3, 4)
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=(1, 8), stride=(1, 1), padding=0, bias=True)
        self.pad2_2 = _StaticPadW(128, 3, 4)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(1, 8), stride=(1, 1), padding=0, bias=True)
        self.pad2_3 = _StaticPadW(128, 3, 4)
        self.conv2_3 = nn.Conv2d(128, 128, kernel_size=(1, 8), stride=(1, 1), padding=0, bias=True)
        self.lstm_fc = nn.Linear(2176, 512, bias=True)
        self.register_buffer("w_cf", torch.zeros(128))
        self.register_buffer("w_ci", torch.zeros(128))
        self.register_buffer("w_co", torch.zeros(128))
        self.forget_bias = 1.0
        self.classifier = nn.Linear(128, 5, bias=True)

        if tflite_path is not None:
            self.load_from_tflite(tflite_path)

    def load_from_tflite(self, tflite_path):
        import tflite
        with open(tflite_path, "rb") as f:
            buf = f.read()
        model = tflite.Model.GetRootAsModel(buf, 0)
        sub = model.Subgraphs(0)

        def get_arr(idx, dtype=np.float32):
            t = sub.Tensors(idx)
            b = model.Buffers(t.Buffer())
            shape = [t.Shape(j) for j in range(t.ShapeLength())]
            return np.frombuffer(b.DataAsNumpy(), dtype=dtype).reshape(shape).copy()

        with torch.no_grad():
            self.conv1.weight.copy_(torch.from_numpy(get_arr(10).transpose(0, 3, 2, 1)))
            self.conv1.bias.copy_(torch.from_numpy(get_arr(6)))
            self.conv2_1.weight.copy_(torch.from_numpy(get_arr(9).transpose(0, 3, 2, 1)))
            self.conv2_1.bias.copy_(torch.from_numpy(get_arr(5)))
            self.conv2_2.weight.copy_(torch.from_numpy(get_arr(8).transpose(0, 3, 2, 1)))
            self.conv2_2.bias.copy_(torch.from_numpy(get_arr(4)))
            self.conv2_3.weight.copy_(torch.from_numpy(get_arr(7).transpose(0, 3, 2, 1)))
            self.conv2_3.bias.copy_(torch.from_numpy(get_arr(3)))

            self.w_cf.copy_(torch.from_numpy(get_arr(12)))
            self.w_ci.copy_(torch.from_numpy(get_arr(13)))
            self.w_co.copy_(torch.from_numpy(get_arr(14)))
            self.forget_bias = float(get_arr(16))

            lstm_w = get_arr(19)
            lstm_w_x = lstm_w[:, :2048].reshape(512, 16, 128).transpose(0, 2, 1).reshape(512, 2048)
            lstm_w_reordered = np.concatenate([lstm_w_x, lstm_w[:, 2048:]], axis=1)
            self.lstm_fc.weight.copy_(torch.from_numpy(lstm_w_reordered))
            self.lstm_fc.bias.copy_(torch.from_numpy(get_arr(15)))

            self.classifier.weight.copy_(torch.from_numpy(get_arr(20)))
            self.classifier.bias.copy_(torch.from_numpy(get_arr(18)))

    def forward(self, x: torch.Tensor, c: torch.Tensor, h: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=(1, 8), stride=(1, 8), padding=(0, 2))
        x = F.relu(self.conv2_1(self.pad2_1(x)))
        x = F.relu(self.conv2_2(self.pad2_2(x)))
        x = F.relu(self.conv2_3(self.pad2_3(x)))
        x = F.max_pool2d(x, kernel_size=(1, 4), stride=(1, 4), ceil_mode=True)
        flat = x.reshape(1, 2048)
        xh = torch.cat([flat, h], dim=1)
        gates = self.lstm_fc(xh)
        i, j, f, o = torch.split(gates, 128, dim=1)
        f_gate = torch.sigmoid(f + self.forget_bias + c * self.w_cf)
        i_gate = torch.sigmoid(i + c * self.w_ci)
        c_next = f_gate * c + i_gate * torch.tanh(j)
        o_gate = torch.sigmoid(o + c_next * self.w_co)
        h_next = o_gate * torch.tanh(c_next)
        logits = self.classifier(h_next)
        return logits, c_next, h_next


class NanoSleepNetOnnxStep(nn.Module):
    """
    Single-step causal NanoSleepNet-TCN with BatchNorm fused into 4D NCHW Conv2d layers.

    Inputs:
        x: (1, 1, 1, 3000) float32 epoch (30 s at 100 Hz)
        past_tokens: (1, 64, 1, 9) float32 tokens of the previous 9 epochs
    Outputs:
        logits: (1, 5) float32 stage logits (W, N1, N2, N3, REM)
        token: (1, 64) float32 token of the current epoch
    """
    def __init__(self, npz_path):
        super().__init__()
        with np.load(npz_path) as weights:
            w = {k: weights[k].astype(np.float32) for k in weights.files}

        def make_fused_conv(prefix, stride=1, padding=0, dilation=1, groups=1):
            weight = w[f"{prefix}.w"][:, :, None, :]
            scale = w[f"{prefix}.scale"]
            shift = w[f"{prefix}.shift"]
            fused_w = weight * scale[:, None, None, None]
            conv = nn.Conv2d(
                in_channels=weight.shape[1] * groups,
                out_channels=weight.shape[0],
                kernel_size=(1, weight.shape[3]),
                stride=(1, stride),
                padding=(0, padding),
                dilation=(1, dilation),
                groups=groups,
                bias=True,
            )
            with torch.no_grad():
                conv.weight.copy_(torch.from_numpy(fused_w))
                conv.bias.copy_(torch.from_numpy(shift))
            return conv

        self.pad_stem = _StaticPadW(1, 4, 5)
        self.stem = make_fused_conv("stem", stride=6)
        self.res1 = make_fused_conv("res1", padding=7, groups=4)
        self.res2 = make_fused_conv("res2", padding=3, groups=4)
        self.pad_down = _StaticPadW(32, 2, 3)
        self.down = make_fused_conv("down", stride=4, groups=8)

        self.eca = nn.Conv2d(1, 1, kernel_size=(1, 5), padding=(0, 2), bias=False)
        with torch.no_grad():
            self.eca.weight.copy_(torch.from_numpy(w["eca.w"][:, :, None, :]))

        self.reduce = make_fused_conv("reduce")
        nb_blocks = len([k for k in w if k.endswith(".dilation")])
        self.dilations = [int(w[f"block{i}.dilation"]) for i in range(nb_blocks)]
        self.dw_pads = nn.ModuleList([
            _StaticPadW(32, self.dilations[i] * 2, 0)
            for i in range(nb_blocks)
        ])
        self.dw_convs = nn.ModuleList([
            make_fused_conv(f"block{i}.dw", dilation=self.dilations[i], groups=32)
            for i in range(nb_blocks)
        ])
        self.pw_convs = nn.ModuleList([
            make_fused_conv(f"block{i}.pw")
            for i in range(nb_blocks)
        ])
        self.expand = make_fused_conv("expand")
        self.classifier = nn.Linear(64, 5, bias=True)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.from_numpy(w["classifier.w"]))
            self.classifier.bias.copy_(torch.from_numpy(w["classifier.b"]))

    @staticmethod
    def _shuffle(x: torch.Tensor, groups: int, c: int, length: int) -> torch.Tensor:
        return x.reshape(groups, c // groups, 1, length).permute(1, 0, 2, 3).reshape(1, c, 1, length)

    def forward(self, x: torch.Tensor, past_tokens: torch.Tensor):
        x = F.gelu(self.stem(self.pad_stem(x)))
        y = self._shuffle(F.gelu(self.res1(x)), 4, 32, 500)
        x = F.gelu(self.res2(y) + x)
        x = self._shuffle(F.gelu(self.down(self.pad_down(x))), 8, 64, 125)
        x = F.avg_pool2d(x, kernel_size=(1, 4), stride=(1, 4), ceil_mode=True, count_include_pad=False)

        ch_mean = x.mean(dim=-1, keepdim=True)
        ch_seq = ch_mean.permute(0, 2, 3, 1)
        gate = torch.sigmoid(self.eca(ch_seq)).permute(0, 3, 1, 2)
        x = x * gate
        token_4d = x.mean(dim=-1, keepdim=True)

        tokens = torch.cat([past_tokens, token_4d], dim=-1)
        t = F.gelu(self.reduce(tokens))
        for pad_dw, dw, pw in zip(self.dw_pads, self.dw_convs, self.pw_convs):
            t = F.gelu(dw(pad_dw(t)))
            t = F.gelu(pw(t))
        features = F.gelu(self.expand(t) + tokens)
        _, last_feat_4d = torch.split(features, [9, 1], dim=3)
        last_feat = last_feat_4d.reshape(1, 64)
        logits = self.classifier(last_feat)
        return logits, token_4d.reshape(1, 64)


def export_spindle_model_to_onnx(net: nn.Module, output_path, window_size: int = 54, opset_version: int = 11):
    """
    Exports a trained `PortiloopNetwork` to a single-step stateful ONNX model (`output_path`).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = PortiloopSpindleOnnxStep(net.cpu().eval()).eval()
    dummy_x = torch.zeros(1, 1, 1, window_size, dtype=torch.float32)
    dummy_h = torch.zeros(1, wrapper.num_layers * wrapper.hidden_size, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_h),
        str(output_path),
        input_names=["x", "h"],
        output_names=["y", "h_out"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    return output_path


def export_tinysleepnet_to_onnx(tflite_path, output_path, opset_version: int = 11):
    """
    Exports TinySleepNet (`tinysleepnet_float.tflite`) to a single-step stateful ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = TinySleepNetOnnxStep(tflite_path).eval()
    dummy_x = torch.zeros(1, 1, 1, 3000, dtype=torch.float32)
    dummy_c = torch.zeros(1, 128, dtype=torch.float32)
    dummy_h = torch.zeros(1, 128, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_x, dummy_c, dummy_h),
        str(output_path),
        input_names=["x", "c", "h"],
        output_names=["logits", "c_out", "h_out"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    return output_path


def export_nanosleepnet_to_onnx(npz_path, output_path, opset_version: int = 11):
    """
    Exports NanoSleepNet-TCN (`nanosleepnet.npz`) to a single-step stateful ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = NanoSleepNetOnnxStep(npz_path).eval()
    dummy_x = torch.zeros(1, 1, 1, 3000, dtype=torch.float32)
    dummy_past = torch.zeros(1, 64, 1, 9, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_x, dummy_past),
        str(output_path),
        input_names=["x", "past_tokens"],
        output_names=["logits", "token"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export Portiloop models to BPU-compatible ONNX (opset 11)")
    parser.add_argument("--output-dir", type=Path, default=Path("onnx_models"),
                        help="Directory where exported .onnx models will be written")
    parser.add_argument("--spindle-checkpoint", type=Path, default=None,
                        help="Path to PortiloopNetwork PyTorch checkpoint (default: pareto_search_15_35_v4_0)")
    parser.add_argument("--tinysleepnet-tflite", type=Path, default=None,
                        help="Optional path to tinysleepnet_float.tflite to also export TinySleepNet")
    parser.add_argument("--nanosleepnet-npz", type=Path, default=None,
                        help="Optional path to nanosleepnet.npz to also export NanoSleepNet")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11 for Horizon BPU)")
    args = parser.parse_args()

    from portiloop_software.portiloop_python.ANN.portiloop_detector_training import (
        get_final_model_config_dict, get_trained_model, PortiloopNetwork, path_dataset
    )

    if args.spindle_checkpoint is not None:
        cfg = get_final_model_config_dict()
        net = PortiloopNetwork(cfg)
        ckpt = torch.load(args.spindle_checkpoint, map_location="cpu", weights_only=False)
        net.load_state_dict(ckpt["model_state_dict"])
    else:
        cfg = get_final_model_config_dict()
        exp_dir = Path(__file__).resolve().parents[2] / "experiments"
        net = get_trained_model(cfg, exp_dir)

    window_size = int(cfg["window_size_s"] * cfg["fe"])
    spindle_out = export_spindle_model_to_onnx(
        net, args.output_dir / "portiloop_spindle.onnx", window_size=window_size, opset_version=args.opset
    )
    print(f"Exported Sleep Spindle ONNX model to: {spindle_out}")

    if args.tinysleepnet_tflite is not None and args.tinysleepnet_tflite.is_file():
        tsn_out = export_tinysleepnet_to_onnx(
            args.tinysleepnet_tflite, args.output_dir / "tinysleepnet.onnx", opset_version=args.opset
        )
        print(f"Exported TinySleepNet ONNX model to:  {tsn_out}")

    if args.nanosleepnet_npz is not None and args.nanosleepnet_npz.is_file():
        nsn_out = export_nanosleepnet_to_onnx(
            args.nanosleepnet_npz, args.output_dir / "nanosleepnet.onnx", opset_version=args.opset
        )
        print(f"Exported NanoSleepNet ONNX model to:  {nsn_out}")


if __name__ == "__main__":
    main()
