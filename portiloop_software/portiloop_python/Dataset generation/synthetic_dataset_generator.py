"""
synthetic_dataset_generator.py: build a SYNTHETIC sleep-spindle dataset in the exact file format that the
Portiloop training code (ANN/portiloop_detector_training.py) and the notebooks read.

Why synthetic?
    The original training files were derived from the Montreal Archive of Sleep Studies (MASS), whose
    recordings are only available on request to the MASS team. This repository therefore ships
    synthetic data: they let you install, run and learn every step of the pipeline (training,
    hyperparameter search, evaluation, real-time simulation) without access to MASS. A network
    trained on them learns synthetic spindles; it is NOT a valid detector for real EEG. To train a
    real detector, obtain an annotated EEG database yourself (see the tutorials) and convert it to
    this format.

What is simulated
    Every subject has its own background EEG level, spindle frequency, amplitude, duration and
    density; "phase 1" subjects are younger (more, larger spindles) and "phase 2" subjects are older
    (fewer, smaller, shorter spindles), loosely following the trends reported for MODA. Each 145 s
    segment contains:
      - background NREM-2-like EEG: a sum of first-order autoregressive processes (about 1/f) plus
        white noise, slow oscillations (0.5 to 2 Hz) and occasional K-complexes;
      - sleep spindles: waxing and waning 10.5 to 16 Hz bursts, 0.4 to 2.2 s long, some of them too
        small to be noticed by every scorer;
      - distractors that are not spindles: alpha bursts (8 to 10.5 Hz), short muscle bursts and 60 Hz
        line noise (removed by the Portiloop notch filter).
    Labels imitate the MODA group consensus: 4 to 8 simulated scorers mark each spindle with a
    probability and a confidence (1, 0.75 or 0.5) that grow with its amplitude, with jittered
    onsets and offsets; their scores are averaged sample by sample (rounded to 0.05), giving the
    regression labels. Samples above the group consensus threshold (0.2 for phase 1, 0.35 for
    phase 2) form events; events closer than 0.1 s are merged and events shorter than 0.3 s or
    longer than 2.5 s are removed, giving the classification labels (0 or 1).

Processing (same chain as portiloop_software/matlab/old_matlab/my_full_dataset_generation.m)
    column 0: 21-tap FIR low-pass (coef30.coe), 60 Hz IIR notch, delay compensated (10 samples), then
              online standardization (alpha_avg = 0.1, alpha_var = 0.001);
    column 1: sigma envelope: 21-tap FIR band-pass (coef1216.coe, a stand-in for the filter of
              envelope_extractor_simulink.slx), online standardization (alpha = 0.001), squaring,
              exponential smoothing (alpha = 0.01);
    column 2: power ratio of compute_ratio.m: mean FFT amplitude 10 to 16 Hz over mean 2 to 8 Hz,
              over the last 125 samples (0.5 s) of column 0;
    column 3: label: -1 during the 15 s lead-in, -2 during the 15 s tail, and inside the 115 s block
              0/1 (classification files) or the consensus score (regression file).
    Each segment is simulated with 30 s of extra signal before it, so that filters and
    standardization have settled; segments are therefore independent and identical across files.

Outputs (in --out, by default portiloop_software/dataset/)
    dataset_classification_{full,p1,p2}_big_250_matlab_standardized_envelope_pf.txt
    dataset_regression_full_big_250_matlab_standardized_envelope_pf.txt
    subject_sequence_{full,p1,p2}_big.txt          (subject ID, first segment, end segment)
    example_data_not_annotated.txt                 (400 s of preprocessed EEG from an extra subject)
    synthetic_dataset_info.json                    (parameters, per-subject settings, label statistics)

Usage (from the repository root; needs numpy and scipy only)
    python "portiloop_software/portiloop_python/Dataset generation/synthetic_dataset_generator.py"
    python ".../synthetic_dataset_generator.py" --out /tmp/data --p1 30 --p2 20 --seed 1
    python ".../synthetic_dataset_generator.py" --zip dataset.zip     # also pack the files as the release archive
"""
import argparse
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

FS = 250
LEAD_S, BLOCK_S, TAIL_S = 15, 115, 15
SEG = (LEAD_S + BLOCK_S + TAIL_S) * FS  # 36,250 samples, as in the original files
PREROLL = 30 * FS
DELAY = 10  # group delay of the 21-tap linear-phase FIR filters, in samples
GCT = {"p1": 0.2, "p2": 0.35}  # MODA group consensus thresholds (gold standard)

REPO_SOFTWARE = Path(__file__).resolve().parents[2]  # .../portiloop_software
COEF_DIR = REPO_SOFTWARE / "matlab" / "old_matlab"
DATASET_SUFFIX = "_big_250_matlab_standardized_envelope_pf.txt"


# ----------------------------------------------------------------------------------------- filters

def read_coe(path):
    """Read the COEFDATA of a Xilinx .coe file (RADIX=10)."""
    text = Path(path).read_text()
    data = text.split("COEFDATA=")[1].split(";")[0]
    return np.array([float(v) for v in data.replace("\n", "").split(",") if v.strip()])


FIR_LP = read_coe(COEF_DIR / "coef30.coe")      # Portiloop low-pass (about 30 Hz)
FIR_SIGMA = read_coe(COEF_DIR / "coef1216.coe")  # band-pass used here for the sigma envelope

# MATLAB [b, a] = iirnotch(60/125, 1/125, 1): b = [g, -2 g cos(w0), g], a = [1, -2 g cos(w0), 2 g - 1]
_NOTCH_GAIN = 0.9936459339823651
_NOTCH_C = math.cos(60 / (FS / 2) * math.pi)
NOTCH_B = np.array([_NOTCH_GAIN, -2 * _NOTCH_GAIN * _NOTCH_C, _NOTCH_GAIN])
NOTCH_A = np.array([1.0, -2 * _NOTCH_GAIN * _NOTCH_C, 2 * _NOTCH_GAIN - 1])


def ema(x, alpha):
    """m[0] = x[0]; m[i] = m[i-1] + alpha (x[i] - m[i-1])  (exact, vectorized)."""
    m, _ = lfilter([alpha], [1, -(1 - alpha)], x[1:], zi=[(1 - alpha) * x[0]])
    return np.concatenate([[x[0]], m])


def online_standardize(x, alpha_avg, alpha_var):
    """Exact vectorized version of the MATLAB loop (first sample left unchanged):
        delta = x(i) - m; m = m + alpha_avg delta; v = (1 - alpha_var)(v + alpha_var delta^2); y(i) = (x(i) - m) / sqrt(v)
    """
    m = ema(x, alpha_avg)
    delta = x[1:] - m[:-1]
    v = lfilter([(1 - alpha_var) * alpha_var], [1, -(1 - alpha_var)], delta ** 2)
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = (x[1:] - m[1:]) / np.sqrt(v)
    return y


def preprocess_columns(raw):
    """raw: PREROLL + n + DELAY samples. Returns columns 0 to 2 for the n samples after the pre-roll."""
    n = len(raw) - PREROLL - DELAY
    lp = lfilter(NOTCH_B, NOTCH_A, lfilter(FIR_LP, [1.0], raw))[DELAY:]  # delay compensated
    col0 = online_standardize(lp, 0.1, 0.001)
    sig = lfilter(FIR_SIGMA, [1.0], raw)[DELAY:]
    sig = online_standardize(sig, 0.001, 0.001)
    col1 = ema(sig ** 2, 0.01)
    # power ratio over the last 125 samples (compute_ratio.m); bins are 2 Hz apart for L = 125
    w = sliding_window_view(col0[PREROLL - 124:PREROLL + n], 125)
    amp = np.abs(np.fft.rfft(w, axis=1))
    col2 = amp[:, 5:9].mean(axis=1) / amp[:, 1:5].mean(axis=1)  # (10, 12, 14, 16 Hz) / (2, 4, 6, 8 Hz)
    return col0[PREROLL:PREROLL + n], col1[PREROLL:PREROLL + n], col2


# ---------------------------------------------------------------------------------------- subjects

def make_subject(rng, phase):
    """Draw the per-subject parameters (arbitrary units comparable to microvolts)."""
    young = phase == "p1"
    return {
        "background_amp": float(rng.uniform(16, 28)),
        "slow_wave_amp": float(rng.uniform(18, 32)),
        "spindle_freq": float(rng.uniform(12.0, 14.5) if young else rng.uniform(11.2, 14.0)),
        "spindle_amp": float(rng.uniform(26, 42) if young else rng.uniform(18, 32)),
        "spindle_dur": float(rng.uniform(0.8, 1.2) if young else rng.uniform(0.6, 0.95)),
        "spindles_per_min": float(rng.uniform(5.5, 8.5) if young else rng.uniform(3.0, 6.0)),
        "alpha_bursts_per_min": float(rng.uniform(0.2, 1.5)),
        "line_noise_amp": float(rng.choice([0.0, rng.uniform(2, 15)])),
    }


# ------------------------------------------------------------------------------------------ signal

def background(rng, n, subj):
    """About 1/f background, slow oscillations and K-complexes."""
    x = np.zeros(n)
    for pole, gain in zip([0.999, 0.99, 0.95, 0.8], [0.035, 0.11, 0.3, 0.45]):
        x += lfilter([gain], [1, -pole], rng.standard_normal(n))
    x = subj["background_amp"] * (0.55 * x + 0.12 * rng.standard_normal(n))
    # slow oscillations: frequency random walk between 0.5 and 2 Hz, updated every second
    nsec = n // FS + 1
    f = np.clip(0.9 + np.cumsum(0.15 * rng.standard_normal(nsec)), 0.5, 2.0)
    phase = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.cumsum(np.repeat(f, FS)[:n]) / FS
    x += 0.5 * subj["slow_wave_amp"] * np.sin(phase)
    # K-complexes: about 1.5 per minute
    t = np.arange(n) / FS
    for c in rng.uniform(1, n / FS - 1, rng.poisson(n / FS / 40)):
        i0, i1 = int((c - 1) * FS), int((c + 1.5) * FS)
        d = t[i0:i1] - c
        a = subj["slow_wave_amp"] * rng.uniform(0.8, 1.3)
        x[i0:i1] += -3.2 * a * np.exp(-d ** 2 / 0.012) + 1.6 * a * np.exp(-(d - 0.35) ** 2 / 0.04)
    return x


def burst(n_samples, freq, amp, rng):
    """Waxing and waning oscillation (slight deceleration), as in docs/assets/js/portiloop.js."""
    u = np.arange(n_samples) / max(1, n_samples)
    env = np.sin(np.pi * u) ** 1.5 * (1 - 0.25 * u)
    return amp * env * np.sin(rng.uniform(0, 2 * np.pi) + 2 * np.pi * freq * (1 - 0.04 * u) * np.arange(n_samples) / FS)


def poisson_times(rng, duration_s, rate_per_min, min_gap_s):
    times, t = [], rng.exponential(60 / rate_per_min) if rate_per_min > 0 else duration_s
    while t < duration_s - 2.5:
        times.append(t)
        t += min_gap_s + rng.exponential(max(0.1, 60 / rate_per_min - min_gap_s))
    return times


def simulate(rng, n, subj, forced_spindles=()):
    """Returns the raw signal and the list of events (spindles and distractors)."""
    x = background(rng, n, subj)
    events = []
    dur_s = n / FS
    starts = [s for s in poisson_times(rng, dur_s, subj["spindles_per_min"], 1.5)
              if all(abs(s - f[0]) > 4 for f in forced_spindles)]
    spec = [(s, None, None) for s in starts] + list(forced_spindles)
    for s0, forced_dur, forced_amp in spec:
        dur = forced_dur or float(np.clip(rng.normal(subj["spindle_dur"], 0.3), 0.4, 2.2))
        # amplitudes are log-normal: most spindles are clear, some are small and easily missed
        amp = forced_amp or float(subj["spindle_amp"] * rng.lognormal(0, 0.45))
        freq = float(np.clip(rng.normal(subj["spindle_freq"], 0.6), 10.5, 16))
        i0 = int(s0 * FS)
        i1 = min(n, i0 + int(dur * FS))
        x[i0:i1] += burst(i1 - i0, freq, amp, rng)
        events.append({"kind": "spindle", "i0": i0, "i1": i1, "amp": amp, "freq": freq})
    for s0 in poisson_times(rng, dur_s, subj["alpha_bursts_per_min"], 3):
        i0 = int(s0 * FS)
        i1 = min(n, i0 + int(rng.uniform(1, 3) * FS))
        x[i0:i1] += burst(i1 - i0, rng.uniform(8, 10.5), subj["background_amp"] * rng.uniform(0.8, 1.6), rng)
        events.append({"kind": "alpha", "i0": i0, "i1": i1, "amp": None, "freq": None})
    for s0 in rng.uniform(0, dur_s - 1, rng.poisson(dur_s / 60)):  # short muscle bursts
        i0 = int(s0 * FS)
        i1 = min(n, i0 + int(rng.uniform(0.2, 0.8) * FS))
        x[i0:i1] += subj["background_amp"] * rng.uniform(1, 3) * rng.standard_normal(i1 - i0)
    if subj["line_noise_amp"] > 0:
        x += subj["line_noise_amp"] * np.sin(2 * np.pi * 60 * np.arange(n) / FS + rng.uniform(0, 2 * np.pi))
    return x, events


# ------------------------------------------------------------------------------------------ labels

def consensus_score(rng, n, events, subj):
    """MODA-like averaged scorer confidence, sample by sample, rounded to 0.05."""
    n_scorers = int(rng.integers(4, 9))
    score = np.zeros(n)
    for ev in events:
        if ev["kind"] == "spindle":
            salience = ev["amp"] / subj["background_amp"]
            p_mark = 1 / (1 + math.exp(-(salience - 1.0) / 0.22))
            p_high = 1 / (1 + math.exp(-(salience - 1.6) / 0.3))
        else:  # alpha bursts are occasionally mistaken for spindles, with low confidence
            p_mark, p_high = 0.06, 0.0
        for _ in range(n_scorers):
            if rng.uniform() >= p_mark:
                continue
            conf = 1.0 if rng.uniform() < p_high else float(rng.choice([0.75, 0.5]))
            length = ev["i1"] - ev["i0"]
            # scorers mark the visible part of the burst; onsets and offsets vary between scorers
            a = ev["i0"] + int(length * 0.12) + int(rng.normal(0, 0.08) * FS)
            b = ev["i1"] - int(length * 0.12) + int(rng.normal(0, 0.08) * FS)
            a, b = max(0, a), min(n, b)
            if b > a:
                score[a:b] += conf
    score = np.round(score / n_scorers / 0.05) * 0.05
    return np.clip(score, 0, 1), n_scorers


def gold_standard(score, threshold):
    """Threshold, merge events closer than 0.1 s, keep events between 0.3 and 2.5 s."""
    above = np.concatenate([[0], (score > threshold + 1e-9).astype(np.int8), [0]])
    d = np.diff(above)
    starts, ends = list(np.flatnonzero(d == 1)), list(np.flatnonzero(d == -1))
    merged = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] < 0.1 * FS:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    label = np.zeros(len(score), dtype=np.int8)
    for s, e in merged:
        if 0.3 * FS <= e - s <= 2.5 * FS:
            label[s:e] = 1
    return label


# --------------------------------------------------------------------------------------- generation

def make_segment(rng, subj, phase):
    n = PREROLL + SEG + DELAY
    raw, events = simulate(rng, n, subj)
    c0, c1, c2 = preprocess_columns(raw)
    # labels refer to the SEG samples after the pre-roll
    for ev in events:
        ev["i0"] -= PREROLL
        ev["i1"] -= PREROLL
    events = [ev for ev in events if ev["i1"] > 0 and ev["i0"] < SEG]
    for ev in events:
        ev["i0"], ev["i1"] = max(0, ev["i0"]), min(SEG, ev["i1"])
    score, n_scorers = consensus_score(rng, SEG, events, subj)
    cls = gold_standard(score, GCT[phase]).astype(np.float32)
    reg = score.astype(np.float32)
    lead, tail = LEAD_S * FS, TAIL_S * FS
    for lab in (cls, reg):
        lab[:lead] = -1
        lab[-tail:] = -2
    cols = np.stack([c0, c1, c2]).T.astype(np.float32)
    return cols, cls, reg, n_scorers


def write_rows(path, cols, labels, label_fmt):
    with open(path, "w") as f:
        np.savetxt(f, np.column_stack([cols, labels]), fmt=["%.7g", "%.7g", "%.7g", label_fmt], delimiter=",")


def generate(out, n_p1, n_p2, n_long, seed, zip_path=None, quiet=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = np.random.SeedSequence(seed)
    subj_seeds = root.spawn(n_p1 + n_p2 + 1)
    order_rng = np.random.default_rng(root.spawn(1)[0])

    subjects = []
    for phase, count, offset in (("p1", n_p1, 0), ("p2", n_p2, n_p1)):
        long_ids = set(order_rng.choice(count, size=min(n_long, count), replace=False).tolist())
        for k in range(count):
            subjects.append({"id": f"syn-{phase}-{k + 1:03d}", "phase": phase,
                             "n_segments": 10 if k in long_ids else 3, "seed": subj_seeds[offset + k]})
    order = order_rng.permutation(len(subjects))  # the full list mixes both phases, as the original
    subjects = [subjects[i] for i in order]

    segs = {}
    info_subjects = []
    for s in subjects:
        rng = np.random.default_rng(s["seed"])
        params = make_subject(rng, s["phase"])
        segs[s["id"]] = []
        for _ in range(s["n_segments"]):
            cols, cls, reg, n_scorers = make_segment(rng, params, s["phase"])
            segs[s["id"]].append((cols, cls, reg))
        info_subjects.append({"id": s["id"], "phase": s["phase"], "n_segments": s["n_segments"],
                              **{k: round(v, 3) for k, v in params.items()}})
        if not quiet:
            print(f"  {s['id']}: {s['n_segments']} segments", file=sys.stderr)

    stats = {}
    for phase in ("full", "p1", "p2"):
        subset = [s for s in subjects if phase == "full" or s["phase"] == phase]
        lines, start = [], 0
        for s in subset:
            lines.append(f"{s['id']} {start} {start + s['n_segments']}")
            start += s["n_segments"]
        (out / f"subject_sequence_{phase}_big.txt").write_text("\n".join(lines) + "\n")
        cols = np.concatenate([c for s in subset for (c, _, _) in segs[s["id"]]])
        cls = np.concatenate([l for s in subset for (_, l, _) in segs[s["id"]]])
        write_rows(out / f"dataset_classification_{phase}{DATASET_SUFFIX}", cols, cls, "%d")
        if phase == "full":
            reg = np.concatenate([r for s in subset for (_, _, r) in segs[s["id"]]])
            write_rows(out / f"dataset_regression_{phase}{DATASET_SUFFIX}", cols, reg, "%g")
        labelled = cls[cls >= 0]
        stats[phase] = {"subjects": len(subset), "segments": start, "rows": int(len(cls)),
                        "spindle_fraction": round(float(labelled.mean()), 4)}
        if not quiet:
            print(f"{phase}: {stats[phase]}", file=sys.stderr)

    # example_data_not_annotated.txt: 400 s from an extra subject that is in none of the files,
    # with a clear spindle at 117.6 s (the notebook offline_inference.ipynb plots 115 to 125 s)
    rng = np.random.default_rng(subj_seeds[-1])
    params = make_subject(rng, "p1")
    n_ex = 100_000
    raw, _ = simulate(rng, PREROLL + n_ex + DELAY, params,
                      forced_spindles=[((PREROLL / FS) + 117.6, 1.6, 2.0 * params["spindle_amp"])])
    c0, _, _ = preprocess_columns(raw)
    np.savetxt(out / "example_data_not_annotated.txt", c0.astype(np.float32), fmt="%.7g")

    info = {
        "description": "SYNTHETIC EEG generated by synthetic_dataset_generator.py; not recorded from any person.",
        "seed": seed, "fs": FS, "segment_s": [LEAD_S, BLOCK_S, TAIL_S],
        "group_consensus_threshold": GCT, "files": stats,
        "example_data_not_annotated": {"samples": n_ex, "spindle_at_s": 117.6, "subject": params},
        "subjects": info_subjects,
    }
    (out / "synthetic_dataset_info.json").write_text(json.dumps(info, indent=1))

    if zip_path:
        names = sorted(p.name for p in out.iterdir() if p.name.startswith(("dataset_", "subject_sequence_"))
                       or p.name in ("example_data_not_annotated.txt", "synthetic_dataset_info.json"))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for name in names:  # same layout as the original release: a top-level dataset/ folder
                z.write(out / name, f"dataset/{name}")
        if not quiet:
            print(f"wrote {zip_path} ({Path(zip_path).stat().st_size / 1e6:.0f} MB)", file=sys.stderr)
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(REPO_SOFTWARE / "dataset"), help="output folder (default: portiloop_software/dataset)")
    p.add_argument("--p1", type=int, default=20, help="number of younger (phase 1) subjects (default 20)")
    p.add_argument("--p2", type=int, default=14, help="number of older (phase 2) subjects (default 14)")
    p.add_argument("--long", type=int, default=2, help="subjects per phase with 10 segments instead of 3 (default 2)")
    p.add_argument("--seed", type=int, default=2022, help="random seed; the same seed gives the same files (default 2022)")
    p.add_argument("--zip", default=None, help="also write a dataset.zip archive (files under dataset/)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)
    if a.p1 < 6 or a.p2 < 6:
        # the training code keeps 80% of each phase for training and halves the rest into test and
        # validation (sklearn train_test_split): with fewer than 6 subjects the rest is a single subject
        p.error("use at least 6 subjects per phase, so that training, validation and test sets are not empty")
    generate(a.out, a.p1, a.p2, a.long, a.seed, a.zip, a.quiet)


if __name__ == "__main__":
    main()
