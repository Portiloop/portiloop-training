"""
preprocess_like_portiloop.py: reproduce in Python the preprocessing that the Portiloop applies online,
so that your own EEG looks like the data the pre-trained spindle detector was trained on.

Pipeline (same order as the device and as portiloop_software/matlab/preprocess_unannotated_data.m):
    1. resample to 250 Hz                 scipy.signal.resample_poly (polyphase, anti-aliasing filter)
    2. 21-tap FIR lowpass (about 30 Hz)   exact coefficients of portiloop_software/matlab/old_matlab/coef30.coe
    3. IIR notch at 60 Hz                 MATLAB iirnotch(60/125, 1/125, 1)
    4. online standardization             exponential moving mean (alpha_avg = 0.1) and variance (alpha_var = 0.001)

Usage examples (run from anywhere; only numpy and scipy are needed, plus pyedflib to read EDF files):
    # a text file with one sample per line, recorded at 256 Hz
    python3 preprocess_like_portiloop.py my_c3.txt --fs 256 -o preprocessed_my_c3.txt

    # directly from an EDF file: channel C3 re-referenced to A2 (labels are matched by substring)
    python3 preprocess_like_portiloop.py night1.edf --channel C3 --reference A2 -o preprocessed_night1.txt

    # check that the coefficients below are identical to the repository's coef30.coe
    python3 preprocess_like_portiloop.py --selftest

Difference with the MATLAB script: by default this tool is CAUSAL, exactly like the device and like
PL.preprocess in docs/assets/js/portiloop.js, so the filtered signal lags the input by the FIR group delay
(10 samples = 40 ms). The MATLAB script removes that delay offline; use --compensate-delay to do the same
(the output is then shifted 10 samples earlier and the end is padded by repeating the last samples).
The resampler is not bit-identical to MATLAB's resample() either (both use Kaiser-windowed polyphase
anti-aliasing filters, with slightly different designs).
"""
import argparse
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import lfilter, resample_poly

FS_PORTILOOP = 250  # Hz

# coef30.coe (RADIX=10), 21 taps, symmetric (linear phase): group delay = (21 - 1) / 2 = 10 samples.
FIR_COEFS = np.array([
    0.001623780150148094927192721215192250384, 0.014988684599373741992978104065059596905,
    0.021287595318265635502275046064823982306, 0.007349500393709578957568417933998716762,
    -0.025127515717112181709014251396183681209, -0.052210507359822452833064687638398027048,
    -0.039273839505489904766477593511808663607, 0.033021568427940004020193498490698402748,
    0.147606943281569008563636202779889572412, 0.254000252034505602516389899392379447818,
    0.297330876398883392486283128164359368384, 0.254000252034505602516389899392379447818,
    0.147606943281569008563636202779889572412, 0.033021568427940004020193498490698402748,
    -0.039273839505489904766477593511808663607, -0.052210507359822452833064687638398027048,
    -0.025127515717112181709014251396183681209, 0.007349500393709578957568417933998716762,
    0.021287595318265635502275046064823982306, 0.014988684599373741992978104065059596905,
    0.001623780150148094927192721215192250384,
])
FIR_DELAY = (len(FIR_COEFS) - 1) // 2  # 10 samples = 40 ms at 250 Hz

# MATLAB [b, a] = iirnotch(60/125, 1/125, 1): b = [g, -2 g cos(w0), g], a = [1, -2 g cos(w0), 2 g - 1]
NOTCH_GAIN = 0.9936459339823651
NOTCH_W0 = 60 / (FS_PORTILOOP / 2) * math.pi


def notch_coefficients(gain=NOTCH_GAIN, w0=NOTCH_W0):
    c = -2 * gain * math.cos(w0)
    return np.array([gain, c, gain]), np.array([1.0, c, 2 * gain - 1])


def resample_to_250(x, fs_in):
    """Polyphase resampling to 250 Hz (fs_in may be any positive number, e.g. 256, 100, 512, 1000)."""
    x = np.asarray(x, dtype=float)
    if fs_in == FS_PORTILOOP:
        return x.copy()
    ratio = Fraction(FS_PORTILOOP / fs_in).limit_denominator(10000)
    return resample_poly(x, ratio.numerator, ratio.denominator)


def portiloop_filter(x, notch=True):
    """Causal FIR lowpass then 60 Hz notch, both starting from zero initial state (as the device after reset)."""
    y = lfilter(FIR_COEFS, [1.0], x)
    lowpass = y
    if notch:
        b, a = notch_coefficients()
        y = lfilter(b, a, y)
    return lowpass, y


def online_standardize(x, alpha_avg=0.1, alpha_var=0.001):
    """
    Vectorized version of the loop in preprocess_unannotated_data.m:
        moving_average = x(1); moving_variance = 0;
        for i = 2:N
            delta = x(i) - moving_average;
            moving_average = moving_average + alpha_av * delta;
            moving_variance = (1 - alpha_var) * (moving_variance + alpha_var * delta^2);
            y(i) = (x(i) - moving_average) / sqrt(moving_variance);
    The first sample is passed through unchanged, as in the MATLAB script.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    y = x.copy()
    if n < 2:
        return y, x.copy(), np.zeros(n)
    a, b = alpha_avg, alpha_var
    mean = np.empty(n)
    mean[0] = x[0]
    # mean[i] = a x[i] + (1 - a) mean[i-1]
    mean[1:], _ = lfilter([a], [1.0, -(1 - a)], x[1:], zi=[(1 - a) * x[0]])
    delta = x[1:] - mean[:-1]
    var = np.zeros(n)
    # var[i] = (1 - b) var[i-1] + (1 - b) b delta[i]^2, var[0] = 0
    var[1:] = lfilter([(1 - b) * b], [1.0, -(1 - b)], delta ** 2)
    sd = np.sqrt(var[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        y[1:] = np.where(sd > 0, (x[1:] - mean[1:]) / sd, 0.0)
    return y, mean, np.sqrt(var)


def preprocess(raw, fs_in, notch=True, alpha_avg=0.1, alpha_var=0.001, compensate_delay=0):
    """Full chain. Returns a dict of numpy arrays at 250 Hz (resampled, lowpass, notched, standardized)."""
    resampled = resample_to_250(raw, fs_in)
    lowpass, filtered = portiloop_filter(resampled, notch=notch)
    if compensate_delay:
        d = int(compensate_delay)
        # like [out(10:end); out(end-7:end)] in MATLAB: shift earlier and pad with the last d samples
        filtered = np.concatenate([filtered[d:], filtered[-d:]])
    standardized, mean, sd = online_standardize(filtered, alpha_avg, alpha_var)
    return {"resampled": resampled, "lowpass": lowpass, "notched": filtered,
            "standardized": standardized, "running_mean": mean, "running_sd": sd}


def load_text(path, column=0):
    """One sample per line; a comma, semicolon, tab or space separated table also works (pick --column)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    delim = "," if "," in first else (";" if ";" in first else None)
    data = np.genfromtxt(path, delimiter=delim, dtype=float, invalid_raise=False)
    data = np.atleast_1d(data)
    if data.ndim == 2:
        data = data[:, column]
    data = data[np.isfinite(data)]  # drops a header line or empty lines (parsed as NaN)
    return data


def load_edf(path, channel="C3", reference=None):
    """Read one channel (optionally minus a reference channel) and its sampling rate with pyedflib."""
    import pyedflib  # installed with the project (setup.py)
    with pyedflib.EdfReader(str(path)) as edf:
        labels = edf.getSignalLabels()
        idx = [i for i, s in enumerate(labels) if channel in s]
        if not idx:
            raise SystemExit(f"No channel containing '{channel}' in {labels}")
        fs = edf.getSampleFrequency(idx[0])
        sig = edf.readSignal(idx[0])
        print(f"channel: {labels[idx[0]]!r} at {fs} Hz, {len(sig) / fs / 60:.1f} min", file=sys.stderr)
        if reference:
            ridx = [i for i, s in enumerate(labels) if reference in s]
            if not ridx:
                raise SystemExit(f"No channel containing '{reference}' in {labels}")
            if edf.getSampleFrequency(ridx[0]) != fs:
                raise SystemExit("Reference channel has a different sampling rate")
            sig = sig - edf.readSignal(ridx[0])
            print(f"re-referenced to {labels[ridx[0]]!r}", file=sys.stderr)
    return sig, fs


def selftest():
    coe = Path(__file__).resolve().parents[2] / "portiloop_software" / "matlab" / "old_matlab" / "coef30.coe"
    ok = True
    if coe.is_file():
        txt = coe.read_text().split("COEFDATA=")[1].replace(";", "")
        ref = np.array([float(v) for v in txt.replace("\n", "").split(",") if v.strip()])
        same = len(ref) == len(FIR_COEFS) and np.array_equal(ref, FIR_COEFS)
        print(f"FIR coefficients identical to {coe.name}: {same}")
        ok &= same
    else:
        print("coef30.coe not found (run from a clone of the repository to compare)")
    b, a = notch_coefficients()
    f = np.array([1.0, 13.0, 30.0, 35.0, 50.0, 59.5, 60.0, 60.5])
    z = np.exp(-1j * 2 * np.pi * f / FS_PORTILOOP)
    hfir = np.polyval(FIR_COEFS[::-1], z)
    hn = (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)
    for fi, g1, g2 in zip(f, np.maximum(20 * np.log10(abs(hfir) + 1e-300), -200), np.maximum(20 * np.log10(abs(hn) + 1e-300), -200)):
        print(f"{fi:5.1f} Hz   FIR {g1:7.2f} dB   notch {g2:7.2f} dB")  # -200 dB means "removed"
    x = np.random.default_rng(0).normal(size=5000)
    y_vec, _, _ = online_standardize(x)
    # reference loop, transcribed line by line from the MATLAB script
    y_loop = x.copy(); m = x[0]; v = 0.0
    for i in range(1, len(x)):
        d = x[i] - m; m += 0.1 * d; v = (1 - 0.001) * (v + 0.001 * d * d)
        y_loop[i] = (x[i] - m) / math.sqrt(v)
    err = np.max(abs(y_vec - y_loop))
    print(f"vectorized vs loop standardization, max |difference| = {err:.2e}")
    ok &= err < 1e-9
    print("SELFTEST", "OK" if ok else "FAILED")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="text file (one sample per line) or .edf file")
    p.add_argument("-o", "--output", help="output text file (default: preprocessed_<input name>.txt)")
    p.add_argument("--fs", type=float, help="sampling rate of a text input in Hz (EDF files carry their own)")
    p.add_argument("--column", type=int, default=0, help="column to read in a multi-column text file (default 0)")
    p.add_argument("--channel", default="C3", help="EDF: substring of the channel label (default C3)")
    p.add_argument("--reference", default=None, help="EDF: substring of a reference channel to subtract (e.g. A2)")
    p.add_argument("--no-notch", action="store_true", help="skip the 60 Hz notch (NOT what the device does)")
    p.add_argument("--alpha-avg", type=float, default=0.1)
    p.add_argument("--alpha-var", type=float, default=0.001)
    p.add_argument("--compensate-delay", type=int, nargs="?", const=FIR_DELAY, default=0,
                   help="remove the FIR delay offline like the MATLAB script (default shift: 10 samples)")
    p.add_argument("--selftest", action="store_true", help="check coefficients and the standardization, then exit")
    args = p.parse_args(argv)

    if args.selftest:
        return 0 if selftest() else 1
    if not args.input:
        p.error("an input file is required")
    path = Path(args.input)
    if path.suffix.lower() in (".edf", ".bdf"):
        raw, fs = load_edf(path, args.channel, args.reference)
    else:
        if not args.fs:
            p.error("--fs is required for text input")
        raw, fs = load_text(path, args.column), args.fs
    out = preprocess(raw, fs, notch=not args.no_notch, alpha_avg=args.alpha_avg,
                     alpha_var=args.alpha_var, compensate_delay=args.compensate_delay)
    dst = Path(args.output) if args.output else path.with_name("preprocessed_" + path.stem + ".txt")
    np.savetxt(dst, out["standardized"].astype(np.float32), fmt="%.7g")
    n = len(out["standardized"])
    print(f"read {len(raw)} samples at {fs:g} Hz -> wrote {n} samples at 250 Hz ({n / 250 / 60:.2f} min) to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
