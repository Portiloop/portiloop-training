/*
 * portiloop.js: shared, dependency-free library for the Portiloop tutorials.
 *
 * Exposes a single global `PL` (also exported for Node via module.exports) with:
 *   - PL.rng(seed)                      deterministic random numbers (uniform, gaussian)
 *   - PL.synth.nrem(opts)               synthetic NREM-2 EEG with labelled sleep spindles
 *   - PL.filters.*                      the exact Portiloop online filter chain + helpers
 *   - PL.OnlineStandardizer             the exponential moving standardization of the Portiloop
 *   - PL.preprocess(raw)                FIR lowpass -> 60 Hz notch -> online standardization
 *   - PL.PortiloopModel                 the real pre-trained spindle detector, running in JS
 *   - PL.metrics.*                      sample-level and event-level precision / recall / F1
 *   - PL.plot(canvas, spec)             small canvas plotting helper (time series, regions, markers)
 *   - PL.ui.*                           tiny helpers to wire sliders to outputs
 *
 * Sources of truth (all in this repository):
 *   FIR coefficients: portiloop_software/matlab/old_matlab/coef30.coe, which is byte-identical to the
 *     FIR stored inside portiloop_software/matlab/filter_lp.slx (21 taps, least squares lowpass).
 *   Notch: MATLAB `iirnotch(60/(250/2), 1/(250/2), 1)` (stored in filter_lp.slx, gain 0.9936459).
 *   Standardization: portiloop_software/matlab/preprocess_unannotated_data.m (alpha_av = 0.1, alpha_var = 0.001).
 *   Network: PortiloopNetwork in portiloop_software/portiloop_python/ANN/portiloop_detector_training.py,
 *     weights exported by docs/tools/export_model_json.py.
 */
(function (root) {
  "use strict";

  const PL = {};
  PL.FS = 250; // Portiloop sampling frequency (Hz)

  /* ------------------------------------------------------------------ random numbers */

  PL.rng = function (seed) {
    let a = (seed >>> 0) || 1;
    const uniform = function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const gaussian = function () {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = uniform();
      v = uniform();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uniform, gaussian, range: (lo, hi) => lo + (hi - lo) * uniform() };
  };

  /* ------------------------------------------------------------------ synthetic EEG */

  PL.synth = {};

  /**
   * Synthetic NREM stage 2 EEG (arbitrary units comparable to microvolts) with labelled spindles.
   * This is a didactic signal, not a physiological model: pink-ish background, slow waves,
   * optional K-complexes, waxing and waning 11 to 16 Hz spindles, optional 60 Hz line noise.
   *
   * opts: { duration_s=30, fs=250, seed=1, spindleRate_per_min=6, spindleAmp=[25,45],
   *         spindleFreq=[11.5,15], spindleDur=[0.5,1.8], backgroundAmp=20, slowWaveAmp=25,
   *         kComplexes=true, lineNoiseAmp=0, spindleTimes=null (array of start times in s) }
   * returns { fs, t: Float64Array, signal: Float64Array, label: Uint8Array, spindles: [{start,end,freq,amp}] }
   */
  PL.synth.nrem = function (opts) {
    const o = Object.assign({
      duration_s: 30, fs: PL.FS, seed: 1, spindleRate_per_min: 6, spindleAmp: [25, 45],
      spindleFreq: [11.5, 15], spindleDur: [0.5, 1.8], backgroundAmp: 20, slowWaveAmp: 25,
      kComplexes: true, lineNoiseAmp: 0, spindleTimes: null,
    }, opts || {});
    const R = PL.rng(o.seed);
    const n = Math.round(o.duration_s * o.fs);
    const t = new Float64Array(n);
    const x = new Float64Array(n);
    const label = new Uint8Array(n);
    for (let i = 0; i < n; i++) t[i] = i / o.fs;

    // Background: sum of AR(1) processes (approximately 1/f) plus white noise.
    const poles = [0.999, 0.99, 0.95, 0.8];
    const gains = [0.035, 0.11, 0.3, 0.45];
    const state = [0, 0, 0, 0];
    for (let i = 0; i < n; i++) {
      let v = 0;
      for (let k = 0; k < poles.length; k++) {
        state[k] = poles[k] * state[k] + gains[k] * R.gaussian();
        v += state[k];
      }
      x[i] = o.backgroundAmp * (0.55 * v + 0.12 * R.gaussian());
    }
    // Slow oscillations (0.5 to 2 Hz) with slowly varying phase.
    let ph = R.range(0, 6.28), f = 0.9;
    for (let i = 0; i < n; i++) {
      if (i % o.fs === 0) f = Math.min(2, Math.max(0.5, f + 0.15 * R.gaussian()));
      ph += 2 * Math.PI * f / o.fs;
      x[i] += 0.5 * o.slowWaveAmp * Math.sin(ph);
    }
    // K-complexes: a sharp negative wave followed by a positive one.
    if (o.kComplexes) {
      const nk = Math.max(0, Math.round(o.duration_s / 30 * (1 + R.uniform())));
      for (let k = 0; k < nk; k++) {
        const c = R.range(1, o.duration_s - 1);
        for (let i = 0; i < n; i++) {
          const d = t[i] - c;
          if (Math.abs(d) > 1) continue;
          x[i] += -3.2 * o.slowWaveAmp * Math.exp(-(d * d) / 0.012) + 1.6 * o.slowWaveAmp * Math.exp(-((d - 0.35) ** 2) / 0.04);
        }
      }
    }
    // Spindles.
    const spindles = [];
    let starts = o.spindleTimes;
    if (!starts) {
      starts = [];
      const meanGap = 60 / Math.max(0.01, o.spindleRate_per_min);
      let cur = R.range(0.8, Math.min(meanGap, 4));
      while (cur < o.duration_s - 2.2) {
        starts.push(cur);
        cur += 2.5 + -Math.log(1 - R.uniform()) * Math.max(0.1, meanGap - 2.5);
      }
    }
    for (const s0 of starts) {
      const dur = R.range(o.spindleDur[0], o.spindleDur[1]);
      const fr = R.range(o.spindleFreq[0], o.spindleFreq[1]);
      const amp = R.range(o.spindleAmp[0], o.spindleAmp[1]);
      const phase = R.range(0, 6.28);
      const i0 = Math.max(0, Math.round(s0 * o.fs));
      const i1 = Math.min(n, Math.round((s0 + dur) * o.fs));
      for (let i = i0; i < i1; i++) {
        const u = (i - i0) / Math.max(1, i1 - i0);
        const env = Math.pow(Math.sin(Math.PI * u), 1.5) * (1 - 0.25 * u); // waxing then waning
        const chirp = fr * (1 - 0.04 * u);                                  // slight deceleration
        x[i] += amp * env * Math.sin(phase + 2 * Math.PI * chirp * (i - i0) / o.fs);
        label[i] = 1;
      }
      spindles.push({ start: i0 / o.fs, end: i1 / o.fs, freq: fr, amp: amp });
    }
    if (o.lineNoiseAmp > 0) {
      for (let i = 0; i < n; i++) x[i] += o.lineNoiseAmp * Math.sin(2 * Math.PI * 60 * t[i]);
    }
    return { fs: o.fs, t, signal: x, label, spindles };
  };

  /* ------------------------------------------------------------------ filters */

  PL.filters = {};

  /** 21-tap least squares FIR lowpass used by the Portiloop (coef30.coe / filter_lp.slx). */
  PL.filters.PORTILOOP_FIR = [
    0.001623780150148095, 0.014988684599373742, 0.021287595318265636, 0.007349500393709579,
    -0.02512751571711218, -0.05221050735982245, -0.039273839505489905, 0.033021568427940004,
    0.147606943281569, 0.2540002520345056, 0.2973308763988834, 0.2540002520345056,
    0.147606943281569, 0.033021568427940004, -0.039273839505489905, -0.05221050735982245,
    -0.02512751571711218, 0.007349500393709579, 0.021287595318265636, 0.014988684599373742,
    0.001623780150148095,
  ];

  /**
   * MATLAB iirnotch(w0, bw, ab) (normalized frequencies, ab in dB), returns {b:[3], a:[3]}.
   * The Portiloop uses iirnotch(60/125, 1/125, 1).
   */
  PL.filters.iirnotch = function (f0, bw, fs, abDb) {
    const w0 = f0 / (fs / 2), b = bw / (fs / 2);
    const Gb = Math.pow(10, -Math.abs(abDb === undefined ? 3.0103 : abDb) / 20);
    const beta = (Math.sqrt(1 - Gb * Gb) / Gb) * Math.tan(b * Math.PI / 2);
    const gain = 1 / (1 + beta);
    const c = Math.cos(w0 * Math.PI);
    return { b: [gain, -2 * gain * c, gain], a: [1, -2 * gain * c, 2 * gain - 1] };
  };
  PL.filters.PORTILOOP_NOTCH = PL.filters.iirnotch(60, 1, PL.FS, 1);

  /** RBJ cookbook band-pass biquad (constant 0 dB peak gain). */
  PL.filters.bandpass = function (fc, Q, fs) {
    const w = 2 * Math.PI * fc / fs, alpha = Math.sin(w) / (2 * Q), cw = Math.cos(w);
    const a0 = 1 + alpha;
    return { b: [alpha / a0, 0, -alpha / a0], a: [1, -2 * cw / a0, (1 - alpha) / a0] };
  };

  /** Streaming FIR filter: f = new PL.filters.FIR(coefs); y = f.step(x). */
  PL.filters.FIR = function (coefs) {
    this.c = coefs.slice();
    this.buf = new Float64Array(coefs.length);
    this.pos = 0;
  };
  PL.filters.FIR.prototype.step = function (x) {
    const L = this.c.length;
    this.buf[this.pos] = x;
    let y = 0, j = this.pos;
    for (let k = 0; k < L; k++) {
      y += this.c[k] * this.buf[j];
      j = j === 0 ? L - 1 : j - 1;
    }
    this.pos = (this.pos + 1) % L;
    return y;
  };
  PL.filters.FIR.prototype.reset = function () { this.buf.fill(0); this.pos = 0; };

  /** Streaming biquad, direct form II (as the Portiloop notch, dfilt.df2). */
  PL.filters.Biquad = function (ba) {
    this.b = ba.b; this.a = ba.a; this.w1 = 0; this.w2 = 0;
  };
  PL.filters.Biquad.prototype.step = function (x) {
    const w0 = x - this.a[1] * this.w1 - this.a[2] * this.w2;
    const y = this.b[0] * w0 + this.b[1] * this.w1 + this.b[2] * this.w2;
    this.w2 = this.w1; this.w1 = w0;
    return y;
  };
  PL.filters.Biquad.prototype.reset = function () { this.w1 = 0; this.w2 = 0; };

  /** Apply a streaming filter object (with .step) to a whole array. */
  PL.filters.apply = function (filter, x) {
    const y = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = filter.step(x[i]);
    return y;
  };

  /**
   * Magnitude response (dB) of a filter given as {b, a} (FIR: a = [1]) at the frequencies `freqs` (Hz).
   */
  PL.filters.freqz = function (ba, freqs, fs) {
    const b = ba.b, a = ba.a || [1];
    return freqs.map(function (f) {
      const w = 2 * Math.PI * f / fs;
      let nr = 0, ni = 0, dr = 0, di = 0;
      for (let k = 0; k < b.length; k++) { nr += b[k] * Math.cos(-w * k); ni += b[k] * Math.sin(-w * k); }
      for (let k = 0; k < a.length; k++) { dr += a[k] * Math.cos(-w * k); di += a[k] * Math.sin(-w * k); }
      const mag = Math.sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
      return 20 * Math.log10(mag + 1e-12);
    });
  };

  /**
   * Sigma-band envelope (for didactic use): band-pass 11-16 Hz, rectify, smooth with a moving RMS.
   * Causal, so it lags the signal by roughly half of win_s.
   */
  PL.filters.sigmaEnvelope = function (x, fs, win_s) {
    fs = fs || PL.FS; win_s = win_s || 0.2;
    const bp1 = new PL.filters.Biquad(PL.filters.bandpass(13.5, 2.2, fs));
    const bp2 = new PL.filters.Biquad(PL.filters.bandpass(13.5, 2.2, fs));
    const y = PL.filters.apply(bp2, PL.filters.apply(bp1, x));
    const W = Math.max(1, Math.round(win_s * fs));
    const env = new Float64Array(x.length);
    let acc = 0;
    for (let i = 0; i < x.length; i++) {
      acc += y[i] * y[i];
      if (i >= W) acc -= y[i - W] * y[i - W];
      env[i] = Math.sqrt(Math.max(0, acc) / Math.min(i + 1, W)) * Math.SQRT2;
    }
    return { band: y, envelope: env };
  };

  /* ------------------------------------------------------------------ online standardization */

  /**
   * Exponential moving standardization, transcribed from preprocess_unannotated_data.m:
   *   delta = x - mean; mean += alpha_av * delta;
   *   var = (1 - alpha_var) * (var + alpha_var * delta^2);  y = (x - mean) / sqrt(var)
   * The first sample initializes the mean and is passed through unchanged (as in the MATLAB script).
   */
  PL.OnlineStandardizer = function (alphaAvg, alphaVar) {
    this.alphaAvg = alphaAvg === undefined ? 0.1 : alphaAvg;
    this.alphaVar = alphaVar === undefined ? 0.001 : alphaVar;
    this.reset();
  };
  PL.OnlineStandardizer.prototype.reset = function () { this.mean = null; this.variance = 0; };
  PL.OnlineStandardizer.prototype.step = function (x) {
    if (this.mean === null) { this.mean = x; return x; }
    const delta = x - this.mean;
    this.mean += this.alphaAvg * delta;
    this.variance = (1 - this.alphaVar) * (this.variance + this.alphaVar * delta * delta);
    const sd = Math.sqrt(this.variance);
    return sd > 0 ? (x - this.mean) / sd : 0;
  };

  /**
   * Full Portiloop preprocessing of a raw 250 Hz signal: FIR lowpass, 60 Hz notch, online standardization.
   * Returns { lowpass, notched, standardized, delay_samples }.
   * delay_samples = 10 is the group delay of the 21-tap linear-phase FIR (the MATLAB script
   * compensates it offline by dropping the first samples; a real-time device cannot).
   */
  PL.preprocess = function (raw, opts) {
    const o = Object.assign({ alphaAvg: 0.1, alphaVar: 0.001, notch: true }, opts || {});
    const fir = new PL.filters.FIR(PL.filters.PORTILOOP_FIR);
    const notch = new PL.filters.Biquad(PL.filters.PORTILOOP_NOTCH);
    const std = new PL.OnlineStandardizer(o.alphaAvg, o.alphaVar);
    const n = raw.length;
    const lowpass = new Float64Array(n), notched = new Float64Array(n), standardized = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      lowpass[i] = fir.step(raw[i]);
      notched[i] = o.notch ? notch.step(lowpass[i]) : lowpass[i];
      standardized[i] = std.step(notched[i]);
    }
    return { lowpass, notched, standardized, delay_samples: (PL.filters.PORTILOOP_FIR.length - 1) / 2 };
  };

  /** Linear resampling (didactic; MATLAB `resample` uses a polyphase anti-aliasing filter). */
  PL.resampleLinear = function (x, fsIn, fsOut) {
    const n = Math.floor(x.length * fsOut / fsIn);
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = i * fsIn / fsOut, k = Math.floor(p), r = p - k;
      y[i] = k + 1 < x.length ? x[k] * (1 - r) + x[k + 1] * r : x[x.length - 1];
    }
    return y;
  };

  /* ------------------------------------------------------------------ the pre-trained network */

  /**
   * PortiloopModel: forward pass of PortiloopNetwork (1 input, RNN=True) in plain JavaScript.
   *   const m = await PL.PortiloopModel.load('../assets/models/portiloop_spindle_v4_0.json');
   *   let h = m.zeroState();
   *   const r = m.step(windowOf54Samples, h); // r.y in [0,1], r.h new hidden state
   * Architecture: [Conv1d(k=7) -> ReLU -> MaxPool1d(k=7)] x 3 -> flatten -> GRU(hidden 7) -> Linear -> sigmoid.
   * Optional `trace` object receives intermediate activations for visualization.
   */
  PL.PortiloopModel = function (json) {
    this.json = json;
    this.cfg = json.config;
    this.windowSize = Math.trunc(json.config.window_size_s * json.config.fe); // Python int() truncates: 0.218 * 250 -> 54
    this.hidden = json.config.hidden_size;
    this.nLayers = json.config.nb_rnn_layers;
    const W = json.weights;
    const get = (k) => ({ shape: W[k].shape, data: Float32Array.from(W[k].data) });
    this.convs = [get("first_layer_input1.conv.weight")];
    this.convBiases = [get("first_layer_input1.conv.bias")];
    for (let i = 0; i < json.config.nb_conv_layers - 1; i++) {
      this.convs.push(get("seq_input1." + i + ".conv.weight"));
      this.convBiases.push(get("seq_input1." + i + ".conv.bias"));
    }
    this.gru = [];
    for (let l = 0; l < this.nLayers; l++) {
      this.gru.push({
        wih: get("gru_input1.weight_ih_l" + l), whh: get("gru_input1.weight_hh_l" + l),
        bih: get("gru_input1.bias_ih_l" + l), bhh: get("gru_input1.bias_hh_l" + l),
      });
    }
    this.fcW = get("fc.weight"); this.fcB = get("fc.bias");
    this.kPool = json.config.kernel_pool;
    this.nParameters = json.n_parameters;
  };

  PL.PortiloopModel.load = async function (url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not load model " + url + " (" + res.status + ")");
    return new PL.PortiloopModel(await res.json());
  };

  PL.PortiloopModel.prototype.zeroState = function () {
    return new Float32Array(this.nLayers * this.hidden);
  };

  function conv1dReluPool(input, Cin, L, w, b, kPool) {
    const Cout = w.shape[0], K = w.shape[2];
    const Lc = L - K + 1;
    const conv = new Float32Array(Cout * Lc);
    for (let co = 0; co < Cout; co++) {
      for (let t = 0; t < Lc; t++) {
        let s = b.data[co];
        for (let ci = 0; ci < Cin; ci++) {
          const wo = (co * Cin + ci) * K, io = ci * L + t;
          for (let k = 0; k < K; k++) s += w.data[wo + k] * input[io + k];
        }
        conv[co * Lc + t] = s > 0 ? s : 0;
      }
    }
    const Lp = Lc - kPool + 1; // stride 1, no padding, dilation 1
    const out = new Float32Array(Cout * Lp);
    for (let co = 0; co < Cout; co++) {
      for (let t = 0; t < Lp; t++) {
        let m = -Infinity;
        for (let k = 0; k < kPool; k++) { const v = conv[co * Lc + t + k]; if (v > m) m = v; }
        out[co * Lp + t] = m;
      }
    }
    return { out, C: Cout, L: Lp };
  }

  const sigmoid = (v) => 1 / (1 + Math.exp(-v));

  /**
   * One inference step. `x`: array-like of windowSize standardized samples (oldest first).
   * `h`: Float32Array state from zeroState() or a previous step. Returns { y, h, logit }.
   */
  PL.PortiloopModel.prototype.step = function (x, h, trace) {
    if (x.length !== this.windowSize) throw new Error("window must have " + this.windowSize + " samples");
    let cur = Float32Array.from(x), C = 1, L = this.windowSize;
    if (trace) trace.featureMaps = [];
    for (let i = 0; i < this.convs.length; i++) {
      const r = conv1dReluPool(cur, C, L, this.convs[i], this.convBiases[i], this.kPool);
      cur = r.out; C = r.C; L = r.L;
      if (trace) trace.featureMaps.push({ C, L, data: cur });
    }
    // flatten (channel-major, as torch.flatten on (C, L)) and run the GRU layers
    let inp = cur;
    const H = this.hidden;
    const hNew = new Float32Array(this.nLayers * H);
    for (let l = 0; l < this.nLayers; l++) {
      const g = this.gru[l], In = g.wih.shape[1];
      const hp = h.subarray(l * H, (l + 1) * H);
      const gi = new Float32Array(3 * H), gh = new Float32Array(3 * H);
      for (let r = 0; r < 3 * H; r++) {
        let s = g.bih.data[r];
        const o = r * In;
        for (let c = 0; c < In; c++) s += g.wih.data[o + c] * inp[c];
        gi[r] = s;
        let s2 = g.bhh.data[r];
        for (let c = 0; c < H; c++) s2 += g.whh.data[r * H + c] * hp[c];
        gh[r] = s2;
      }
      const out = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const rg = sigmoid(gi[j] + gh[j]);              // reset gate
        const zg = sigmoid(gi[H + j] + gh[H + j]);      // update gate
        const ng = Math.tanh(gi[2 * H + j] + rg * gh[2 * H + j]); // candidate
        out[j] = (1 - zg) * ng + zg * hp[j];
      }
      hNew.set(out, l * H);
      inp = out;
    }
    let logit = this.fcB.data[0];
    for (let j = 0; j < H; j++) logit += this.fcW.data[j] * inp[j];
    if (trace) { trace.hidden = Float32Array.from(inp); trace.logit = logit; }
    return { y: sigmoid(logit), h: hNew, logit };
  };

  /**
   * Run the model over a whole standardized signal the way the Portiloop does online:
   * a new window every `networkStride` samples, with `nbParallel` = floor(seqStride / networkStride) interleaved
   * hidden states used round-robin (8 for the published model: seq_stride 42 samples, network stride 5).
   * With this uniform schedule each hidden state sees windows nbParallel * networkStride = 40 samples apart,
   * a close approximation of the 42 used in training; the simulation notebook and the device instead use
   * gaps of 5,5,5,6,... so that each state sees exactly 42 (see tutorial 08).
   * Returns { idx: Int32Array (sample index of the last sample of each window), y: Float32Array }.
   */
  PL.PortiloopModel.prototype.runStream = function (signal, opts) {
    const seqStride = Math.trunc(this.cfg.seq_stride_s * this.cfg.fe); // int(0.17 * 250) = 42 in the Python code
    const o = Object.assign({ networkStride: 5 }, opts || {});
    const nbParallel = o.nbParallel || Math.max(1, Math.floor(seqStride / o.networkStride));
    const W = this.windowSize;
    const states = [];
    for (let k = 0; k < nbParallel; k++) states.push(this.zeroState());
    const idx = [], ys = [];
    let k = 0;
    for (let end = W - 1; end < signal.length; end += o.networkStride) {
      const r = this.step(signal.subarray ? signal.subarray(end - W + 1, end + 1) : signal.slice(end - W + 1, end + 1), states[k]);
      states[k] = r.h;
      idx.push(end); ys.push(r.y);
      k = (k + 1) % nbParallel;
    }
    return { idx: Int32Array.from(idx), y: Float32Array.from(ys), nbParallel, networkStride: o.networkStride };
  };

  /* ------------------------------------------------------------------ metrics */

  PL.metrics = {};

  /** Sample-level confusion counts from boolean-like arrays of equal length. */
  PL.metrics.confusion = function (pred, truth) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < pred.length; i++) {
      const p = !!pred[i], t = !!truth[i];
      if (p && t) tp++; else if (p && !t) fp++; else if (!p && t) fn++; else tn++;
    }
    return { tp, fp, fn, tn };
  };

  /** precision, recall, F1 from counts (same epsilon convention as get_metrics in the training code). */
  PL.metrics.prf = function (c) {
    const eps = 1e-7;
    const precision = c.tp / (c.tp + c.fp + eps);
    const recall = c.tp / (c.tp + c.fn + eps);
    const f1 = 2 * precision * recall / (precision + recall + eps);
    return { precision, recall, f1 };
  };

  /** Convert a boolean-like array to events [{start, end}] in samples (end exclusive). */
  PL.metrics.toEvents = function (mask) {
    const ev = [];
    let s = -1;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] && s < 0) s = i;
      if (!mask[i] && s >= 0) { ev.push({ start: s, end: i }); s = -1; }
    }
    if (s >= 0) ev.push({ start: s, end: mask.length });
    return ev;
  };

  /**
   * Event-level matching: a detection is a true positive if it overlaps a true event with
   * intersection-over-union >= iouThreshold (default 0.2, a common convention for spindles).
   */
  PL.metrics.eventMatch = function (detEvents, trueEvents, iouThreshold) {
    const thr = iouThreshold === undefined ? 0.2 : iouThreshold;
    const used = new Array(trueEvents.length).fill(false);
    let tp = 0;
    for (const d of detEvents) {
      let best = -1, bestIou = 0;
      trueEvents.forEach(function (t, j) {
        if (used[j]) return;
        const inter = Math.max(0, Math.min(d.end, t.end) - Math.max(d.start, t.start));
        const uni = Math.max(d.end, t.end) - Math.min(d.start, t.start);
        const iou = uni > 0 ? inter / uni : 0;
        if (iou > bestIou) { bestIou = iou; best = j; }
      });
      if (best >= 0 && bestIou >= thr) { used[best] = true; tp++; }
    }
    return { tp, fp: detEvents.length - tp, fn: trueEvents.length - tp };
  };

  /* ------------------------------------------------------------------ plotting */

  function cssVar(name, fallback) {
    if (typeof getComputedStyle === "undefined") return fallback;
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }
  PL.color = function (name) {
    const map = {
      signal: ["--plot-signal", "#2b6cb0"], model: ["--plot-model", "#c05621"],
      label: ["--plot-label", "rgba(56,161,105,0.22)"], stim: ["--plot-stim", "#d53f8c"],
      grid: ["--plot-grid", "rgba(128,128,128,0.18)"], axis: ["--plot-axis", "#666"],
      text: ["--fg-muted", "#555"], accent: ["--accent", "#2b6cb0"], warn: ["--warn", "#b7791f"],
      c1: ["--plot-c1", "#2b6cb0"], c2: ["--plot-c2", "#c05621"], c3: ["--plot-c3", "#2f855a"],
      c4: ["--plot-c4", "#6b46c1"], c5: ["--plot-c5", "#b83280"],
    };
    const m = map[name];
    return m ? cssVar(m[0], m[1]) : name;
  };

  /**
   * Draw on a <canvas>. The canvas is sized to its CSS box (set height in CSS or via spec.height).
   * spec = {
   *   xRange:[x0,x1], yRange:[y0,y1] (auto if omitted), xLabel, yLabel, title,
   *   series:[{x?:array, y:array, color, width, dash:[..], label, step:bool, points:bool}],
   *   regions:[{x0, x1, color, label?}],   vertical shaded bands (e.g. spindle labels)
   *   hlines:[{y, color, dash, label}],    horizontal reference lines (e.g. threshold)
   *   markers:[{x, color, label, shape:'line'|'triangle'}],  vertical markers (e.g. stimulations)
   *   xTicks: n, yTicks: n, legend: true, margin:{l,r,t,b}
   * }
   * Returns a transform {toPx(x,y), fromPx(px,py)} for interaction.
   */
  PL.plot = function (canvas, spec) {
    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    const cssW = canvas.clientWidth || 600;
    const cssH = spec.height || canvas.clientHeight || 220;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const m = Object.assign({ l: 52, r: 24, t: spec.title ? 26 : 12, b: spec.xLabel ? 40 : 24 }, spec.margin || {});
    const W = cssW - m.l - m.r, H = cssH - m.t - m.b;
    const series = spec.series || [];
    let x0, x1, y0, y1;
    if (spec.xRange) { x0 = spec.xRange[0]; x1 = spec.xRange[1]; }
    else {
      x0 = Infinity; x1 = -Infinity;
      series.forEach(function (s) {
        const xs = s.x || null, n = s.y.length;
        const a = xs ? xs[0] : 0, b = xs ? xs[n - 1] : n - 1;
        x0 = Math.min(x0, a); x1 = Math.max(x1, b);
      });
      if (!isFinite(x0)) { x0 = 0; x1 = 1; }
    }
    if (spec.yRange) { y0 = spec.yRange[0]; y1 = spec.yRange[1]; }
    else {
      y0 = Infinity; y1 = -Infinity;
      series.forEach(function (s) {
        const xs = s.x;
        for (let i = 0; i < s.y.length; i++) {
          const xv = xs ? xs[i] : i;
          if (xv < x0 || xv > x1) continue;
          const v = s.y[i];
          if (isFinite(v)) { if (v < y0) y0 = v; if (v > y1) y1 = v; }
        }
      });
      if (!isFinite(y0)) { y0 = 0; y1 = 1; }
      if (y0 === y1) { y0 -= 1; y1 += 1; }
      const pad = 0.06 * (y1 - y0); y0 -= pad; y1 += pad;
    }
    const X = (x) => m.l + (x - x0) / (x1 - x0) * W;
    const Y = (y) => m.t + (1 - (y - y0) / (y1 - y0)) * H;
    const font = "12px " + cssVar("--font-sans", "system-ui, sans-serif");
    ctx.font = font;

    // regions
    (spec.regions || []).forEach(function (r) {
      const a = Math.max(X(r.x0), m.l), b = Math.min(X(r.x1), m.l + W);
      if (b <= a) return;
      ctx.fillStyle = r.color || PL.color("label");
      ctx.fillRect(a, m.t, b - a, H);
    });
    // grid and ticks
    const nice = function (lo, hi, n) {
      const span = hi - lo, step0 = span / Math.max(1, n);
      const mag = Math.pow(10, Math.floor(Math.log10(step0)));
      const err = step0 / mag;
      const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
      const out = [];
      for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
      return { ticks: out, step };
    };
    const fmt = (v, step) => { const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9)); return v.toFixed(Math.min(d, 4)); };
    ctx.strokeStyle = PL.color("grid"); ctx.lineWidth = 1; ctx.fillStyle = PL.color("text");
    const xt = nice(x0, x1, spec.xTicks || Math.max(2, Math.floor(W / 90)));
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    xt.ticks.forEach(function (v) {
      const px = X(v); ctx.beginPath(); ctx.moveTo(px, m.t); ctx.lineTo(px, m.t + H); ctx.stroke();
      ctx.fillText(fmt(v, xt.step), px, m.t + H + 4);
    });
    const yt = nice(y0, y1, spec.yTicks || Math.max(2, Math.floor(H / 45)));
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    yt.ticks.forEach(function (v) {
      const py = Y(v); ctx.beginPath(); ctx.moveTo(m.l, py); ctx.lineTo(m.l + W, py); ctx.stroke();
      ctx.fillText(fmt(v, yt.step), m.l - 6, py);
    });
    ctx.strokeStyle = PL.color("axis");
    ctx.strokeRect(m.l + 0.5, m.t + 0.5, W, H);
    if (spec.xLabel) { ctx.textAlign = "center"; ctx.textBaseline = "bottom"; ctx.fillText(spec.xLabel, m.l + W / 2, cssH - 2); }
    if (spec.yLabel) {
      ctx.save(); ctx.translate(12, m.t + H / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(spec.yLabel, 0, 0); ctx.restore();
    }
    if (spec.title) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.font = "600 " + font; ctx.fillText(spec.title, m.l, 6); ctx.font = font; }

    ctx.save();
    ctx.beginPath(); ctx.rect(m.l, m.t, W, H); ctx.clip();
    (spec.hlines || []).forEach(function (h) {
      ctx.strokeStyle = h.color || PL.color("warn"); ctx.setLineDash(h.dash || [5, 4]); ctx.lineWidth = h.width || 1.2;
      ctx.beginPath(); ctx.moveTo(m.l, Y(h.y)); ctx.lineTo(m.l + W, Y(h.y)); ctx.stroke(); ctx.setLineDash([]);
      if (h.label) { ctx.fillStyle = h.color || PL.color("warn"); ctx.textAlign = "right"; ctx.textBaseline = "bottom"; ctx.fillText(h.label, m.l + W - 4, Y(h.y) - 2); }
    });
    // series (decimated with min/max per pixel column for long signals)
    series.forEach(function (s) {
      const xs = s.x, ys = s.y, n = ys.length;
      ctx.strokeStyle = s.color || PL.color("signal"); ctx.lineWidth = s.width || 1.2;
      ctx.setLineDash(s.dash || []);
      if (s.points) {
        ctx.fillStyle = s.color || PL.color("signal");
        for (let i = 0; i < n; i++) {
          const xv = xs ? xs[i] : i; if (xv < x0 || xv > x1) continue;
          ctx.beginPath(); ctx.arc(X(xv), Y(ys[i]), s.radius || 2.5, 0, 2 * Math.PI); ctx.fill();
        }
        return;
      }
      ctx.beginPath();
      let started = false, lastPx = null, minY = 0, maxY = 0, prevY = null;
      const visibleCount = n; const dense = visibleCount > 2 * W;
      for (let i = 0; i < n; i++) {
        const xv = xs ? xs[i] : i;
        if (xv < x0 - (x1 - x0) * 0.01 || xv > x1 + (x1 - x0) * 0.01) continue;
        const v = ys[i]; if (!isFinite(v)) { started = false; continue; }
        const px = X(xv), py = Y(v);
        if (dense) {
          const col = Math.round(px);
          if (col === lastPx) { minY = Math.min(minY, py); maxY = Math.max(maxY, py); continue; }
          if (lastPx !== null) { ctx.lineTo(lastPx, minY); ctx.lineTo(lastPx, maxY); }
          lastPx = col; minY = maxY = py;
          if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        } else if (s.step && started) {
          ctx.lineTo(px, prevY); ctx.lineTo(px, py);
        } else {
          if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
        }
        prevY = py;
      }
      ctx.stroke(); ctx.setLineDash([]);
    });
    (spec.markers || []).forEach(function (mk) {
      const px = X(mk.x); if (px < m.l || px > m.l + W) return;
      ctx.strokeStyle = ctx.fillStyle = mk.color || PL.color("stim");
      if (mk.shape === "triangle") {
        ctx.beginPath(); ctx.moveTo(px, m.t + 2); ctx.lineTo(px - 5, m.t - 6 + 2 + 12); ctx.lineTo(px + 5, m.t - 6 + 2 + 12); ctx.closePath(); ctx.fill();
      } else {
        ctx.lineWidth = mk.width || 1.5; ctx.beginPath(); ctx.moveTo(px, m.t); ctx.lineTo(px, m.t + H); ctx.stroke();
      }
      if (mk.label) { ctx.textAlign = "left"; ctx.textBaseline = "top"; ctx.fillText(mk.label, px + 3, m.t + 2); }
    });
    ctx.restore();

    if (spec.legend !== false) {
      const items = series.filter((s) => s.label).map((s) => ({ label: s.label, color: s.color || PL.color("signal") }))
        .concat((spec.regions || []).filter((r) => r.label).slice(0, 1).map((r) => ({ label: r.label, color: r.color || PL.color("label"), box: true })));
      let lx = m.l + 8, ly = m.t + 8;
      ctx.textAlign = "left"; ctx.textBaseline = "middle";
      items.forEach(function (it) {
        const w = ctx.measureText(it.label).width;
        if (lx > m.l + 8 && lx + w + 30 > m.l + W) { lx = m.l + 8; ly += 18; } // wrap on narrow plots
        ctx.fillStyle = cssVar("--plot-legend-bg", "rgba(255,255,255,0.8)");
        ctx.fillRect(lx - 4, ly - 8, w + 30, 16);
        if (it.box) { ctx.fillStyle = it.color; ctx.fillRect(lx, ly - 5, 16, 10); }
        else { ctx.strokeStyle = it.color; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke(); }
        ctx.fillStyle = PL.color("text"); ctx.fillText(it.label, lx + 21, ly);
        lx += w + 38;
      });
    }
    return {
      toPx: (x, y) => [X(x), Y(y)],
      fromPx: (px, py) => [x0 + (px - m.l) / W * (x1 - x0), y0 + (1 - (py - m.t) / H) * (y1 - y0)],
      box: { l: m.l, t: m.t, w: W, h: H },
    };
  };

  /** Redraw callback on resize and on light/dark theme change. */
  PL.onRedraw = function (fn) {
    if (typeof window === "undefined") return;
    let raf = null;
    const go = () => { if (raf) cancelAnimationFrame(raf); raf = requestAnimationFrame(fn); };
    window.addEventListener("resize", go);
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", go);
    document.addEventListener("pl-theme-change", go);
  };

  /* ------------------------------------------------------------------ UI helpers */

  PL.ui = {};
  /**
   * Bind a range/number/select input to a callback and an optional output element.
   * PL.ui.bind('#alpha', v => redraw(), v => v.toFixed(3))
   */
  PL.ui.bind = function (selector, onChange, format) {
    const el = typeof selector === "string" ? document.querySelector(selector) : selector;
    if (!el) throw new Error("PL.ui.bind: no element " + selector);
    const out = el.id ? document.querySelector('output[for="' + el.id + '"]') : null;
    const read = () => (el.type === "checkbox" ? el.checked : el.type === "range" || el.type === "number" ? parseFloat(el.value) : el.value);
    const update = () => { const v = read(); if (out) out.textContent = format ? format(v) : v; onChange(v); };
    // one event per interaction: 'input' for sliders and text/number fields, 'change' for checkboxes, radios and selects
    const discrete = el.tagName === "SELECT" || el.type === "checkbox" || el.type === "radio";
    el.addEventListener(discrete ? "change" : "input", update);
    if (out) out.textContent = format ? format(read()) : read();
    return read;
  };
  PL.ui.$ = (s, r) => (r || document).querySelector(s);
  PL.ui.$$ = (s, r) => Array.from((r || document).querySelectorAll(s));

  /** Load the published model once per page (cached promise). Path is relative to tutorials/. */
  let modelPromise = null;
  PL.loadModel = function (url) {
    if (!modelPromise) modelPromise = PL.PortiloopModel.load(url || "../assets/models/portiloop_spindle_v4_0.json");
    return modelPromise;
  };

  root.PL = PL;
  if (typeof module !== "undefined" && module.exports) module.exports = PL;
})(typeof window !== "undefined" ? window : globalThis);
