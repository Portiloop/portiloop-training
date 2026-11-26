# Portiloop tutorials (GitHub Pages site)

Interactive tutorials for neuroscientists on training, evaluating and deploying Portiloop sleep spindle detectors. The site is plain static HTML: no build step, no Jekyll (`.nojekyll` is present).

## Preview locally

The interactive widgets load the model with `fetch()`, which browsers block on `file://` pages, so serve the folder over HTTP:

```bash
python3 -m http.server 8000 -d docs
# then open http://localhost:8000
```

## Publish on GitHub Pages

In the repository on GitHub: **Settings → Pages → Build and deployment → Source: Deploy from a branch**, branch `main`, folder `/docs`. The site is then served at `https://<owner>.github.io/<repository>/` (for this repository: `https://portiloop.github.io/portiloop-training/`).

## Layout

```
docs/
├── index.html                  home page (cards are generated from the table of contents in nav.js)
├── tutorials/NN-name.html      one page per tutorial; _template.html shows every component
├── assets/css/style.css        shared styles (light and dark themes via CSS variables)
├── assets/js/nav.js            top bar, sidebar, prev/next, footer; the table of contents lives here
├── assets/js/portiloop.js      shared library: synthetic EEG, Portiloop filters, online standardization,
│                               the pre-trained network in JavaScript, metrics, plotting
├── assets/models/              pre-trained network exported to JSON for the browser
└── tools/                      scripts that produce and check the assets
```

## Adding a tutorial

1. Copy `tutorials/_template.html` to `tutorials/NN-short-name.html` and set `<body data-page="NN-short-name">`.
2. Add `["NN-short-name", "Title"]` to the `TOC` array in `assets/js/nav.js` (the sidebar, prev/next links and home-page cards update automatically) and a blurb in the `blurbs` object of `index.html`.

## The in-browser model

`assets/models/portiloop_spindle_v4_0.json` contains the weights of the published 1-input model (`portiloop_software/experiments/pareto_search_15_35_v4_0`) and a reference test vector computed with the real PyTorch `PortiloopNetwork`. To regenerate and verify it:

```bash
python3 docs/tools/export_model_json.py    # needs torch and numpy
node docs/tools/check_model_js.mjs         # JavaScript vs PyTorch, must print OK
```

The Portiloop filter constants in `portiloop.js` come from `portiloop_software/matlab/old_matlab/coef30.coe` (identical to the FIR stored in `portiloop_software/matlab/filter_lp.slx`) and from the 60 Hz notch stored in the same Simulink model.
