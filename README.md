# Gaussian hot-spot pyrometry

Interactive tools for the **apparent-temperature bias** of a Gaussian hot spot seen
through a finite collection pinhole, and its correction via the universal
`ρ(R/σ, ξ) = T_app/T₀` surface.

When you fit a single Planck curve to the light collected through a pinhole, the fitted
("apparent") temperature is biased **below** the true peak, because the detector sums the
area-weighted emission of every radius inside the aperture. This app quantifies that bias,
corrects for it, and — its main purpose — processes **time series** of simulated radial
temperature profiles (e.g. from a hydrocode) to produce temperature histories.

## Run it

### Live (Streamlit Community Cloud)
Push this folder to a GitHub repo, then at [share.streamlit.io](https://share.streamlit.io)
create an app pointing at `streamlit_app.py`. Dependencies install from `requirements.txt`.

### Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tabs

- **⏱ Time series** *(main)* — upload a sequence of radial profiles; get the four
  temperature histories (actual peak, fitted-Gaussian peak, apparent, pinhole-edge), a
  `T(r,t)` map, the collected emission spectra vs time, the fit-geometry/bias evolution,
  and a snapshot explorer showing each profile + Gaussian fit (pinhole shaded) and its
  collected spectrum. Downloadable results.
- **📈 Single profile** — one snapshot: apparent T, collected spectrum, Gaussian fit, and
  the universal-corrected peak.
- **🎯 Gaussian & universal** — slider explorer for `(T₀, σ, R)`; apparent T, bias, the
  master curve `ξE₁(ξ)eˣ`, and the `ρ(R/σ, ξ)` lookup table with your point marked.
- **🌐 Universality (batch)** — sweep `(T₀, σ)` families and see the bias collapse.
- **📖 About / theory** — the model, dimensionless variables, and methods.

## Input file formats

**Time series** — wide CSV: column 0 = radius [µm]; each further column = `T` [K] at one
time. Put the times in a comment line:
```
# times = 0 2.5 5 7.5 ... 100
0,   300,  310,  ...
0.5, 300,  340,  ...
```
If the times line is absent, column indices are used. Comma or whitespace delimited.

**Single profile** — two columns: radius [µm], `T` [K].

Bundled samples: `sample_Tseries_diffusion.csv` (10 µm-FWHM beam + radial diffusion over
100 µs), `sample_Tprofile.csv` (peaked), `near_Gaussian_Tprofile.csv`.

## How T_app is obtained (time-series tab)

| method | speed | accuracy |
|---|---|---|
| `table` *(default)* | O(1) interpolation of a precomputed `ρ(R/σ, ξ)` grid | exact across all `R/σ` |
| `gaussian` | Planck-fit the fitted Gaussian's analytic emission | exact-to-Gaussian |
| `numerical` | Planck-fit the emission integrated from the tabulated profile | exact ground truth |

The lookup table is built once per spectrometer window (~3 s) and cached to disk.

## Repository layout

| file | role |
|---|---|
| `streamlit_app.py` | the app (UI only) |
| `planck_model.py` | physics engine (numpy/scipy; no plotting) |
| `planck_plots.py` | shared matplotlib figures |
| `requirements.txt`, `.streamlit/config.toml` | deployment |
| `sample_*.csv` | example inputs |

`planck_fit_gui.py` (Tkinter) and `planck_fit_demo.py` (batch CLI) are the earlier desktop
front-ends over the same engine, kept for reference.
