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

**Pipeline (1 → 4)**

1. **📥 COMSOL import** — upload a COMSOL 1-D radial line-graph export, set Δt if the
   file carries no times, preview the profiles and the `T(r,t)` map, and **download the
   app-format series** for tab 2.
2. **🔬 Simulated SOP** — load that series, interpolate it onto a finer time base, build
   synthetic emission spectra through the pinhole, **bin them over your detector time
   window**, and fit each bin: apparent `T`, Gaussian profile fit (`T₀`, `σ`, `R/σ`),
   lookup-table apparent `T`, peak and edge temperatures. Binned spectrogram, histories,
   bin explorer, CSV download.
3. **🔭 Experimental spectra** — upload a wide file of measured spectra (one column per
   time) and fit a Planck over the spectrometer window to each, giving the **experimental
   apparent `T` vs time** plus fitted amplitude. Spectrogram, spectrum inspector, CSV.
4. **📊 Compare** — overlay the simulated and experimental apparent temperatures with an
   adjustable time shift; mean/RMS difference, reduced χ², Pearson r and R², residuals,
   and optionally the universal correction of the experimental data to peak `T`.

**Supporting tools**

- **📈 Single profile** — one snapshot: apparent T, collected spectrum, Gaussian fit, and
  the universal-corrected peak.
- **🔎 Spectrum → T(r)** — fit a single measured spectrum for apparent T, then invert
  under the Gaussian assumption to peak T and a radial profile (σ known, saturated, or
  fitted from the spectral shape).
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

**Experimental spectra series** (tab 3) — wide CSV: column 0 = wavelength [nm]; each
further column = intensity at one time; times in a `# times = ...` comment.

**Measured spectrum** — two columns: wavelength, intensity (arb.). The wavelength unit
(nm / µm / m / Å) and the column indices are selectable in the tab.

Bundled samples live in `synthetic_temperatures/` (radial profiles and series) and
`synthetic_spectrums/` (single spectra and the `sample_spectra_series.csv` time series).

## How T_app is obtained (time-series tab)

| method | speed | accuracy |
|---|---|---|
| `table` *(default)* | O(1) interpolation of a precomputed `ρ(R/σ, ξ)` grid | exact across all `R/σ` |
| `gaussian` | Planck-fit the fitted Gaussian's analytic emission | exact-to-Gaussian |
| `numerical` | Planck-fit the emission integrated from the tabulated profile | exact ground truth |

The lookup table is built once per spectrometer window (~3 s) and cached to disk.

## Plots

All figures are **Plotly**: drag to pan, scroll or box-select to zoom, double-click to
reset, hover for values, and use the mode bar to download a PNG. `planck_plots.py`
(matplotlib) is retained for the offline scripts.

## Repository layout

| file | role |
|---|---|
| `streamlit_app.py` | the app (UI only) |
| `planck_model.py` | physics engine (numpy/scipy; no plotting) |
| `plotly_plots.py` | interactive Plotly figures used by the app |
| `planck_plots.py` | matplotlib figures (offline scripts) |
| `requirements.txt`, `.streamlit/config.toml` | deployment |
| `sample_*.csv` | example inputs |

`planck_fit_gui.py` (Tkinter) and `planck_fit_demo.py` (batch CLI) are the earlier desktop
front-ends over the same engine, kept for reference.
