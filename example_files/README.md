# Example input files

A matched pair for the tab 1 → 2 → 3 → 4 pipeline. Both are generated from the **same**
physical model, so the simulated and experimental SOP results should agree to within the
injected noise.

## The physics

A Gaussian heat source of **20 µm FWHM** (σ₀ = 8.49 µm) deposits **100 pulses, one every
1 µs** (100 µs total). Between pulses the heat diffuses radially (α = 2×10⁻⁵ m²/s), so
each pulse spikes the centre and then decays before the next arrives. Old pulses
accumulate outward, so the profile broadens with time:

| | |
|---|---|
| fitted σ at t = 1 / 20 / 50 / 100 µs | 10.9 / 14.4 / 16.3 / 20.9 µm |
| peak T | 300 K → 6500 K |
| inter-pulse cooling (mid-run) | ≈ 440 K per cycle |
| R/σ over the run (R = 15 µm) | 1.38 → 0.72 |

## Files

**`COMSOL_radial_T_100pulses_20umFWHM.txt`** → **tab 1**
COMSOL 1-D line-graph format (8 `%` header lines; column 0 = radius in metres, one
temperature column per time). 60 radial nodes over 0–80 µm, 1001 times.
**The times are not stored in the file — set Δt = 0.1 µs in tab 1.**

**`experimental_SOP_spectra_100pulses.csv`** → **tab 3**
Synthetic *experimental* SOP spectra for the same run: wide format (column 0 =
wavelength in nm, one intensity column per time, times in the `# times =` header).
260 wavelengths over 560–800 nm, 200 times at 0.5 µs.
Collected through a **15 µm pinhole**, integrated over a **0.5 µs gate**, with **2 %**
noise.

## Suggested walkthrough

1. **Tab 1** — upload the COMSOL file, set **Δt = 0.1 µs**, download the converted series.
2. **Tab 2** — load that series; pinhole **R = 15 µm**, interpolation step **0.05 µs**,
   detector bin **0.5 µs** (matching the experimental gate).
3. **Tab 3** — upload the spectra file.
4. **Tab 4** — compare. Expect **Pearson r ≈ 0.997**, mean Δ ≈ +20 K, RMS ≈ 87 K.

The apparent temperature sits about **−13 %** below the true peak — the bias the
universal correction is designed to remove.

`generate_examples.py` reproduces both files (run it from the app folder).
