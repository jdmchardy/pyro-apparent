"""
Synthetic SOP (streak-optical-pyrometry) engine.

Reproduces the pipeline of the "Final_synthetic_SOP_code" notebook:

  radial T(r,t) from FEM
      -> resample onto the detector time base (gate dt)
      -> truncate at a synthetic pinhole r_max
      -> annulus-weighted Planck emission  =>  spectrogram  I(lambda, t)
      -> integrate over lambda             =>  total emission vs time
      -> gate-average the spectra and fit Planck (T + amplitude)
                                            =>  apparent T(t) and emissivity A(t)
      -> compare against experimental SOP temperatures / integrated emission

Pure numpy/scipy: no matplotlib, no Streamlit.

Note on the annulus weight: the notebook weights each ring by
``2*pi*(r_{i+1}^2 - r_i^2)`` (twice the geometric annulus area ``pi*(...)``).
That constant factor cancels in every normalised quantity and in the fitted
temperature; it only rescales the fitted amplitude A. ``area_factor`` selects the
convention -- 2.0 reproduces the notebook, 1.0 is the true annulus area.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.interpolate import RegularGridInterpolator, interp1d

h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23


# ------------------------------------------------------------------ Planck (T, amplitude)
def planck_TA(lam, T, A):
    """Planck spectral radiance with a free amplitude/emissivity A (notebook Planck_fit).

    lam [m], T [K]; the exponent is clipped to avoid overflow."""
    lam = np.asarray(lam, float)
    T = np.where(np.asarray(T, float) <= 1e-3, np.nan, T)
    expo = np.clip(h * c / (lam * kB * T), None, 700.0)
    return A * (2 * h * c**2 / lam**5) / np.expm1(expo)


def fit_planck_TA(lam, spectrum, T_guess=2500.0, A_guess=2e-10, maxfev=10000):
    """Least-squares fit of (T, A) to one spectrum. Returns (T, A) or (nan, nan)."""
    lam = np.asarray(lam, float)
    spectrum = np.asarray(spectrum, float)
    good = np.isfinite(spectrum)
    if good.sum() < 3 or not np.any(spectrum[good] > 0):
        return np.nan, np.nan
    try:
        popt, _ = curve_fit(planck_TA, lam[good], spectrum[good],
                            p0=[T_guess, A_guess], maxfev=maxfev)
        return float(popt[0]), float(popt[1])
    except (RuntimeError, ValueError):
        return np.nan, np.nan


# ------------------------------------------------------------------ resampling
def resample_profiles(r, t, T_matrix, t_new):
    """Interpolate T(r, t) onto a new time base (linear, extrapolating).

    r [m] (n_r,), t [s] (n_t,), T_matrix (n_r, n_t) -> T_new (n_r, len(t_new))."""
    r = np.asarray(r, float); t = np.asarray(t, float)
    T_matrix = np.asarray(T_matrix, float); t_new = np.asarray(t_new, float)
    interp = RegularGridInterpolator((r, t), T_matrix, method="linear",
                                     bounds_error=False, fill_value=None)
    Rg, Tg = np.meshgrid(r, t_new, indexing="ij")
    pts = np.stack([Rg.ravel(), Tg.ravel()], axis=-1)
    return interp(pts).reshape(r.size, t_new.size)


def apply_pinhole(r, T_matrix, r_max=None):
    """Keep only radii r <= r_max (the synthetic pinhole). Returns (r_cut, T_cut)."""
    r = np.asarray(r, float); T_matrix = np.asarray(T_matrix, float)
    if r_max is None:
        return r, T_matrix
    m = r <= r_max
    if m.sum() < 2:
        raise ValueError("pinhole radius keeps fewer than 2 radial points")
    return r[m], T_matrix[m]


# ------------------------------------------------------------------ synthetic SOP
def synthetic_spectrogram(r, T_cols, lambds, eps=1.0, area_factor=2.0):
    """Annulus-weighted collected emission for every time column.

    r [m] (n_r,), T_cols (n_r, n_t) [K], lambds [m] (n_lam,)
    -> spectrogram (n_lam, n_t) in W/sr/m (times the area factor convention).
    """
    r = np.asarray(r, float)
    T_cols = np.atleast_2d(np.asarray(T_cols, float))
    lambds = np.asarray(lambds, float)
    w = area_factor * np.pi * (r[1:]**2 - r[:-1]**2)      # ring weights (n_r-1,)
    T_in = T_cols[:-1, :]                                  # inner-edge T of each ring
    pref = 2 * h * c**2 / lambds**5                        # (n_lam,)
    out = np.empty((lambds.size, T_cols.shape[1]), float)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for j in range(T_cols.shape[1]):
            Tj = np.where(T_in[:, j] > 1e-3, T_in[:, j], np.nan)
            expo = h * c / (lambds[:, None] * kB * Tj[None, :])
            B = eps * pref[:, None] / np.expm1(np.clip(expo, None, 700.0))
            out[:, j] = np.where(np.isfinite(B), B, 0.0) @ w
    return out


def total_emission(spectro, lambds):
    """Integrate the spectrogram over wavelength -> total emission vs time."""
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return trapz(np.asarray(spectro, float), np.asarray(lambds, float), axis=0)


def gated_apparent_temperature(times, spectro, lambds, gate,
                               T_guess=2500.0, A_guess=2e-10):
    """Average the spectra inside consecutive gates of width `gate` [s] and fit Planck.

    Mimics the detector integration time: returns (t_mid, T_app, A_app) arrays, with
    each fit seeded from the previous one (as the notebook does).
    """
    times = np.asarray(times, float)
    spectro = np.asarray(spectro, float)
    if times.size == 0:
        return np.array([]), np.array([]), np.array([])
    gate = float(gate)
    if gate <= 0:
        raise ValueError("gate must be > 0")
    t_mid, T_app, A_app = [], [], []
    start, t_end = times[0], times[-1]
    while start < t_end:
        stop = start + gate
        idx = np.where((times >= start) & (times < stop))[0]
        if idx.size:
            mean_spec = np.nanmean(spectro[:, idx], axis=1)
            T_fit, A_fit = fit_planck_TA(lambds, mean_spec, T_guess, A_guess)
            if np.isfinite(T_fit):
                T_guess, A_guess = T_fit, A_fit
            t_mid.append(0.5 * (start + stop)); T_app.append(T_fit); A_app.append(A_fit)
        start = stop
    return np.array(t_mid), np.array(T_app), np.array(A_app)


def fit_spectra_vs_time(lambds, spectro, T_guess=2500.0, A_guess=2e-10):
    """Fit (T, A) to every time column -> (T_vs_time, A_vs_time). A is the apparent
    emissivity/amplitude the notebook plots against time."""
    spectro = np.asarray(spectro, float)
    n_t = spectro.shape[1]
    T_out = np.full(n_t, np.nan); A_out = np.full(n_t, np.nan)
    for i in range(n_t):
        T_fit, A_fit = fit_planck_TA(lambds, spectro[:, i], T_guess, A_guess)
        T_out[i], A_out[i] = T_fit, A_fit
        if np.isfinite(T_fit):
            T_guess, A_guess = T_fit, A_fit
    return T_out, A_out


# ------------------------------------------------------------------ comparison / stats
def compare_series(t_ref, T_ref, t_comp, T_comp, delay=0.0,
                   t_cut_min=None, t_cut_max=None):
    """Interpolate a comparison series onto the reference time grid and score it.

    Mirrors the notebook's compare_temperature_series: returns chi2, reduced chi2,
    Pearson r, R^2, the overlap bounds, and the interpolated arrays for plotting.
    `delay` shifts the comparison series (same units as the time arrays).
    """
    t_ref = np.asarray(t_ref, float); T_ref = np.asarray(T_ref, float)
    t_comp = np.asarray(t_comp, float) + float(delay)
    T_comp = np.asarray(T_comp, float)

    t_min = max(np.nanmin(t_ref), np.nanmin(t_comp))
    t_max = min(np.nanmax(t_ref), np.nanmax(t_comp))
    if t_cut_min is not None:
        t_min = max(t_min, t_cut_min)
    if t_cut_max is not None:
        t_max = min(t_max, t_cut_max)
    if not (t_max > t_min):
        raise ValueError("the two series do not overlap in time")

    m = (t_comp >= t_min) & (t_comp <= t_max)
    if m.sum() < 2:
        raise ValueError("fewer than 2 comparison points inside the overlap")
    f = interp1d(t_comp[m], T_comp[m], kind="linear",
                 bounds_error=False, fill_value=np.nan)
    T_interp = f(t_ref)
    good = np.isfinite(T_interp) & np.isfinite(T_ref)
    if good.sum() < 2:
        raise ValueError("fewer than 2 valid overlapping points")

    resid = T_ref[good] - T_interp[good]
    chi2 = float(np.sum(resid**2))
    r_val = float(pearsonr(T_ref[good], T_interp[good])[0])
    return dict(chi2=chi2, chi2_reduced=chi2/resid.size,
                r=r_val, r2=r_val**2, n_points=int(resid.size),
                time_bounds=(float(t_min), float(t_max)),
                t_valid=t_ref[good], T_ref_valid=T_ref[good],
                T_interp_valid=T_interp[good], residuals=resid)


def optimal_scale_chi2(data, model):
    """Best multiplicative scale a = sum(d*m)/sum(m^2) and Pearson chi2 before/after.

    Used for the normalised integrated-emission comparison (absolute intensity is
    lost in acquisition, so only the shape is meaningful)."""
    data = np.asarray(data, float); model = np.asarray(model, float)
    good = np.isfinite(data) & np.isfinite(model) & (model != 0)
    d, m = data[good], model[good]
    if d.size == 0:
        return dict(a_opt=np.nan, chi2=np.nan, chi2_reduced=np.nan,
                    chi2_unscaled=np.nan, n_points=0)
    a_opt = float(np.sum(d * m) / np.sum(m**2))
    chi2_uncorr = float(np.sum((d - m)**2 / np.abs(m)))
    scaled = a_opt * m
    chi2 = float(np.sum((d - scaled)**2 / np.abs(scaled)))
    return dict(a_opt=a_opt, chi2=chi2, chi2_reduced=chi2/d.size,
                chi2_unscaled=chi2_uncorr, n_points=int(d.size))


def load_two_column(path, x_col=1, y_col=0, x_scale=1.0):
    """Load an experimental 2-column file. The notebook's files store
    (value, time): temperatures with time in ns (x_scale=1e-3 -> µs) and
    integrated emission with time already in µs (x_scale=1.0).
    Returns (x, y) = (time, value)."""
    d = np.atleast_2d(np.loadtxt(path))
    return d[:, x_col] * x_scale, d[:, y_col]
