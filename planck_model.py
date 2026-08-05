"""
Physics engine for the Gaussian-hot-spot grey-body pyrometry model.

Pure numpy/scipy -- NO matplotlib, NO plotting, NO I/O side effects -- so it can be
imported by both the batch script (planck_fit_demo.py) and the GUI (planck_fit_gui.py)
without forcing a matplotlib backend.

Model
-----
    T(r)  = T0 exp(-r^2 / 2 sigma^2)                       Gaussian radial profile
    F(l)  = 4 pi eps h c^2 sigma^2 / l^5 * I_trunc(x, u)   emission collected from r < R
with x = hc/(l kB T0), u = R^2/(2 sigma^2).

Universal scaling
-----------------
The single-Planck fit returns an apparent temperature T_app < T0. The bias depends on
only two dimensionless groups (sigma and absolute size cancel):
    * geometry   R/sigma
    * window     xi = c2 / (lambda_c T0),  lambda_c = sqrt(lam_lo lam_hi),  c2 = hc/kB
Saturated (R/sigma >~ 1.5) closed form (Wien limit):  T_app/T0 = xi E1(xi) e^xi.
"""
import os
import re
import numpy as np
from scipy import special, optimize
from scipy.interpolate import RegularGridInterpolator

h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
WIEN_B   = 2.897771955e-3   # Wien displacement constant [m K]
C2       = h * c / kB        # second radiation constant hc/kB [m K]

# ----------------------------------------------------------------- model
def I_trunc(x, u_max, n_max=20000, tol=1e-14):
    """sum_n [E1(nx) - E1(nx e^umax)]  -- truncated radial integral."""
    x = np.atleast_1d(x).astype(float)
    tot, n = np.zeros_like(x), 1
    while n <= n_max:
        term = special.exp1(n * x) - special.exp1(n * x * np.exp(u_max))
        tot += term
        if np.all(term < tol * np.maximum(tot, 1e-300)):
            break
        n += 1
    return tot

def spectrum(lam, T0, sigma, R, eps=1.0):
    """Emission collected from r < R for T(r) = T0 exp(-r^2/2sigma^2)."""
    x = h * c / (lam * kB * T0)
    return 4*np.pi*eps*h*c**2*sigma**2 / lam**5 * I_trunc(x, R**2/(2*sigma**2))

def T_radial(r, T0, sigma):
    """Gaussian radial temperature profile T(r) = T0 exp(-r^2/2sigma^2)."""
    return T0 * np.exp(-r**2 / (2*sigma**2))

def planck(lam, T, A=1.0):
    return A * 2*h*c**2 / lam**5 / np.expm1(h*c/(lam*kB*T))

# ----------------------------------------------------------------- fit
def fit_temperature(lam, F, T_guess=3000.0):
    """Fit a single grey-body Planck curve (free amplitude + T) in log space.

    (The log-space *residual* is only a numerical device to weight the fit evenly
    across orders of magnitude -- it is unrelated to any plot axis scaling.)"""
    def resid(p):
        logA, T = p
        return np.log(planck(lam, T, np.exp(logA))) - np.log(F)
    A0 = F[len(F)//2] / planck(lam[len(lam)//2], T_guess)
    out = optimize.least_squares(resid, [np.log(A0), T_guess],
                                 bounds=([-np.inf, 100.], [np.inf, 1e5]))
    return out.x[1], np.exp(out.x[0])

# ----------------------------------------------------------------- scaling helpers
def xi_window(T0, lam_lo, lam_hi):
    """Window Wien parameter xi = c2 / (lambda_c T0), lambda_c = geometric-mean window."""
    lam_c = np.sqrt(lam_lo * lam_hi)
    return C2 / (lam_c * T0)

def wien_saturated_ratio(xi):
    """Analytic saturated (R->inf) apparent-T ratio in the Wien limit: xi E1(xi) e^xi."""
    return xi * special.exp1(xi) * np.exp(xi)

def apparent_temperature(T0, R_over_sigma, lam_lo, lam_hi, sigma=10e-6, n_lam=200):
    """Forward model: apparent (fitted) temperature for a spot of true peak T0.

    sigma is immaterial to the *ratio* T_app/T0 (only R/sigma and the window matter),
    so its default value is arbitrary."""
    lam_fit = np.linspace(lam_lo, lam_hi, n_lam)
    F_fit   = spectrum(lam_fit, T0, sigma, R_over_sigma*sigma)
    T_app, _ = fit_temperature(lam_fit, F_fit, T_guess=T0)
    return T_app

def bias_ratio(T0, R_over_sigma, lam_lo, lam_hi):
    """Exact model correction factor rho = T_app / T0 for a given true T0."""
    return apparent_temperature(T0, R_over_sigma, lam_lo, lam_hi) / T0

def solve_bias(T0, R_over_sigma, lam_lo, lam_hi):
    """Fractional bias T_app/T0 - 1 for a bare (T0, R/sigma) point."""
    return bias_ratio(T0, R_over_sigma, lam_lo, lam_hi) - 1.0

# ----------------------------------------------------------------- LOOKUP TABLE
# Precompute rho = T_app/T0 on a (R/sigma, xi) grid from the exact forward model, so it
# can be interpolated in O(1). Unlike the closed form xi E1(xi) e^xi (the R/sigma->inf
# limit), the table captures the full R/sigma dependence for ANY collection geometry.
_TABLE_CACHE = {}
_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

def default_ros_grid():
    # dense where rho changes fast (R/sigma < 2), coarse in the saturated tail
    return np.unique(np.concatenate([np.linspace(0.05, 2.0, 40),
                                     np.linspace(2.25, 6.0, 16)]))

def default_xi_grid():
    return np.geomspace(1.0, 30.0, 48)   # T0 ~ 700 K .. 21000 K for a 575-800 nm window

def build_ratio_table(lam_lo, lam_hi, ros=None, xi=None):
    """Tabulate rho(R/sigma, xi) = T_app/T0 over a grid, from the exact forward model."""
    ros = default_ros_grid() if ros is None else np.asarray(ros, float)
    xi  = default_xi_grid()  if xi  is None else np.asarray(xi, float)
    lam_c = np.sqrt(lam_lo * lam_hi)
    rho = np.empty((ros.size, xi.size))
    for j, x in enumerate(xi):
        T0 = C2 / (lam_c * x)                     # xi -> T0 for this window
        for i, rs in enumerate(ros):
            rho[i, j] = 1.0 + solve_bias(T0, rs, lam_lo, lam_hi)
    return dict(ros=ros, xi=xi, rho=rho, lam_lo=lam_lo, lam_hi=lam_hi)

def get_ratio_table(lam_lo, lam_hi, cache=True):
    """Return the rho table for a window, building it once (memory + optional disk cache)."""
    key = f"{lam_lo*1e9:.2f}_{lam_hi*1e9:.2f}"
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    path = os.path.join(_CACHE_DIR, f"ratio_table_{key}nm.npz")
    if cache and os.path.exists(path):
        d = np.load(path)
        tab = dict(ros=d["ros"], xi=d["xi"], rho=d["rho"], lam_lo=lam_lo, lam_hi=lam_hi)
    else:
        tab = build_ratio_table(lam_lo, lam_hi)
        if cache:
            try:
                np.savez(path, ros=tab["ros"], xi=tab["xi"], rho=tab["rho"])
            except Exception:
                pass
    _TABLE_CACHE[key] = tab
    return tab

def ratio_from_table(tab, ros_q, xi_q):
    """Bilinearly interpolate rho at (R/sigma, xi); queries are clamped to the grid
    (R/sigma beyond the grid is saturated; xi beyond the grid is clamped)."""
    interp = tab.get("_interp")
    if interp is None:
        interp = RegularGridInterpolator((tab["ros"], tab["xi"]), tab["rho"],
                                         bounds_error=False, fill_value=None)
        tab["_interp"] = interp
    ros_q = np.clip(ros_q, tab["ros"][0], tab["ros"][-1])
    xi_q  = np.clip(xi_q,  tab["xi"][0],  tab["xi"][-1])
    pts = np.stack([np.atleast_1d(ros_q), np.atleast_1d(xi_q)], axis=-1)
    out = interp(pts)
    return out if np.ndim(ros_q) or np.ndim(xi_q) else float(out[0])

# ----------------------------------------------------------------- INVERSION
def correct_temperature(T_app, R_over_sigma, lam_lo, lam_hi,
                        tol=1e-2, max_iter=50, return_info=False):
    """Recover the true peak T0 from a measured apparent temperature T_app.

    Inverts  T_app = rho(R/sigma, xi(T0)) * T0  by fixed-point iteration
        T0 <- T_app / rho(R/sigma, xi(T0)),
    where rho is evaluated from the EXACT forward model (valid at any xi, not just
    the Wien limit). Converges in a few iterations because rho depends only weakly
    on T0 through xi. Note only R/sigma is needed, not the absolute sigma.
    """
    T0 = float(T_app)
    converged = False
    for i in range(1, max_iter + 1):
        rho = bias_ratio(T0, R_over_sigma, lam_lo, lam_hi)
        T0_new = T_app / rho
        if abs(T0_new - T0) < tol:
            T0, converged = T0_new, True
            break
        T0 = T0_new
    if return_info:
        return T0, dict(iterations=i, converged=converged,
                        rho=rho, xi=float(xi_window(T0, lam_lo, lam_hi)))
    return T0

# ----------------------------------------------------------------- per-config solve
def run_config(cfg):
    """Solve one configuration dict; return arrays + scalar results (no plotting).

    cfg keys: T0, sigma, R, fit_lo, fit_hi, and optional 'group'/'label' for display.
    """
    T0, sigma, R = cfg["T0"], cfg["sigma"], cfg["R"]
    fit_lo, fit_hi = cfg["fit_lo"], cfg["fit_hi"]

    lam_full = np.logspace(np.log10(300e-9), np.log10(20e-6), 400)
    F_full   = spectrum(lam_full, T0, sigma, R)

    lam_fit  = np.linspace(fit_lo, fit_hi, 200)
    F_fit    = spectrum(lam_fit, T0, sigma, R)
    T_app, A_app = fit_temperature(lam_fit, F_fit, T_guess=T0)

    r = np.linspace(0, max(R, 3*sigma), 300)
    T0_rec = correct_temperature(T_app, R/sigma, fit_lo, fit_hi)   # inversion self-check

    return dict(cfg=cfg, group=cfg.get("group", cfg.get("label", "config")),
                T0=T0, sigma=sigma, R=R,
                lam_full=lam_full, F_full=F_full,
                lam_fit=lam_fit, F_fit=F_fit,
                T_app=T_app, A_app=A_app,
                r=r, T_r=T_radial(r, T0, sigma),
                R_over_sigma=R/sigma, xi=xi_window(T0, fit_lo, fit_hi),
                bias_K=T_app - T0, bias_frac=T_app/T0 - 1.0,
                T0_rec=T0_rec, rec_err=T0_rec - T0)

# ----------------------------------------------------------------- ARBITRARY (SIMULATED) PROFILE
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # np.trapz removed in NumPy 2.x

def spectrum_from_profile(lam, r, T_r, R=None, eps=1.0):
    """Collected spectrum for a TABULATED radial profile T(r) (e.g. from a hydrocode).

        F(lam) = integral_0^R  2 pi r eps B(lam, T(r)) dr

    with B the full Planck spectral radiance (no Wien approximation, no Gaussian
    assumption). r, T_r are 1-D arrays [m], [K]; lam [m]. If R is given the profile
    is truncated there (with an interpolated endpoint); otherwise the whole profile
    is used. Returns F(lam) with the same units as spectrum() [W sr^-1 m^-1].
    """
    lam = np.atleast_1d(np.asarray(lam, float))
    r   = np.asarray(r, float)
    T_r = np.asarray(T_r, float)
    order = np.argsort(r); r, T_r = r[order], T_r[order]
    if R is not None and R < r[-1]:
        T_at_R = float(np.interp(R, r, T_r))       # interpolate the endpoint at R
        keep = r <= R
        r   = np.append(r[keep], R)
        T_r = np.append(T_r[keep], T_at_R)

    # Planck radiance B(lam, T(r)) via broadcasting; B = 0 wherever T <= 0
    Tpos = np.where(T_r > 0, T_r, np.nan)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        expo = h * c / (lam[:, None] * kB * Tpos[None, :])
        B = 2 * h * c**2 / lam[:, None]**5 / np.expm1(expo)
    B = np.where(np.isfinite(B), B, 0.0)
    integrand = 2 * np.pi * eps * r[None, :] * B         # (n_lam, n_r)
    return _trapz(integrand, r, axis=1)

def load_profile(path):
    """Load a two-column radial profile file: radius[um], temperature[K].

    Accepts comma- or whitespace-delimited, '#' comments, and an optional header
    row. Returns (r [m], T [K]).
    """
    attempts = (dict(delimiter=","), dict(), dict(delimiter=",", skiprows=1), dict(skiprows=1))
    for kw in attempts:
        try:
            data = np.loadtxt(path, comments="#", **kw)
        except Exception:
            continue
        data = np.atleast_2d(data)
        if data.shape[1] >= 2 and np.all(np.isfinite(data[:, :2])):
            return data[:, 0] * 1e-6, data[:, 1]
    raise ValueError("could not parse profile; need 2 columns: radius[um], temperature[K]")

def fit_gaussian_profile(r, T_r, R=None):
    """Least-squares fit T(r) = T0 exp(-r^2 / 2 sigma^2) to the profile points inside R.

    Returns (T0, sigma). Uses only datapoints with r <= R (the collection aperture),
    since that is the region the spectrometer actually sees.
    """
    r = np.asarray(r, float); T_r = np.asarray(T_r, float)
    if R is not None:
        m = r <= R
        r, T_r = r[m], T_r[m]
    if r.size < 3:
        raise ValueError("need at least 3 profile points inside R to fit a Gaussian")
    T0_0 = float(T_r.max())
    w = np.clip(T_r, 0, None)
    sig0 = np.sqrt(np.sum(w * r**2) / np.sum(w)) if np.sum(w) > 0 else (np.ptp(r)/2 or 1e-6)
    def resid(p):
        T0g, sig = p
        return T0g * np.exp(-r**2 / (2*sig**2)) - T_r
    out = optimize.least_squares(resid, [T0_0, max(sig0, 1e-9)],
                                 bounds=([1.0, 1e-9], [1e6, 1e-2]))
    return float(out.x[0]), float(out.x[1])

def evaluate_profile(r, T_r, lam_lo, lam_hi, R=None, eps=1.0, compare_gaussian=True):
    """Forward observables for a simulated profile: collected spectrum + apparent T.

    Returns a dict with the broadband spectrum, the in-window fit, the fitted
    apparent temperature T_app, the true peak T_peak, the bias, the window Wien
    parameter xi (referenced to T_peak), and the spectrally-integrated collected
    power P_total.

    If compare_gaussian is True, also fits a Gaussian T(r) to the profile points
    inside R and adds the Gaussian universal-model prediction (see fit_gaussian_profile).
    """
    r = np.asarray(r, float); T_r = np.asarray(T_r, float)
    Reff = float(r.max()) if R is None else float(R)
    T_peak = float(np.max(T_r))

    lam_full = np.logspace(np.log10(300e-9), np.log10(20e-6), 400)
    F_full   = spectrum_from_profile(lam_full, r, T_r, R, eps)
    lam_fit  = np.linspace(lam_lo, lam_hi, 200)
    F_fit    = spectrum_from_profile(lam_fit, r, T_r, R, eps)
    T_app, A_app = fit_temperature(lam_fit, F_fit, T_guess=T_peak)

    res = dict(r=r, T_r=T_r, R=Reff, T_peak=T_peak,
               lam_full=lam_full, F_full=F_full,
               lam_fit=lam_fit, F_fit=F_fit,
               T_app=T_app, A_app=A_app,
               bias_frac=T_app / T_peak - 1.0,
               xi=float(xi_window(T_peak, lam_lo, lam_hi)),
               P_total=float(_trapz(F_full, lam_full)))

    if compare_gaussian:
        try:
            T0_g, sig_g = fit_gaussian_profile(r, T_r, Reff)
            ros_g   = Reff / sig_g
            T_g_r   = T0_g * np.exp(-r**2 / (2*sig_g**2))          # fitted profile curve
            F_g_full = spectrum(lam_full, T0_g, sig_g, Reff, eps)  # Gaussian-model spectrum
            F_g_fit  = spectrum(lam_fit,  T0_g, sig_g, Reff, eps)
            T_app_g, A_g = fit_temperature(lam_fit, F_g_fit, T_guess=T0_g)
            # apply the universal correction to the ACTUAL data using the fitted R/sigma
            T0_universal = correct_temperature(T_app, ros_g, lam_lo, lam_hi)
            res.update(gauss=dict(
                T0=T0_g, sigma=sig_g, R_over_sigma=ros_g, T_g_r=T_g_r,
                F_full=F_g_full, T_app=T_app_g, A_app=A_g,
                bias_frac=T_app_g / T0_g - 1.0,
                T0_universal=T0_universal,
                pred_err=T_app_g - T_app,          # universal prediction - true apparent T
                recover_err=T0_universal - T_peak))  # universal-recovered peak - true peak
        except Exception:
            res["gauss"] = None
    return res

# ----------------------------------------------------------------- TIME SERIES OF PROFILES
def load_profile_series(path):
    """Load a time series of radial profiles from one wide file.

    Layout: column 0 = radius [um]; each further column = temperature [K] at one time.
    The times may be supplied in a comment line, e.g.
        # times = 0.0 0.5 1.0 1.5 ...
    (numbers after the word 'time'/'times'); if absent or mismatched, column indices
    0,1,2,... are used. Comma- or whitespace-delimited.

    Returns (times [n_t], r [m, n_r], T_cols [K, shape (n_r, n_t)]).
    """
    times = None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                if "time" in s.lower():
                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
                    if nums:
                        times = np.array([float(x) for x in nums], float)
                        break   # first times line wins; ignore later comments
                continue
            break
    data = None
    for kw in (dict(delimiter=","), dict()):
        try:
            data = np.loadtxt(path, comments="#", **kw)
        except Exception:
            data = None
            continue
        data = np.atleast_2d(data)
        if data.shape[1] >= 2:
            break
    if data is None or data.ndim < 2 or data.shape[1] < 2:
        raise ValueError("need a radius column plus >=1 temperature column")
    r = data[:, 0] * 1e-6
    T_cols = data[:, 1:]
    n_t = T_cols.shape[1]
    if times is None or len(times) != n_t:
        times = np.arange(n_t, dtype=float)
    return times, r, T_cols

def parse_comsol_line_graph(path, t_end=None, times=None):
    """Parse a COMSOL 1-D 'Line graph' export into (times, r[m], T[n_r, n_t]).

    The file has radius rows (column 0 = R in metres) and one temperature column per
    time. Times are read from the column headers if they carry '@ t=' / 't=' / 'Time='
    labels; otherwise from `times`, else 0..t_end in equal steps, else 0..n_t-1.
    """
    desc = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("%"):
                if "Temperature" in line:
                    desc = line            # the R / Temperature column-descriptor line
            else:
                break
    data = np.atleast_2d(np.loadtxt(path, comments="%"))
    if data.shape[1] < 2:
        raise ValueError("expected >= 2 columns (R + temperature snapshots)")
    r = data[:, 0].astype(float)
    T_cols = data[:, 1:].astype(float)
    n_t = T_cols.shape[1]
    t = None
    if desc is not None:                    # try to read times from the header labels
        found = re.findall(r"(?:@\s*)?(?:time|t)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                           desc, re.I)
        if len(found) != n_t:
            found = re.findall(r"@\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", desc)
        if len(found) == n_t:
            t = np.array([float(x) for x in found], float)
    if t is None:
        if times is not None:
            t = np.asarray(times, float)
        elif t_end is not None:
            t = np.linspace(0.0, float(t_end), n_t)
        else:
            t = np.arange(n_t, dtype=float)
    if t.size != n_t:
        raise ValueError(f"number of times ({t.size}) != temperature columns ({n_t})")
    order = np.argsort(r)
    return t, r[order], T_cols[order]

def write_series_csv(path, times, r, T_cols, note=""):
    """Write (times, r[m], T[n_r, n_t]) as the app's wide CSV: radius_um then T columns,
    with the times in a '# times = ...' comment header (whatever unit `times` is in)."""
    times = np.asarray(times, float); r = np.asarray(r, float); T_cols = np.asarray(T_cols, float)
    if T_cols.shape != (r.size, times.size):
        raise ValueError("T_cols must have shape (len(r), len(times))")
    hdr = "times = " + " ".join(f"{x:g}" for x in times)
    if note:
        hdr += "\n" + note
    np.savetxt(path, np.column_stack([r*1e6, T_cols]),
               delimiter=",", header=hdr, comments="# ", fmt="%.6g")

def evaluate_profile_series(times, r, T_cols, lam_lo, lam_hi, R=None, method="table"):
    """Reduce a time series of profiles to temperature histories.

    For each time (column of T_cols) returns, versus time:
      * T_peak  -- the actual peak temperature of the data, max_r T(r);
      * T_gauss -- the fitted peak of the Gaussian fitted inside the pinhole r<=R;
      * T_app   -- the apparent temperature a single-Planck fit would return;
      * T_edge  -- the temperature at the pinhole edge, T(R).

    ``method`` selects how T_app is obtained (all but "numerical" fit a Gaussian to the
    pinhole region first, then map the fitted (R/sigma, xi) to rho = T_app/T0):
      * "table" (default) -- interpolate rho from a precomputed (R/sigma, xi) lookup
        table (get_ratio_table). O(1), and exact across the full R/sigma dependence.
      * "universal" -- closed-form saturated law  T_app = T0 * xi E1(xi) e^xi. Fast, but
        assumes the pinhole is saturated (R/sigma >~ 1.5) and Wien-valid (xi >~ 2).
      * "gaussian" -- Planck-fit the Gaussian's ANALYTIC emission (any R/sigma; slower).
      * "numerical" -- Planck-fit the emission integrated directly from the tabulated
        profile (exact for the given data; slowest).
    T_peak and T_edge are always read straight from the data.
    """
    times = np.asarray(times, float)
    r = np.asarray(r, float)
    T_cols = np.asarray(T_cols, float)
    n_t = T_cols.shape[1]
    Reff = float(r.max()) if R is None else float(R)

    T_peak = np.empty(n_t); T_app = np.full(n_t, np.nan)
    T_gauss = np.full(n_t, np.nan); T_edge = np.empty(n_t); sigma = np.full(n_t, np.nan)
    lam_fit = np.linspace(lam_lo, lam_hi, 200)
    tab = get_ratio_table(lam_lo, lam_hi) if method == "table" else None
    for j in range(n_t):
        Tt = T_cols[:, j]
        T_peak[j] = float(np.max(Tt))
        T_edge[j] = float(np.interp(Reff, r, Tt))
        try:
            T0g, sg = fit_gaussian_profile(r, Tt, Reff)
            T_gauss[j], sigma[j] = T0g, sg
        except Exception:
            T0g = None
        if method == "numerical":
            F = spectrum_from_profile(lam_fit, r, Tt, R)
            T_app[j], _ = fit_temperature(lam_fit, F, T_guess=max(T_peak[j], 300.0))
        elif T0g is None:
            pass
        elif method == "gaussian":                         # analytic Gaussian emission
            F = spectrum(lam_fit, T0g, sg, Reff)
            T_app[j], _ = fit_temperature(lam_fit, F, T_guess=T0g)
        elif method == "table":                            # interpolate rho (any R/sigma)
            rho = ratio_from_table(tab, Reff/sg, xi_window(T0g, lam_lo, lam_hi))
            T_app[j] = T0g * rho
        else:                                              # "universal" closed form
            T_app[j] = T0g * wien_saturated_ratio(xi_window(T0g, lam_lo, lam_hi))
    return dict(times=times, R=Reff, T_peak=T_peak, T_gauss=T_gauss,
                T_app=T_app, T_edge=T_edge, sigma=sigma, method=method)
