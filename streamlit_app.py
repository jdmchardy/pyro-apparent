"""
Grey-body pyrometry of a Gaussian hot spot -- Streamlit app.

Pipeline (tabs 1-4):
  1 COMSOL import       -- convert a COMSOL radial export into the app's series format.
  2 Simulated SOP       -- interpolate + time-bin a series, build synthetic SOP
                           spectra, fit Gaussian profiles and apparent temperature.
  3 Experimental spectra-- fit Planck to measured spectra -> apparent T vs time.
  4 Compare             -- simulated vs experimental apparent T vs time.

Supporting tools:
  * Single profile       -- one snapshot: apparent T, spectrum, Gaussian comparison.
  * Spectrum -> T(r)     -- fit one spectrum, invert to peak T and radial profile.
  * Gaussian & universal -- explore the analytic model, master curve, lookup table.
  * Universality (batch) -- collapse of the bias across (T0, sigma) families.
  * About / theory       -- what the app computes and how.

All physics is in planck_model.py; shared figures in planck_plots.py.
Run locally:   streamlit run streamlit_app.py
"""
import os
import io
import contextlib
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import streamlit as st

from planck_model import (spectrum, spectrum_from_profile, planck, fit_temperature,
                          fit_gaussian_profile, xi_window, wien_saturated_ratio,
                          correct_temperature, run_config,
                          load_profile, evaluate_profile,
                          load_profile_series, evaluate_profile_series,
                          parse_comsol_line_graph, load_spectra_series,
                          resample_series_time, bin_columns, write_series_csv,
                          get_ratio_table, ratio_from_table, C2)
from planck_plots import plot_comparison, plot_profile_eval

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD_LABEL = {"table": "lookup table", "gaussian": "analytic Gaussian",
                "numerical": "numerical (exact)"}

st.set_page_config(page_title="Gaussian hot-spot pyrometry",
                   page_icon="🔥", layout="wide")


# ============================================================ cached compute
@st.cache_data(show_spinner=False)
def _read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()

@st.cache_data(show_spinner=False)
def _load_series(file_bytes, name):
    suffix = os.path.splitext(name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes); path = f.name
    try:
        times, r, T = load_profile_series(path)
    finally:
        os.unlink(path)
    return times, r, T

@st.cache_data(show_spinner=False)
def _load_single(file_bytes, name):
    suffix = os.path.splitext(name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes); path = f.name
    try:
        r, T = load_profile(path)
    finally:
        os.unlink(path)
    return r, T

def _is_comsol(file_bytes):
    head = file_bytes[:400].decode("utf-8", "ignore")
    return "COMSOL" in head or head.lstrip().startswith("% Model")

@st.cache_data(show_spinner=False)
def _series_arrays(file_bytes, name, dt_us):
    """Load a series from either the app CSV format or a COMSOL line-graph export.
    Returns (times [file units, µs for COMSOL], r [m], T [n_r, n_t]).
    For COMSOL files without embedded times, dt_us is the fixed step between snapshots."""
    if _is_comsol(file_bytes):
        suffix = os.path.splitext(name)[1] or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(file_bytes); path = f.name
        try:
            dt = (dt_us * 1e-6) if dt_us else None
            t, r, T = parse_comsol_line_graph(path, dt=dt)
        finally:
            os.unlink(path)
        return t * 1e6, r, T            # times in µs
    return _load_series(file_bytes, name)

@st.cache_data(show_spinner=False)
def _eval_series(file_bytes, name, R, method, lo, hi, dt_us):
    times, r, T = _series_arrays(file_bytes, name, dt_us)
    return evaluate_profile_series(times, r, T, lo, hi, R=R, method=method)

@st.cache_data(show_spinner=False)
def _eval_snapshot(file_bytes, name, j, R, lo, hi, dt_us):
    _, r, T = _series_arrays(file_bytes, name, dt_us)
    return evaluate_profile(r, T[:, j], lo, hi, R=R)


@st.cache_data(show_spinner=False, max_entries=4)
def _run_sim(file_bytes, name, dt_us, R_um, step_us, bin_us, lo, hi, n_lam):
    """Interpolate a series onto a finer time base, build synthetic spectra through the
    pinhole, bin them over the detector window, and fit each bin."""
    times, r, T = _series_arrays(file_bytes, name, dt_us)
    R = R_um * 1e-6
    t_fine = np.arange(times.min(), times.max() + 0.5*step_us, step_us)
    T_fine = resample_series_time(times, r, T, t_fine)
    lam = np.linspace(lo, hi, int(n_lam))

    spec = np.empty((lam.size, t_fine.size))
    for j in range(t_fine.size):
        spec[:, j] = spectrum_from_profile(lam, r, T_fine[:, j], R)

    t_bin, spec_bin, counts = bin_columns(t_fine, spec, bin_us)
    _, prof_bin, _ = bin_columns(t_fine, T_fine, bin_us)

    n = t_bin.size
    T_app = np.full(n, np.nan); A_app = np.full(n, np.nan)
    T_peak = np.full(n, np.nan); T_gauss = np.full(n, np.nan)
    sigma = np.full(n, np.nan); T_edge = np.full(n, np.nan)
    T_app_tab = np.full(n, np.nan)
    tab = get_ratio_table(lo, hi)
    floor = 1e-9 * np.nanmax(spec_bin) if np.isfinite(spec_bin).any() else 0.0
    for j in range(n):
        T_peak[j] = float(np.nanmax(prof_bin[:, j]))
        T_edge[j] = float(np.interp(R, r, prof_bin[:, j]))
        col = spec_bin[:, j]
        if np.isfinite(col).all() and np.nanmax(col) > floor and np.all(col > 0):
            try:
                T_app[j], A_app[j] = fit_temperature(lam, col,
                                                     T_guess=max(T_peak[j], 500.0))
            except Exception:
                pass
        try:
            T0g, sg = fit_gaussian_profile(r, prof_bin[:, j], R)
            T_gauss[j], sigma[j] = T0g, sg
            T_app_tab[j] = float(T0g * ratio_from_table(
                tab, R/sg, float(xi_window(T0g, lo, hi))))
        except Exception:
            pass
    return dict(t_fine=t_fine, n_fine=int(t_fine.size), lam=lam, t_bin=t_bin,
                spec_bin=spec_bin, counts=counts, prof_bin=prof_bin,
                T_app=T_app, A_app=A_app, T_peak=T_peak, T_gauss=T_gauss,
                sigma=sigma, T_edge=T_edge, T_app_tab=T_app_tab)


@st.cache_data(show_spinner=False, max_entries=4)
def _fit_spectra_series(file_bytes, name, lo, hi):
    """Fit a Planck (free amplitude + T) to every column of a wide spectra file."""
    suffix = os.path.splitext(name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes); path = f.name
    try:
        times, lam, I = load_spectra_series(path)
    finally:
        os.unlink(path)
    m = (lam >= lo) & (lam <= hi)
    if m.sum() < 5:
        raise ValueError(f"only {int(m.sum())} points inside {lo*1e9:.0f}-{hi*1e9:.0f} nm "
                         f"(data spans {lam.min()*1e9:.0f}-{lam.max()*1e9:.0f} nm)")
    n = I.shape[1]
    T = np.full(n, np.nan); A = np.full(n, np.nan)
    guess = 3000.0
    for j in range(n):
        col = I[m, j]
        good = np.isfinite(col) & (col > 0)
        if good.sum() < 5:
            continue
        try:
            T[j], A[j] = fit_temperature(lam[m][good], col[good], T_guess=guess)
            if np.isfinite(T[j]):
                guess = T[j]
        except Exception:
            pass
    return dict(times=times, lam=lam, I=I, mask=m, T_app=T, A_app=A)


@st.cache_data(show_spinner=False)
def _read_xy(file_bytes, name, x_col, y_col, x_scale=1.0):
    """Load a 2-column data file from uploaded bytes -> (x * x_scale, y)."""
    suffix = os.path.splitext(name)[1] or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes); path = f.name
    try:
        try:
            d = np.atleast_2d(np.loadtxt(path, delimiter=",", comments=("#", "%")))
        except ValueError:
            d = np.atleast_2d(np.loadtxt(path, comments=("#", "%")))
    finally:
        os.unlink(path)
    if d.shape[1] <= max(x_col, y_col):
        raise ValueError(f"file has {d.shape[1]} columns; "
                         f"need columns {x_col} and {y_col}")
    return d[:, x_col] * x_scale, d[:, y_col]


@st.cache_data(show_spinner=False)
def fit_shape_3param(lam, I, T_guess=4000.0, ros_guess=1.0, n_max=200):
    """Fit (amplitude, T_peak, R/sigma) directly to a measured spectrum.

    The collected spectrum is NOT a Planck function -- its departure from one encodes
    the temperature spread, hence R/sigma. This recovers both T_peak and R/sigma from
    the spectral SHAPE alone, with no imaging input. It only works below saturation:
    for R/sigma >~ 1.5 the shape becomes independent of R/sigma (a true degeneracy),
    so R/sigma runs free there while T_peak stays well determined.
    """
    from scipy.optimize import curve_fit
    lam = np.asarray(lam, float); I = np.asarray(I, float)
    if lam.size > n_max:                       # subsample: the E1 series is the cost
        k = np.linspace(0, lam.size - 1, n_max).round().astype(int)
        lam, I = lam[k], I[k]
    sig0 = 10e-6                               # nominal; only R/sigma matters
    scale = np.nanmax(I)

    def model(l, logA, T0, ros):
        return np.exp(logA) * spectrum(l, T0, sig0, ros * sig0) / scale

    A0 = np.log(max(scale, 1e-300) / max(np.nanmax(
        spectrum(lam, T_guess, sig0, ros_guess * sig0)), 1e-300)) + np.log(scale)
    popt, pcov = curve_fit(model, lam, I / scale,
                           p0=[A0, T_guess, ros_guess],
                           bounds=([-80, 300, 0.05], [80, 40000, 6.0]), maxfev=20000)
    err = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(
        np.isfinite(pcov)) else np.full(3, np.nan)
    resid = 100 * (model(lam, *popt) / (I / scale) - 1)
    return dict(T_peak=float(popt[1]), R_over_sigma=float(popt[2]),
                T_err=float(err[1]), ros_err=float(err[2]),
                rms_resid=float(np.sqrt(np.nanmean(resid**2))))


def gauss_apparent(res, lam_lo, lam_hi):
    """Apparent T under the Gaussian assumption, via the lookup table:
    T0_fit * rho(R/sigma, xi).  Returns None if no Gaussian fit is available."""
    g = res.get("gauss")
    if not g:
        return None
    tab = get_ratio_table(lam_lo, lam_hi)
    xi_g = float(xi_window(g["T0"], lam_lo, lam_hi))
    return float(g["T0"] * ratio_from_table(tab, g["R_over_sigma"], xi_g))


def data_source(default_path, label, key):
    """File uploader. Nothing is loaded until the user uploads a file, or explicitly
    opts into the bundled example. Returns (bytes, name) or (None, None)."""
    up = st.file_uploader(label, type=["csv", "txt", "dat"], key=key)
    if up is not None:
        return up.getvalue(), up.name
    name = os.path.basename(default_path)
    if os.path.exists(default_path):
        if st.checkbox(f"use the bundled example ({name})", value=False, key=key + "_ex"):
            return _read_bytes(default_path), name
        st.info("Upload a file to begin — or tick the box to load the bundled example.")
    else:
        st.info("Upload a file to begin.")
    return None, None


class _SkipTab(Exception):
    """Bail out of one tab without halting the whole app (raise _SkipTab() would)."""


@contextlib.contextmanager
def tab_body():
    try:
        yield
    except _SkipTab:
        pass


# ============================================================ sidebar
st.sidebar.title("🔥 Hot-spot pyrometry")
st.sidebar.markdown(
    "Apparent-temperature bias for a Gaussian hot spot seen through a finite pinhole, "
    "and its correction via the universal $\\rho(R/\\sigma,\\xi)$ surface.")
st.sidebar.subheader("Spectrometer window")
lo_nm = st.sidebar.number_input("window low [nm]", 200.0, 5000.0, 575.0, step=25.0)
hi_nm = st.sidebar.number_input("window high [nm]", 200.0, 5000.0, 775.0, step=25.0)
lo, hi = lo_nm * 1e-9, hi_nm * 1e-9
if lo >= hi:
    st.sidebar.error("window low must be < high")
lam_c = float(np.sqrt(lo * hi))
st.sidebar.caption(f"$\\lambda_c=\\sqrt{{\\lambda_1\\lambda_2}}$ = {lam_c*1e9:.0f} nm  ·  "
                   f"$c_2/\\lambda_c$ = {C2/lam_c:.0f} K (so $\\xi=${C2/lam_c:.0f}$/T_0$)")

(tab_imp, tab_sim, tab_exp, tab_cmp, tab_prof, tab_spec,
 tab_uni, tab_batch, tab_about) = st.tabs(
    ["1 · 📥 COMSOL import", "2 · 🔬 Simulated SOP", "3 · 🔭 Experimental spectra",
     "4 · 📊 Compare", "📈 Single profile", "🔎 Spectrum → T(r)",
     "🎯 Gaussian & universal", "🌐 Universality (batch)", "📖 About / theory"])


# ============================================================ TAB 1: COMSOL IMPORT
with tab_imp, tab_body():
    st.subheader("1 · Import a COMSOL radial export → app series format")
    st.markdown(
        "Upload a COMSOL **1-D line-graph** export (`Radial_T_profile_*.txt`: column 0 = "
        "radius in metres, one temperature column per time). The times are read from the "
        "column headers when present; otherwise give the fixed sampling interval Δt. "
        "Download the converted file and use it in tab 2.")

    i0, i1 = st.columns([2, 1])
    with i0:
        ibytes, iname = data_source(os.path.join(HERE, "example_files",
                                                 "COMSOL_radial_T_100pulses_20umFWHM.txt"),
                                    "Upload COMSOL export (or an app CSV to inspect)",
                                    key="imp_f")
    with i1:
        imp_dt = st.number_input("Δt between steps [µs]", 1e-4, 1e4, 0.1, step=0.1,
                                 format="%.4f", key="imp_dt",
                                 help="used only when the file carries no times")
    if ibytes is None:
        raise _SkipTab()
    try:
        it, ir, iT = _series_arrays(ibytes, iname, imp_dt)
    except Exception as exc:
        st.error(f"could not read this file: {exc}")
        st.info("Tab 1 needs a **radial** export (column 0 = radius). The point-probe "
                "file (`Ttab_*.txt`, column 0 = time) is a different format.")
        raise _SkipTab()

    im = st.columns(5)
    im[0].metric("radial nodes", f"{ir.size}")
    im[1].metric("time steps", f"{iT.shape[1]}")
    im[2].metric("radius range", f"0 – {ir.max()*1e6:.1f} µm")
    im[3].metric("time range", f"{it.min():g} – {it.max():g} µs")
    im[4].metric("T range", f"{iT.min():.0f} – {iT.max():.0f} K")

    p0, p1 = st.columns(2)
    with p0:
        st.markdown("#### Radial profiles")
        fig, ax = plt.subplots(figsize=(6, 4.0))
        idx = np.linspace(0, iT.shape[1]-1, min(8, iT.shape[1])).round().astype(int)
        for c, k in zip(colormaps["inferno"](np.linspace(.1, .85, len(idx))), idx):
            ax.plot(ir*1e6, iT[:, k], color=c, lw=1.5, label=f"{it[k]:g}")
        ax.set_xlabel(r"$r$ (µm)"); ax.set_ylabel(r"$T(r)$ (K)")
        ax.legend(fontsize=7, title="time (µs)", ncol=2); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
    with p1:
        st.markdown("#### $T(r,t)$ map")
        fig, ax = plt.subplots(figsize=(6, 4.0))
        pm = ax.pcolormesh(it, ir*1e6, iT, shading="auto", cmap="inferno")
        fig.colorbar(pm, ax=ax, label="$T$ (K)")
        ax.set_xlabel("time (µs)"); ax.set_ylabel(r"$r$ (µm)")
        st.pyplot(fig, use_container_width=True)

    conv = ("# times = " + " ".join(f"{x:g}" for x in it) +
            "\n# radius_um, T[K] per time (µs); converted from " + str(iname) +
            "\n" + "\n".join(",".join(f"{v:.6g}" for v in row)
                             for row in np.column_stack([ir*1e6, iT])))
    st.session_state["tab1_series"] = dict(data=conv.encode("utf-8"),
                                           name=f"tab 1 · {iname}")
    d0, d1 = st.columns([1, 2])
    d0.download_button("⬇ download app-format series (CSV)", conv,
                       "series_for_tab2.csv", "text/csv", type="primary")
    d1.success("This series is now available in **tab 2** — no need to re-upload "
               "(the download is only if you want to keep it).")


# ============================================================ TAB 2: SIMULATED SOP
with tab_sim, tab_body():
    st.subheader("2 · Simulated SOP — interpolate, time-bin, and fit")
    st.markdown(
        "Load a series (the tab-1 output). The profiles are interpolated onto a finer "
        "time base, synthetic emission spectra are built through the pinhole, then "
        "**binned over your detector window** and fitted for apparent temperature. "
        "Gaussian profile fits are done on the same binned intervals.")

    handoff = st.session_state.get("tab1_series")
    s0, s1 = st.columns([2, 1])
    with s0:
        use_t1 = False
        if handoff is not None:
            use_t1 = st.checkbox(f"use the series from **{handoff['name']}**",
                                 value=True, key="sim_uset1")
        if use_t1:
            sbytes, sname = handoff["data"], handoff["name"]
            st.caption("Using the series imported in tab 1. Untick to load a file instead.")
        else:
            sbytes, sname = data_source(
                os.path.join(HERE, "synthetic_temperatures", "Tseries_diffusion.csv"),
                "Upload series CSV (from tab 1)", key="sim_f")
    with s1:
        sim_dt = st.number_input("Δt between steps [µs] (COMSOL only)", 1e-4, 1e4, 0.1,
                                 step=0.1, format="%.4f", key="sim_dt")
    if sbytes is None:
        raise _SkipTab()
    try:
        st_t, st_r, st_T = _series_arrays(sbytes, sname, sim_dt)
    except Exception as exc:
        st.error(f"could not read this file: {exc}"); raise _SkipTab()

    native = float(np.median(np.diff(np.sort(st_t)))) if st_t.size > 1 else 1.0
    c0, c1, c2, c3 = st.columns(4)
    R_um = c0.number_input("pinhole radius R [µm]", 0.1, 500.0, 15.0, step=1.0, key="sim_R")
    step_us = c1.number_input("interpolation step [µs]", 1e-4, 1e4,
                              float(f"{native/4:.4g}"), format="%.4f", key="sim_step",
                              help=f"file's native step is {native:g} µs")
    bin_us = c2.number_input("detector time bin [µs]", 1e-4, 1e4,
                             float(f"{native:.4g}"), format="%.4f", key="sim_bin",
                             help="spectra are averaged over this window before fitting")
    n_lam = c3.select_slider("wavelength points", [100, 200, 400], value=200, key="sim_nl")

    R = R_um * 1e-6
    if R > st_r.max()*1.0000001:
        st.warning(f"R = {R_um:g} µm exceeds the data extent ({st_r.max()*1e6:.2f} µm) — "
                   f"no truncation applied; effective radius {st_r.max()*1e6:.2f} µm.")

    try:
        with st.spinner("interpolating, building spectra and binning…"):
            sim = _run_sim(sbytes, sname, sim_dt, R_um, step_us, bin_us, lo, hi, n_lam)
    except Exception as exc:
        st.error(f"simulation failed: {exc}"); raise _SkipTab()

    st.session_state["sim_result"] = dict(
        t=sim["t_bin"], T_app=sim["T_app"], T_peak=sim["T_peak"],
        T_gauss=sim["T_gauss"], sigma=sim["sigma"], R_um=R_um, name=str(sname))

    m = st.columns(5)
    m[0].metric("interpolated steps", f"{sim['n_fine']}")
    m[1].metric("time bins", f"{len(sim['t_bin'])}")
    m[2].metric("samples per bin", f"{sim['counts'].mean():.1f}")
    m[3].metric("max peak T", f"{np.nanmax(sim['T_peak']):.0f} K")
    m[4].metric("max apparent T", f"{np.nanmax(sim['T_app']):.0f} K",
                f"{np.nanmax(sim['T_app'])-np.nanmax(sim['T_peak']):+.0f} K")

    st.markdown("#### Simulated SOP spectrogram (binned)")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    pm = ax.pcolormesh(sim["t_bin"], sim["lam"]*1e9, sim["spec_bin"],
                       shading="auto", cmap="inferno")
    fig.colorbar(pm, ax=ax, label="emission (a.u.)")
    ax.set_xlabel("time (µs)"); ax.set_ylabel("wavelength (nm)"); ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

    a0, a1 = st.columns(2)
    with a0:
        st.markdown("#### Temperature history")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(sim["t_bin"], sim["T_peak"], "-o", color="k", ms=3, lw=1.6,
                label="peak $T$ (data)")
        ax.plot(sim["t_bin"], sim["T_gauss"], "--", color="seagreen", lw=1.5,
                label="fitted Gaussian peak $T_0$")
        ax.plot(sim["t_bin"], sim["T_app"], "-", color="crimson", lw=1.8,
                label="apparent $T$ (binned spectra)")
        ax.plot(sim["t_bin"], sim["T_app_tab"], "-.", color="darkorange", lw=1.5,
                label="apparent $T$ (Gaussian + lookup)")
        ax.plot(sim["t_bin"], sim["T_edge"], ":", color="steelblue", lw=1.8,
                label=f"$T$ at edge (R={R_um:g} µm)")
        ax.set_xlabel("time (µs)"); ax.set_ylabel("temperature (K)")
        ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
    with a1:
        st.markdown("#### Fit geometry and bias")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(sim["t_bin"], R/np.array(sim["sigma"]), color="purple", lw=1.7,
                label=r"$R/\sigma$")
        ax.set_xlabel("time (µs)"); ax.set_ylabel(r"$R/\sigma$", color="purple")
        ax.tick_params(axis="y", labelcolor="purple"); ax.grid(alpha=.25)
        ax2 = ax.twinx()
        ax2.plot(sim["t_bin"], (np.array(sim["T_app"])/np.array(sim["T_peak"])-1)*100,
                 color="crimson", lw=1.7)
        ax2.set_ylabel("apparent-$T$ bias (%)", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Bin explorer")
    jb = st.slider("time bin", 0, len(sim["t_bin"])-1,
                   int(np.nanargmax(sim["T_peak"])), key="sim_j")
    st.write(f"**t = {sim['t_bin'][jb]:g} µs**  ·  {sim['counts'][jb]} interpolated samples")
    b0, b1 = st.columns(2)
    with b0:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.axvspan(0, R_um, color="skyblue", alpha=.20, label="pinhole")
        ax.plot(st_r*1e6, sim["prof_bin"][:, jb], "k-", lw=2, label="binned $T(r)$")
        sg, T0g = sim["sigma"][jb], sim["T_gauss"][jb]
        ax.plot(st_r*1e6, T0g*np.exp(-(st_r**2)/(2*sg**2)), "--", color="seagreen",
                lw=1.6, label=f"Gaussian fit ($\\sigma$={sg*1e6:.1f} µm)")
        ax.axhline(sim["T_app"][jb], color="crimson", ls="--", lw=1.3,
                   label=f"apparent {sim['T_app'][jb]:.0f} K")
        ax.set_xlabel(r"$r$ (µm)"); ax.set_ylabel("$T$ (K)")
        ax.legend(fontsize=7.5); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
    with b1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        sp = sim["spec_bin"][:, jb]
        ax.plot(sim["lam"]*1e9, sp, "k-", lw=2, label="binned spectrum")
        ax.plot(sim["lam"]*1e9, planck(sim["lam"], sim["T_app"][jb], sim["A_app"][jb]),
                "r--", lw=1.6, label=f"Planck fit {sim['T_app'][jb]:.0f} K")
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("emission (a.u.)")
        ax.legend(fontsize=7.5); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)

    out = np.column_stack([sim["t_bin"], sim["T_peak"], sim["T_gauss"], sim["T_app"],
                           sim["T_app_tab"], sim["T_edge"], np.array(sim["sigma"])*1e6])
    csv = ("time_us,T_peak_K,T_gauss_K,T_app_K,T_app_lookup_K,T_edge_K,sigma_um\n"
           + "\n".join(",".join(f"{v:.6g}" for v in row) for row in out))
    st.download_button("⬇ download simulated history (CSV)", csv,
                       "simulated_sop_history.csv", "text/csv")
# ============================================================ TAB 2: SINGLE PROFILE
with tab_prof, tab_body():
    st.subheader("Single radial profile → forward evaluation")
    c0, c1 = st.columns([2, 1])
    with c0:
        pbytes, pname = data_source(os.path.join(HERE, "synthetic_temperatures", "Tprofile.csv"),
                                    "Upload a T(r) profile (radius[µm], T[K])", key="pf")
    with c1:
        full = st.checkbox("use full profile (no pinhole cut)", value=False, key="pf_full")
        Rp_um = st.number_input("pinhole R [µm]", 0.5, 500.0, 25.0, step=1.0,
                                key="pf_R", disabled=full)
    if pbytes is not None:
        r, T = _load_single(pbytes, pname)
        Rp = None if full else Rp_um * 1e-6
        res = evaluate_profile(r, T, lo, hi, R=Rp)
        g = res.get("gauss") or {}
        mm = st.columns(5)
        mm[0].metric("peak T(r)", f"{res['T_peak']:.0f} K")
        mm[1].metric("apparent T", f"{res['T_app']:.0f} K",
                     f"{res['bias_frac']*100:+.1f} %")
        mm[2].metric("Gaussian fit σ", f"{g.get('sigma', float('nan'))*1e6:.1f} µm")
        mm[3].metric("universal recovered T0", f"{g.get('T0_universal', float('nan')):.0f} K",
                     f"{g.get('recover_err', float('nan')):+.0f} K vs peak")
        mm[4].metric("ξ", f"{res['xi']:.2f}")
        fig = plt.figure(figsize=(11, 4.3))
        plot_profile_eval(fig, res, lo, hi, t_app_gauss=gauss_apparent(res, lo, hi))
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        st.pyplot(fig, use_container_width=True)
        st.caption("Left: data (black), Gaussian fitted inside the pinhole (green), pinhole "
                   "shaded blue. Right: **collected spectrum** = numerical integral of the "
                   "data $T(r)$ (black); **best-fit single Planck** at $T_{app}$ (red); the "
                   "analytic Gaussian-surrogate spectrum (green, dotted).")


# ============================================================ TAB 3: GAUSSIAN & UNIVERSAL
with tab_uni, tab_body():
    st.subheader("Gaussian model & the universal correction")
    c0, c1, c2 = st.columns(3)
    T0 = c0.slider("true peak $T_0$ [K]", 1000, 12000, 5500, step=100)
    sig_um = c1.slider("Gaussian σ [µm]", 1.0, 40.0, 10.0, step=0.5)
    R_um3 = c2.slider("pinhole R [µm]", 0.5, 80.0, 15.0, step=0.5)
    ros = R_um3 / sig_um
    xi = float(xi_window(T0, lo, hi))
    tab = get_ratio_table(lo, hi)
    rho = float(ratio_from_table(tab, ros, xi))
    T_app = T0 * rho
    T_rec = correct_temperature(T_app, ros, lo, hi)
    mm = st.columns(4)
    mm[0].metric("R/σ", f"{ros:.2f}")
    mm[1].metric("ξ = c₂/(λ_c T₀)", f"{xi:.2f}")
    mm[2].metric("apparent T (table)", f"{T_app:.0f} K", f"{(rho-1)*100:+.1f} %")
    mm[3].metric("inversion recovers", f"{T_rec:.0f} K", f"{T_rec-T0:+.0f} K")

    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Universal master curve (saturated limit)")
        fig, ax = plt.subplots(figsize=(6, 4.3))
        xg = np.linspace(1.2, 12, 300)
        ax.plot(xg, (wien_saturated_ratio(xg)-1)*100, "r--", lw=1.6,
                label=r"$\xi E_1(\xi)e^{\xi}-1$")
        ax.plot([xi], [(wien_saturated_ratio(xi)-1)*100], "ko", ms=8,
                label=f"this spot (ξ={xi:.2f})")
        ax.set_xlabel(r"$\xi$"); ax.set_ylabel("saturated bias (%)")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
    with cB:
        st.markdown("#### Lookup table $\\rho(R/\\sigma,\\xi)$")
        fig, ax = plt.subplots(figsize=(6, 4.3))
        pm = ax.pcolormesh(tab["xi"], tab["ros"], tab["rho"], shading="auto",
                           cmap="viridis")
        fig.colorbar(pm, ax=ax, label=r"$\rho=T_{\rm app}/T_0$")
        ax.plot([xi], [ros], "o", color="crimson", ms=9, mec="w")
        ax.set_xlabel(r"$\xi$"); ax.set_ylabel(r"$R/\sigma$")
        ax.set_ylim(tab["ros"][0], min(tab["ros"][-1], max(3, ros*1.5)))
        st.pyplot(fig, use_container_width=True)
    st.caption("The lookup table (right) is the full surface; the master curve (left) is its "
               "$R/\\sigma\\to\\infty$ limit. The app corrects data by interpolating the table.")


# ============================================================ TAB 4: BATCH UNIVERSALITY
with tab_batch, tab_body():
    st.subheader("Universality across $(T_0,\\sigma)$ families")
    st.caption("Sweep families to check that the bias collapses onto $\\Phi(R/\\sigma,\\xi)$ "
               "and the master curve.")
    c0, c1, c2 = st.columns(3)
    T0_txt = c0.text_input("T₀ values [K]", "3000, 5500, 8000")
    sig_txt = c1.text_input("σ values [µm]", "10, 20")
    ros_txt = c2.text_input("R/σ sweep", "0.25, 0.5, 1, 1.5, 2, 3, 5")
    if st.button("Run batch", type="primary"):
        try:
            T0s = [float(x) for x in T0_txt.split(",") if x.strip()]
            sigs = [float(x)*1e-6 for x in sig_txt.split(",") if x.strip()]
            ross = [float(x) for x in ros_txt.split(",") if x.strip()]
            CONFIGS = []
            for T0 in T0s:
                for sg in sigs:
                    group = f"$T_0$={T0:.0f} K, $\\sigma$={sg*1e6:.0f} µm"
                    for rs in ross:
                        CONFIGS.append(dict(group=group, T0=T0, sigma=sg,
                                            R=rs*sg, fit_lo=lo, fit_hi=hi))
            with st.spinner(f"running {len(CONFIGS)} configs…"):
                results = [run_config(cfg) for cfg in CONFIGS]
            fig = plt.figure(figsize=(13, 9))
            plot_comparison(fig, results, lo, hi)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            st.pyplot(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"could not run batch: {exc}")


# ============================================================ TAB 5: ABOUT
with tab_about, tab_body():
    st.markdown(r"""
### What this app computes

A hot spot with a Gaussian radial temperature profile
$T(r)=T_{\rm peak}\,e^{-r^2/2\sigma^2}$ is viewed through a pinhole of radius $R$. The
detector collects the **area-weighted** emission from all radii inside the pinhole,

$$I(\lambda)=\int_0^{R} 2\pi r\,\epsilon\,B(\lambda,T(r))\,\mathrm{d}r .$$

Fitting a **single Planck curve** over a fixed window $[\lambda_1,\lambda_2]$ returns an
*apparent* temperature $T_{\rm app}<T_{\rm peak}$ (cooler outer annuli pull the fit down).

The bias depends on only two dimensionless groups (size cancels):

* geometry $\;R/\sigma$
* window $\;\xi = c_2/(\lambda_c T_{\rm peak})$, with $c_2=hc/k_B$ and
  $\lambda_c=\sqrt{\lambda_1\lambda_2}$.

So $T_{\rm app}/T_{\rm peak}=\rho(R/\sigma,\xi)$. The **lookup table** tabulates this
surface from the exact forward model; the closed form $\rho_\infty(\xi)=\xi E_1(\xi)e^{\xi}$
is its $R/\sigma\to\infty$ limit. Inverting $\rho$ recovers $T_{\rm peak}$ from a measurement.

### The four time-series curves
* **actual peak $T$** — $\max_r T(r)$ from the data.
* **fitted Gaussian peak $T_0$** — Gaussian fitted to the profile *inside the pinhole*.
* **apparent $T$** — what a single-Planck fit reports (biased low).
* **$T$ at pinhole edge** — $T(R)$, i.e. the temperature at the aperture rim.

### T_app methods (time-series tab)
* **table** *(default)* — fit a Gaussian, interpolate $\rho$ from the precomputed
  $(R/\sigma,\xi)$ grid. Fast (O(1)) and accurate for any $R/\sigma$.
* **gaussian** — Planck-fit the fitted Gaussian's analytic emission.
* **numerical** — Planck-fit the emission integrated straight from the tabulated profile
  (exact ground truth; slowest).

### Input file format
Time series (wide CSV): first column radius [µm]; each further column $T$ [K] at one time;
times in a comment line `# times = t1 t2 ...`. Single profile: two columns radius[µm], T[K].

The Gaussian approximation is reliable when the *pinhole-region* profile is Gaussian; the
predicted-vs-true apparent-temperature gap gauges any departure.
""")
    st.caption("Engine: planck_model.py · figures: planck_plots.py · UI: streamlit_app.py")


# ============================================================ TAB: MEASURED SPECTRUM
with tab_spec, tab_body():
    st.subheader("Measured emission spectrum → apparent T → radial $T(r)$")
    st.markdown(
        "Fit a measured spectrum over the spectrometer window to get the **apparent "
        "temperature**, then invert the pinhole bias under the Gaussian assumption to "
        "recover the **true peak temperature** and the radial profile.")

    u0, u1, u2, u3 = st.columns([2, 1, 1, 1])
    with u0:
        xbytes, xname = data_source(os.path.join(HERE, "synthetic_temperatures", "SOP_spectrum.csv"),
                                    "Upload spectrum (2 columns: wavelength, intensity)",
                                    key="sp_f")
    lam_unit = u1.selectbox("wavelength unit", ["nm", "µm", "m", "Å"], index=0, key="sp_u")
    lam_col = u2.number_input("wavelength column", 0, 20, 0, step=1, key="sp_lc")
    int_col = u3.number_input("intensity column", 0, 20, 1, step=1, key="sp_ic")
    if xbytes is None:
        raise _SkipTab()
    uscale = {"nm": 1e-9, "µm": 1e-6, "m": 1.0, "Å": 1e-10}[lam_unit]

    try:
        lam_all, I_all = _read_xy(xbytes, xname, int(lam_col), int(int_col), uscale)
        order = np.argsort(lam_all)
        lam_all, I_all = lam_all[order], I_all[order]
    except Exception as exc:
        st.error(f"could not read the spectrum: {exc}"); raise _SkipTab()

    m_fit = (lam_all >= lo) & (lam_all <= hi) & np.isfinite(I_all) & (I_all > 0)
    if m_fit.sum() < 5:
        st.error(f"only {int(m_fit.sum())} usable points inside the window "
                 f"{lo*1e9:.0f}–{hi*1e9:.0f} nm (data spans "
                 f"{lam_all.min()*1e9:.0f}–{lam_all.max()*1e9:.0f} nm). "
                 "Adjust the window in the sidebar or check the wavelength unit.")
        raise _SkipTab()

    # ---- geometry ----
    g0, g1, g2 = st.columns([1, 1, 2])
    D_um = g0.number_input("pinhole diameter [µm]", 0.1, 1000.0, 30.0, step=1.0,
                           key="sp_D", help="collection aperture; R = D/2")
    R_sp = 0.5 * D_um * 1e-6
    mode = g1.radio("spot width", ["assume saturated", "known σ", "fit from shape"],
                    key="sp_mode",
                    help="A single T_app cannot fix both T_peak and σ. Supply σ, assume "
                         "saturation, or fit the spectral SHAPE for both (needs good SNR "
                         "and an under-filled pinhole).")
    shape_fit = None
    if mode == "fit from shape":
        g2.info("Fitting **(amplitude, $T_{peak}$, $R/\\sigma$)** to the spectrum shape. "
                "The departure from a pure Planck encodes the temperature spread — no "
                "imaging needed. Requires SNR ≳ 0.1 % and $R/\\sigma\\lesssim1.5$; above "
                "saturation the shape stops depending on $R/\\sigma$.")
    elif mode == "known σ":
        sig_um = g2.number_input("Gaussian σ [µm] (from imaging)", 0.05, 500.0, 10.0,
                                 step=0.5, key="sp_sig")
        ros_sp = R_sp / (sig_um * 1e-6)
        g2.caption(f"R/σ = {ros_sp:.2f}"
                   + ("  ·  saturated" if ros_sp >= 1.5 else
                      "  ·  **under-filled** — the bias still depends on R/σ"))
    else:
        ros_sp = 5.0            # deep in the saturated plateau: rho depends on xi only
        g2.info("Saturated: only ξ matters, so $T_{peak}$ follows from $T_{app}$ alone. "
                "σ is then **not** determined — it is only constrained to σ ≲ R/1.5. "
                "Pick a σ below to draw a representative profile.")
        sig_um = g2.number_input("σ for display only [µm]", 0.05, 500.0,
                                 float(f"{R_sp*1e6/1.5:.2f}"), step=0.5, key="sp_sigd")

    # ---- fit + invert ----
    try:
        T_app_sp, A_sp = fit_temperature(lam_all[m_fit], I_all[m_fit], T_guess=3000.0)
        if mode == "fit from shape":
            with st.spinner("fitting the spectral shape…"):
                shape_fit = fit_shape_3param(lam_all[m_fit], I_all[m_fit],
                                             T_guess=max(T_app_sp, 500.0))
            ros_sp = shape_fit["R_over_sigma"]
            T_peak_sp = shape_fit["T_peak"]
            sig_um = (R_sp / ros_sp) * 1e6 if ros_sp > 0 else np.nan
            info = dict(rho=T_app_sp / T_peak_sp if T_peak_sp else np.nan,
                        xi=float(xi_window(T_peak_sp, lo, hi)))
        else:
            T_peak_sp, info = correct_temperature(T_app_sp, ros_sp, lo, hi,
                                                  return_info=True)
    except Exception as exc:
        st.error(f"fit/inversion failed: {exc}"); raise _SkipTab()

    k = st.columns(5)
    k[0].metric("points fitted", f"{int(m_fit.sum())}")
    k[1].metric("apparent T", f"{T_app_sp:.0f} K")
    k[2].metric("recovered peak T", f"{T_peak_sp:.0f} K",
                f"{T_peak_sp - T_app_sp:+.0f} K")
    k[3].metric("correction ρ", f"{info['rho']:.3f}")
    k[4].metric("ξ, R/σ", f"{info['xi']:.2f}, {ros_sp:.2f}")

    if shape_fit is not None:
        s0, s1, s2 = st.columns(3)
        s0.metric("fitted R/σ", f"{shape_fit['R_over_sigma']:.2f}",
                  f"± {shape_fit['ros_err']:.2f}" if np.isfinite(shape_fit['ros_err']) else None)
        s1.metric("implied σ", f"{sig_um:.2f} µm")
        s2.metric("shape-fit residual", f"{shape_fit['rms_resid']:.3f} %")
        if shape_fit["R_over_sigma"] > 1.45:
            st.warning(
                f"fitted R/σ = {shape_fit['R_over_sigma']:.2f} is at/above saturation — "
                "there the spectrum shape no longer depends on R/σ, so σ is **not** "
                "constrained by these data (T_peak above is still reliable). Treat the "
                "width as an upper bound and prefer imaging, or use a smaller pinhole.")
        elif shape_fit["rms_resid"] > 0.5:
            st.warning(
                f"residual scatter ({shape_fit['rms_resid']:.2f} %) is large compared with "
                "the shape signature (≲0.3 %), so R/σ is likely noise-dominated. "
                "T_peak remains far more robust than the width.")

    # ---- spectrum + fit ----
    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### Spectrum and Planck fit")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot(lam_all*1e9, I_all, "-", color="0.6", lw=1.0, label="measured (all)")
        ax.plot(lam_all[m_fit]*1e9, I_all[m_fit], "k-", lw=2, label="fitted range")
        ax.plot(lam_all[m_fit]*1e9, planck(lam_all[m_fit], T_app_sp, A_sp), "r--", lw=1.8,
                label=f"Planck fit, $T_{{app}}$={T_app_sp:.0f} K")
        ax.axvspan(lo*1e9, hi*1e9, color="orange", alpha=.18, label="fit window")
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("intensity (a.u.)")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)

        resid = 100*(planck(lam_all[m_fit], T_app_sp, A_sp)/I_all[m_fit] - 1)
        st.caption(f"fit residual: mean {resid.mean():+.2f} %, RMS {np.sqrt((resid**2).mean()):.2f} %")

    # ---- reconstructed radial profile ----
    sig_m = sig_um * 1e-6
    r_plot = np.linspace(0, max(R_sp*1.6, 3*sig_m), 400)
    T_of_r = T_peak_sp * np.exp(-r_plot**2 / (2*sig_m**2))
    with cB:
        st.markdown("#### Reconstructed radial temperature")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.axvspan(0, R_sp*1e6, color="skyblue", alpha=.20,
                   label=f"pinhole ($R$={R_sp*1e6:.1f} µm)")
        ax.axvline(R_sp*1e6, color="steelblue", ls=":", lw=1)
        ax.plot(r_plot*1e6, T_of_r, "k-", lw=2, label="inferred $T(r)$")
        ax.axhline(T_peak_sp, color="grey", ls=":", lw=1,
                   label=f"peak {T_peak_sp:.0f} K")
        ax.axhline(T_app_sp, color="crimson", ls="--", lw=1.4,
                   label=f"apparent {T_app_sp:.0f} K")
        ax.plot(R_sp*1e6, T_peak_sp*np.exp(-R_sp**2/(2*sig_m**2)), "o",
                color="steelblue", ms=7,
                label=f"edge {T_peak_sp*np.exp(-R_sp**2/(2*sig_m**2)):.0f} K")
        ax.set_xlabel(r"$r$ (µm)"); ax.set_ylabel(r"$T(r)$ (K)")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
        if mode == "assume saturated":
            st.caption("σ is assumed for display only — the width is not constrained by "
                       "the spectrum. $T_{peak}$ above **is** determined.")

    prof = np.column_stack([r_plot*1e6, T_of_r])
    csv = ("# inferred Gaussian profile: T_peak=%.1f K, sigma=%.3f um, R=%.3f um\n"
           "# radius_um, T_K\n" % (T_peak_sp, sig_um, R_sp*1e6)) + "\n".join(
        ",".join(f"{v:.6g}" for v in row) for row in prof)
    st.download_button("⬇ download inferred T(r) (CSV)", csv, "inferred_profile.csv",
                       "text/csv",
                       help="two-column profile — reusable in the Single profile tab")


# ============================================================ TAB 3: EXPERIMENTAL SPECTRA
with tab_exp, tab_body():
    st.subheader("3 · Experimental emission spectra → apparent T vs time")
    st.markdown(
        "Upload a **wide spectra file**: column 0 = wavelength [nm], each further column "
        "= measured intensity at one time, with the times in a `# times = …` comment. "
        "A Planck (free amplitude + temperature) is fitted to every column over the "
        "spectrometer window.")

    e0, e1 = st.columns([3, 1])
    with e0:
        ebytes, ename = data_source(os.path.join(HERE, "example_files",
                                                 "experimental_SOP_spectra_100pulses.csv"),
                                    "Upload experimental spectra (wide format)",
                                    key="exp_f")
    if ebytes is None:
        raise _SkipTab()
    with e1:
        elabel = st.text_input("time unit label", "µs", key="exp_u")

    try:
        with st.spinner("fitting Planck to each spectrum…"):
            ex = _fit_spectra_series(ebytes, ename, lo, hi)
    except Exception as exc:
        st.error(f"could not process this file: {exc}"); raise _SkipTab()

    st.session_state["exp_result"] = dict(t=ex["times"], T_app=ex["T_app"],
                                          A_app=ex["A_app"], name=str(ename))
    good = np.isfinite(ex["T_app"])
    em = st.columns(5)
    em[0].metric("spectra", f"{ex['I'].shape[1]}")
    em[1].metric("fitted", f"{int(good.sum())}")
    em[2].metric("points in window", f"{int(ex['mask'].sum())}")
    em[3].metric("median apparent T", f"{np.nanmedian(ex['T_app']):.0f} K")
    em[4].metric("T range", f"{np.nanmin(ex['T_app']):.0f} – {np.nanmax(ex['T_app']):.0f} K")

    st.markdown("#### Measured spectrogram")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    pm = ax.pcolormesh(ex["times"], ex["lam"]*1e9, ex["I"], shading="auto", cmap="inferno")
    fig.colorbar(pm, ax=ax, label="intensity (a.u.)")
    ax.axhline(lo*1e9, color="cyan", ls="--", lw=1)
    ax.axhline(hi*1e9, color="cyan", ls="--", lw=1)
    ax.set_xlabel(f"time ({elabel})"); ax.set_ylabel("wavelength (nm)"); ax.invert_yaxis()
    st.pyplot(fig, use_container_width=True)

    x0, x1 = st.columns(2)
    with x0:
        st.markdown("#### Apparent temperature")
        fig, ax = plt.subplots(figsize=(6, 4.0))
        ax.plot(ex["times"][good], ex["T_app"][good], "o-", color="crimson", ms=3, lw=1.2)
        ax.axhline(np.nanmedian(ex["T_app"]), color="k", ls=":", lw=1,
                   label=f"median {np.nanmedian(ex['T_app']):.0f} K")
        ax.set_xlabel(f"time ({elabel})"); ax.set_ylabel("apparent $T$ (K)")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)
    with x1:
        st.markdown("#### Fitted amplitude / emissivity")
        fig, ax = plt.subplots(figsize=(6, 4.0))
        ax.plot(ex["times"][good], ex["A_app"][good], "o-", color="seagreen", ms=3, lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel(f"time ({elabel})"); ax.set_ylabel("fitted amplitude (a.u.)")
        ax.grid(alpha=.25, which="both")
        st.pyplot(fig, use_container_width=True)

    st.markdown("#### Spectrum inspector")
    je = st.slider("spectrum", 0, ex["I"].shape[1]-1,
                   int(np.nanargmax(np.where(good, ex["T_app"], -np.inf))), key="exp_j")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(ex["lam"]*1e9, ex["I"][:, je], "-", color="0.6", lw=1, label="measured (all)")
    ax.plot(ex["lam"][ex["mask"]]*1e9, ex["I"][ex["mask"], je], "k-", lw=1.8,
            label="fit window")
    if np.isfinite(ex["T_app"][je]):
        ax.plot(ex["lam"][ex["mask"]]*1e9,
                planck(ex["lam"][ex["mask"]], ex["T_app"][je], ex["A_app"][je]),
                "r--", lw=1.6, label=f"Planck fit {ex['T_app'][je]:.0f} K")
    ax.axvspan(lo*1e9, hi*1e9, color="orange", alpha=.15)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("intensity (a.u.)")
    ax.set_title(f"t = {ex['times'][je]:g} {elabel}")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    st.pyplot(fig, use_container_width=True)

    csv = ("time,T_app_K,amplitude\n" + "\n".join(
        ",".join(f"{v:.6g}" for v in row)
        for row in np.column_stack([ex["times"], ex["T_app"], ex["A_app"]])))
    st.download_button("⬇ download experimental apparent T (CSV)", csv,
                       "experimental_apparent_T.csv", "text/csv")


# ============================================================ TAB 4: COMPARE
with tab_cmp, tab_body():
    st.subheader("4 · Simulation vs experiment")
    sim_r = st.session_state.get("sim_result")
    exp_r = st.session_state.get("exp_result")
    if sim_r is None or exp_r is None:
        missing = []
        if sim_r is None: missing.append("**tab 2** (simulated SOP)")
        if exp_r is None: missing.append("**tab 3** (experimental spectra)")
        st.info("Run " + " and ".join(missing) + " first — their results are compared here.")
    else:
        st.caption(f"simulation: `{sim_r['name']}`  ·  experiment: `{exp_r['name']}`")
        d0, d1, d2 = st.columns(3)
        delay = d0.number_input("shift experiment by [time units]", -1e5, 1e5, 0.0,
                                step=0.1, format="%.4f", key="cmp_d")
        show_peak = d1.checkbox("show simulated peak $T$", value=True, key="cmp_pk")
        corr_exp = d2.checkbox("correct experiment → peak $T$", value=False, key="cmp_cor",
                               help="apply the universal inversion to the experimental "
                                    "apparent T using the simulation's R/σ")
        te = np.asarray(exp_r["t"], float) + delay
        Te = np.asarray(exp_r["T_app"], float)
        tsm = np.asarray(sim_r["t"], float)
        Tsm = np.asarray(sim_r["T_app"], float)

        try:
            t_min = max(np.nanmin(te), np.nanmin(tsm))
            t_max = min(np.nanmax(te), np.nanmax(tsm))
            if not (t_max > t_min):
                raise ValueError("the two records do not overlap in time — adjust the shift")
            gs = np.isfinite(tsm) & np.isfinite(Tsm)
            sim_at_exp = np.interp(te, tsm[gs], Tsm[gs], left=np.nan, right=np.nan)
            ok = np.isfinite(sim_at_exp) & np.isfinite(Te) & (te >= t_min) & (te <= t_max)
            if ok.sum() < 2:
                raise ValueError("fewer than 2 overlapping points")
            resid = Te[ok] - sim_at_exp[ok]
            chi2 = float(np.sum(resid**2))
            r_p = float(np.corrcoef(Te[ok], sim_at_exp[ok])[0, 1])
            k = st.columns(5)
            k[0].metric("overlapping points", f"{int(ok.sum())}")
            k[1].metric("mean Δ (exp − sim)", f"{resid.mean():+.0f} K")
            k[2].metric("RMS Δ", f"{np.sqrt((resid**2).mean()):.0f} K")
            k[3].metric("reduced χ²", f"{chi2/resid.size:.3e}")
            k[4].metric("Pearson r / R²", f"{r_p:.3f} / {r_p**2:.3f}")
        except Exception as exc:
            st.warning(f"comparison statistics unavailable: {exc}")
            ok = None

        fig, ax = plt.subplots(figsize=(11.5, 4.6))
        if show_peak:
            ax.plot(tsm, sim_r["T_peak"], "-", color="k", lw=1.3, alpha=.65,
                    label="simulated peak $T$")
        ax.plot(tsm, Tsm, "-", color="steelblue", lw=2, label="simulated apparent $T$")
        ax.plot(te, Te, "o", color="crimson", ms=3.5, label="experimental apparent $T$")
        if corr_exp:
            tabr = get_ratio_table(lo, hi)
            ros = np.interp(te, tsm, sim_r["R_um"]*1e-6/np.asarray(sim_r["sigma"], float))
            Tc = np.array([correct_temperature(t_, rr, lo, hi)
                           if np.isfinite(t_) and np.isfinite(rr) and rr > 0 else np.nan
                           for t_, rr in zip(Te, ros)])
            ax.plot(te, Tc, "s", color="darkorange", ms=3.5,
                    label="experiment corrected → peak $T$")
        ax.set_xlabel("time"); ax.set_ylabel("temperature (K)")
        ax.legend(fontsize=8.5, ncol=2); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)

        if ok is not None and ok.any():
            st.markdown("#### Residual (experiment − simulation)")
            fig, ax = plt.subplots(figsize=(11.5, 2.8))
            ax.plot(te[ok], resid, "o-", color="black", ms=3, lw=1)
            ax.axhline(0, color="grey", ls=":", lw=.8)
            ax.set_xlabel("time"); ax.set_ylabel(r"$\Delta T$ (K)")
            ax.grid(alpha=.25)
            st.pyplot(fig, use_container_width=True)
