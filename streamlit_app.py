"""
Grey-body pyrometry of a Gaussian hot spot -- Streamlit app.

Tabs:
  * Time series (main)   -- evaluate a sequence of radial T(r) snapshots.
  * Measured spectrum    -- fit an experimental spectrum, invert to peak T and T(r).
  * Single profile       -- one snapshot: apparent T, spectrum, Gaussian comparison.
  * Gaussian & universal -- explore the analytic model, master curve, lookup table.
  * Universality (batch) -- collapse of the bias across (T0, sigma) families.
  * About / theory       -- what the app computes and how.

All physics is in planck_model.py; shared figures in planck_plots.py.
Run locally:   streamlit run streamlit_app.py
"""
import os
import io
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
                          parse_comsol_line_graph,
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
    """File uploader with a bundled-sample fallback; returns (bytes, name)."""
    up = st.file_uploader(label, type=["csv", "txt", "dat"], key=key)
    if up is not None:
        return up.getvalue(), up.name
    name = os.path.basename(default_path)
    if os.path.exists(default_path):
        st.caption(f"Using bundled sample **{name}** (upload a file to replace).")
        return _read_bytes(default_path), name
    st.warning(f"No file uploaded and bundled sample {name} not found.")
    return None, None


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

(tab_ts, tab_spec, tab_prof, tab_uni, tab_batch, tab_about) = st.tabs(
    ["⏱ Time series", "🔭 Measured spectrum → T(r)", "📈 Single profile",
     "🎯 Gaussian & universal", "🌐 Universality (batch)", "📖 About / theory"])


# ============================================================ TAB 1: TIME SERIES
with tab_ts:
    st.subheader("Time series of radial temperature profiles")
    st.markdown(
        "Upload the app **wide CSV** (column 0 = radius [µm]; further columns = $T$ [K] per "
        "time; times in a `# times = …` comment) **or a COMSOL 1-D line-graph export** — "
        "COMSOL files are converted automatically.")

    c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
    with c0:
        fbytes, fname = data_source(os.path.join(HERE, "sample_Tseries_diffusion.csv"),
                                    "Upload time-series CSV or COMSOL .txt", key="ts")
    with c1:
        R_um = st.number_input("pinhole R [µm]", 0.5, 500.0, 25.0, step=1.0, key="ts_R")
    with c2:
        method = st.selectbox("T_app method", ["table", "gaussian", "numerical"],
                              index=0, key="ts_m",
                              help="table: fast + exact across R/σ; gaussian: analytic "
                                   "Gaussian fit; numerical: integrate the real profile.")
    with c3:
        tlabel = st.text_input("time axis label", "time (µs)", key="ts_t")

    if fbytes is None:
        st.stop()

    # COMSOL line-graph files carry no times -> snapshots at a fixed interval dt
    is_comsol = _is_comsol(fbytes)
    dt_us = 0.0
    if is_comsol:
        cc0, cc1 = st.columns([1, 3])
        dt_us = cc0.number_input("COMSOL Δt between steps [µs]", 1e-4, 1e4, 0.1,
                                 step=0.1, format="%.4f", key="ts_dt")
        cc1.info("COMSOL line-graph detected — times aren't in the file, so they're set to "
                 "0, Δt, 2Δt, … from the fixed sampling interval. Set Δt (µs) for this run; "
                 "replace with a time-labelled export later and it will be read automatically.")

    R = R_um * 1e-6
    try:
        times, r, T_cols = _series_arrays(fbytes, fname, dt_us)
        n_t = T_cols.shape[1]
        with st.spinner("evaluating series…"):
            s = _eval_series(fbytes, fname, R, method, lo, hi, dt_us)
            s_num = s if method == "numerical" else _eval_series(
                fbytes, fname, R, "numerical", lo, hi, dt_us)
    except Exception as exc:
        st.error(f"could not read this file: {exc}")
        st.stop()

    # ---- headline metrics (bias/apparent are always the true numerical fit) ----
    jpk = int(np.nanargmax(s_num["T_peak"]))
    bias_num = (s_num["T_app"]/s_num["T_peak"] - 1)*100
    approx_gap = float(np.nanmax(np.abs(s["T_app"] - s_num["T_app"])))
    m = st.columns(5)
    m[0].metric("snapshots", f"{n_t}")
    m[1].metric("max peak T", f"{np.nanmax(s_num['T_peak']):.0f} K")
    m[2].metric("apparent T @ hottest (numerical)", f"{s_num['T_app'][jpk]:.0f} K")
    m[3].metric("max bias (numerical)", f"{np.nanmin(bias_num):+.1f} %",
                help="apparent (numerical) vs true peak")
    if method != "numerical":
        m[4].metric("max Gaussian error", f"{approx_gap:.0f} K",
                    help=f"max |{METHOD_LABEL[method]} − numerical| apparent-T over the run")
    else:
        m[4].metric("edge T @ hottest", f"{s_num['T_edge'][jpk]:.0f} K")

    # ---- main history plot ----
    st.markdown("#### Temperature history through the pinhole")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    t = s_num["times"]
    ax.fill_between(t, s_num["T_app"], s_num["T_peak"], color="crimson", alpha=.07,
                    label="measurement bias (numerical)")
    ax.plot(t, s_num["T_peak"], "-o", color="k", ms=3, lw=1.9, label="actual peak $T$ (data)")
    ax.plot(t, s_num["T_gauss"], "--", color="seagreen", lw=1.6, label="fitted Gaussian peak $T_0$")
    ax.plot(t, s_num["T_app"], "-", color="crimson", lw=2.0,
            label="apparent $T$ (numerical, true fit)")
    if method != "numerical":
        ax.plot(t, s["T_app"], "-.", color="darkorange", lw=1.8,
                label=f"apparent $T$ (Gaussian, {METHOD_LABEL[method]})")
    ax.plot(t, s_num["T_edge"], ":", color="steelblue", lw=2.2,
            label=f"$T$ at pinhole edge (R={R_um:g} µm)")
    ax.set_xlabel(tlabel); ax.set_ylabel("temperature (K)")
    ax.legend(ncol=2, fontsize=9); ax.grid(alpha=.25)
    st.pyplot(fig, use_container_width=True)

    # ---- space-time heatmap + spectra overlay ----
    cA, cB = st.columns(2)
    with cA:
        st.markdown("#### $T(r,t)$ map")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        pm = ax.pcolormesh(t, r*1e6, T_cols, shading="auto", cmap="inferno")
        ax.axhline(R_um, color="cyan", ls="--", lw=1.2, label=f"pinhole R={R_um:g} µm")
        fig.colorbar(pm, ax=ax, label="$T$ (K)")
        ax.set_ylim(0, min(r.max()*1e6, R_um*3))
        ax.set_xlabel(tlabel); ax.set_ylabel(r"$r$ (µm)")
        ax.legend(fontsize=8, loc="upper right")
        st.pyplot(fig, use_container_width=True)
    with cB:
        st.markdown("#### Collected emission spectra vs time")
        fig, ax = plt.subplots(figsize=(6, 4.2))
        lam = np.logspace(np.log10(300e-9), np.log10(3e-6), 240)
        idx = np.linspace(0, n_t-1, min(7, n_t)).round().astype(int)
        cols = colormaps["inferno"](np.linspace(0.1, 0.9, len(idx)))
        for k, c in zip(idx, cols):
            F = spectrum_from_profile(lam, r, T_cols[:, k], R)
            if F.max() > 0:
                ax.plot(lam*1e6, F/F.max(), color=c, lw=1.6, label=f"{t[k]:g}")
        ax.axvspan(lo*1e6, hi*1e6, color="orange", alpha=.2)
        ax.set_xlim(0, 3); ax.set_xlabel(r"$\lambda$ (µm)")
        ax.set_ylabel(r"$F/F_{\max}$ (per snapshot)")
        ax.legend(fontsize=7, title=tlabel, ncol=2); ax.grid(alpha=.25)
        st.pyplot(fig, use_container_width=True)

    # ---- geometry / bias evolution ----
    st.markdown("#### Fit geometry and bias over time")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(t, R / s_num["sigma"], color="purple", lw=1.8, label=r"$R/\sigma$ (fit)")
    ax.set_xlabel(tlabel); ax.set_ylabel(r"$R/\sigma$", color="purple")
    ax.tick_params(axis="y", labelcolor="purple"); ax.grid(alpha=.25)
    ax2 = ax.twinx()
    ax2.plot(t, (s_num["T_app"]/s_num["T_peak"]-1)*100, color="crimson", lw=1.8,
             label="bias (%)")
    ax2.set_ylabel("apparent-$T$ bias (numerical, %)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    st.pyplot(fig, use_container_width=True)

    # ---- snapshot explorer ----
    st.markdown("#### Snapshot explorer")
    st.caption("Pick a time to inspect the radial profile + Gaussian fit (pinhole shaded), and "
               "the **numerically-collected** spectrum (black) with its **best-fit single "
               "Planck** (red). Green dotted = analytic Gaussian-surrogate spectrum.")
    j = st.slider("snapshot", 0, n_t-1, jpk, key="ts_snap",
                  help="index into the time series")
    st.write(f"**{tlabel} = {t[j]:g}**")
    res = _eval_snapshot(fbytes, fname, j, R, lo, hi, dt_us)
    t_app_tab = gauss_apparent(res, lo, hi)          # Gaussian-assumption apparent T (table)
    fig = plt.figure(figsize=(11, 4.3))
    plot_profile_eval(fig, res, lo, hi, t_app_gauss=t_app_tab)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    st.pyplot(fig, use_container_width=True)
    g = res.get("gauss") or {}
    sm = st.columns(5)
    sm[0].metric("actual peak", f"{res['T_peak']:.0f} K")
    sm[1].metric("apparent (numerical)", f"{res['T_app']:.0f} K")
    if t_app_tab is not None:
        sm[2].metric("apparent (Gaussian/table)", f"{t_app_tab:.0f} K",
                     f"{t_app_tab - res['T_app']:+.0f} K vs numerical")
    else:
        sm[2].metric("apparent (Gaussian/table)", "—")
    sm[3].metric("edge T(R)", f"{float(np.interp(R, r, T_cols[:, j])):.0f} K")
    sm[4].metric("R/σ, ξ", f"{g.get('R_over_sigma', float('nan')):.2f}, {res['xi']:.2f}")

    # ---- download ----
    out = np.column_stack([t, s_num["T_peak"], s_num["T_gauss"], s_num["T_app"],
                           s["T_app"], s_num["T_edge"], s_num["sigma"]*1e6])
    header = f"time,T_peak_K,T_gauss_K,T_app_numerical_K,T_app_{method}_K,T_edge_K,sigma_um"
    csv = header + "\n" + "\n".join(
        ",".join(f"{v:.6g}" for v in row) for row in out)
    dl0, dl1 = st.columns(2)
    dl0.download_button("⬇ download history (CSV)", csv, "temperature_history.csv",
                        "text/csv")
    if is_comsol:
        conv_hdr = ("# times = " + " ".join(f"{x:g}" for x in times) +
                    "\n# radius_um, T[K] per time (µs); converted from COMSOL line graph")
        prof = np.column_stack([r*1e6, T_cols])
        conv = conv_hdr + "\n" + "\n".join(
            ",".join(f"{v:.6g}" for v in row) for row in prof)
        dl1.download_button("⬇ download converted app CSV", conv,
                            "converted_series.csv", "text/csv",
                            help="the COMSOL profiles in the app's reusable CSV format")


# ============================================================ TAB 2: SINGLE PROFILE
with tab_prof:
    st.subheader("Single radial profile → forward evaluation")
    c0, c1 = st.columns([2, 1])
    with c0:
        pbytes, pname = data_source(os.path.join(HERE, "sample_Tprofile.csv"),
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
with tab_uni:
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
with tab_batch:
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
with tab_about:
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
with tab_spec:
    st.subheader("Measured emission spectrum → apparent T → radial $T(r)$")
    st.markdown(
        "Fit a measured spectrum over the spectrometer window to get the **apparent "
        "temperature**, then invert the pinhole bias under the Gaussian assumption to "
        "recover the **true peak temperature** and the radial profile.")

    u0, u1, u2, u3 = st.columns([2, 1, 1, 1])
    with u0:
        xbytes, xname = data_source(os.path.join(HERE, "sample_spectrum.csv"),
                                    "Upload spectrum (2 columns: wavelength, intensity)",
                                    key="sp_f")
    lam_unit = u1.selectbox("wavelength unit", ["nm", "µm", "m", "Å"], index=0, key="sp_u")
    lam_col = u2.number_input("wavelength column", 0, 20, 0, step=1, key="sp_lc")
    int_col = u3.number_input("intensity column", 0, 20, 1, step=1, key="sp_ic")
    if xbytes is None:
        st.stop()
    uscale = {"nm": 1e-9, "µm": 1e-6, "m": 1.0, "Å": 1e-10}[lam_unit]

    try:
        lam_all, I_all = _read_xy(xbytes, xname, int(lam_col), int(int_col), uscale)
        order = np.argsort(lam_all)
        lam_all, I_all = lam_all[order], I_all[order]
    except Exception as exc:
        st.error(f"could not read the spectrum: {exc}"); st.stop()

    m_fit = (lam_all >= lo) & (lam_all <= hi) & np.isfinite(I_all) & (I_all > 0)
    if m_fit.sum() < 5:
        st.error(f"only {int(m_fit.sum())} usable points inside the window "
                 f"{lo*1e9:.0f}–{hi*1e9:.0f} nm (data spans "
                 f"{lam_all.min()*1e9:.0f}–{lam_all.max()*1e9:.0f} nm). "
                 "Adjust the window in the sidebar or check the wavelength unit.")
        st.stop()

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
        st.error(f"fit/inversion failed: {exc}"); st.stop()

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
