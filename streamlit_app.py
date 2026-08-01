"""
Grey-body pyrometry of a Gaussian hot spot -- Streamlit app.

Tabs:
  * Time series (main)   -- evaluate a sequence of radial T(r) snapshots.
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

@st.cache_data(show_spinner=False)
def _eval_series(file_bytes, name, R, method, lo, hi):
    times, r, T = _load_series(file_bytes, name)
    return evaluate_profile_series(times, r, T, lo, hi, R=R, method=method)

@st.cache_data(show_spinner=False)
def _eval_snapshot(file_bytes, name, j, R, lo, hi):
    _, r, T = _load_series(file_bytes, name)
    return evaluate_profile(r, T[:, j], lo, hi, R=R)


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
hi_nm = st.sidebar.number_input("window high [nm]", 200.0, 5000.0, 800.0, step=25.0)
lo, hi = lo_nm * 1e-9, hi_nm * 1e-9
if lo >= hi:
    st.sidebar.error("window low must be < high")
lam_c = float(np.sqrt(lo * hi))
st.sidebar.caption(f"$\\lambda_c=\\sqrt{{\\lambda_1\\lambda_2}}$ = {lam_c*1e9:.0f} nm  ·  "
                   f"$c_2/\\lambda_c$ = {C2/lam_c:.0f} K (so $\\xi=${C2/lam_c:.0f}$/T_0$)")

tab_ts, tab_prof, tab_uni, tab_batch, tab_about = st.tabs(
    ["⏱ Time series", "📈 Single profile", "🎯 Gaussian & universal",
     "🌐 Universality (batch)", "📖 About / theory"])


# ============================================================ TAB 1: TIME SERIES
with tab_ts:
    st.subheader("Time series of radial temperature profiles")
    st.markdown(
        "Upload a **wide CSV**: column 0 = radius [µm], each further column = $T$ [K] at "
        "one time. Put the times in a comment line, e.g. `# times = 0 2.5 5 ...`.")

    c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
    with c0:
        fbytes, fname = data_source(os.path.join(HERE, "sample_Tseries_diffusion.csv"),
                                    "Upload time-series CSV", key="ts")
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
    R = R_um * 1e-6
    times, r, T_cols = _load_series(fbytes, fname)
    n_t = T_cols.shape[1]
    with st.spinner("evaluating series…"):
        s = _eval_series(fbytes, fname, R, method, lo, hi)

    # ---- headline metrics ----
    jpk = int(np.nanargmax(s["T_peak"]))
    m = st.columns(5)
    m[0].metric("snapshots", f"{n_t}")
    m[1].metric("max peak T", f"{np.nanmax(s['T_peak']):.0f} K")
    m[2].metric("apparent T at peak", f"{s['T_app'][jpk]:.0f} K",
                f"{s['T_app'][jpk]-s['T_peak'][jpk]:+.0f} K vs true")
    m[3].metric("max bias", f"{np.nanmin(s['T_app']/s['T_peak']-1)*100:+.1f} %")
    m[4].metric("edge T at peak", f"{s['T_edge'][jpk]:.0f} K")

    # ---- main history plot ----
    st.markdown("#### Temperature history through the pinhole")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    t = s["times"]
    ax.fill_between(t, s["T_app"], s["T_peak"], color="crimson", alpha=.08,
                    label="measurement bias")
    ax.plot(t, s["T_peak"], "-o", color="k", ms=3, lw=1.9, label="actual peak $T$ (data)")
    ax.plot(t, s["T_gauss"], "--", color="seagreen", lw=1.9, label="fitted Gaussian peak $T_0$")
    ax.plot(t, s["T_app"], "-", color="crimson", lw=1.9,
            label=f"apparent $T$ ({METHOD_LABEL[method]})")
    ax.plot(t, s["T_edge"], ":", color="steelblue", lw=2.2,
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
    ax.plot(t, R / s["sigma"], color="purple", lw=1.8, label=r"$R/\sigma$ (fit)")
    ax.set_xlabel(tlabel); ax.set_ylabel(r"$R/\sigma$", color="purple")
    ax.tick_params(axis="y", labelcolor="purple"); ax.grid(alpha=.25)
    ax2 = ax.twinx()
    ax2.plot(t, (s["T_app"]/s["T_peak"]-1)*100, color="crimson", lw=1.8, label="bias (%)")
    ax2.set_ylabel("apparent-$T$ bias (%)", color="crimson")
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
    res = _eval_snapshot(fbytes, fname, j, R, lo, hi)
    fig = plt.figure(figsize=(11, 4.3))
    plot_profile_eval(fig, res, lo, hi)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    st.pyplot(fig, use_container_width=True)
    g = res.get("gauss") or {}
    sm = st.columns(5)
    sm[0].metric("actual peak", f"{res['T_peak']:.0f} K")
    sm[1].metric("Gaussian peak", f"{g.get('T0', float('nan')):.0f} K")
    sm[2].metric("apparent T", f"{res['T_app']:.0f} K")
    sm[3].metric("edge T(R)", f"{float(np.interp(R, r, T_cols[:, j])):.0f} K")
    sm[4].metric("R/σ, ξ", f"{g.get('R_over_sigma', float('nan')):.2f}, {res['xi']:.2f}")

    # ---- download ----
    out = np.column_stack([t, s["T_peak"], s["T_gauss"], s["T_app"], s["T_edge"],
                           s["sigma"]*1e6])
    csv = "time,T_peak_K,T_gauss_K,T_app_K,T_edge_K,sigma_um\n" + "\n".join(
        ",".join(f"{v:.6g}" for v in row) for row in out)
    st.download_button("⬇ download history (CSV)", csv, "temperature_history.csv",
                       "text/csv")


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
        plot_profile_eval(fig, res, lo, hi)
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
