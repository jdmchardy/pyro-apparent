"""
Interactive (Plotly) figures for the pyrometry app.

Every plot in the app goes through these helpers so that panning, box-zoom, hover
readout and PNG export work consistently. Pure plotly + numpy: no Streamlit here.

  lines(...)      one axes with N traces (optionally a secondary y axis)
  heatmap(...)    a 2-D map (spectrogram, T(r,t))
  profile_eval(...)  the two-panel radial-profile / collected-spectrum view
  comparison(...)    the 2x2 universality figure
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from planck_model import xi_window, wien_saturated_ratio, solve_bias, T_radial, planck

# a colour-blind-safe qualitative set, used when a trace gives no explicit colour
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
           "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]

_AXIS = dict(showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False,
             showline=True, linecolor="rgba(0,0,0,0.35)", mirror=True, ticks="outside")


def _layout(fig, xlab, ylab, title=None, height=400, legend=True, y2lab=None):
    if title:
        fig.update_layout(title=dict(text=title, x=0.01, xanchor="left",
                                     font=dict(size=14)))
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=40 if title else 18, b=48),
        hovermode="x unified",
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.6)"),
        dragmode="pan",
    )
    fig.update_xaxes(title_text=xlab, **_AXIS)
    fig.update_yaxes(title_text=ylab, **_AXIS)
    if y2lab is not None:
        fig.update_yaxes(title_text=y2lab, secondary_y=True, showgrid=False,
                         zeroline=False, showline=True, ticks="outside")
    return fig


def lines(traces, xlab, ylab, title=None, height=400, legend=True,
          ylog=False, xrange=None, yrange=None, y2lab=None,
          vrects=(), hlines=(), vlines=()):
    """Build a line/scatter figure.

    `traces` is a list of dicts with keys: x, y, name, and optionally
    mode ('lines'|'markers'|'lines+markers'), color, dash, width, size,
    y2 (bool), error_y, fill, opacity, hovertemplate.
    """
    fig = make_subplots(specs=[[{"secondary_y": y2lab is not None}]])
    for i, t in enumerate(traces):
        col = t.get("color", PALETTE[i % len(PALETTE)])
        mode = t.get("mode", "lines")
        kw = dict(x=np.asarray(t["x"]), y=np.asarray(t["y"]),
                  name=t.get("name", f"trace {i+1}"), mode=mode,
                  opacity=t.get("opacity", 1.0),
                  showlegend=t.get("showlegend", True))
        if "lines" in mode:
            kw["line"] = dict(color=col, width=t.get("width", 2),
                              dash=t.get("dash", "solid"))
        if "markers" in mode:
            kw["marker"] = dict(color=col, size=t.get("size", 6),
                                line=dict(width=t.get("mlw", 0), color="black"))
        if t.get("error_y") is not None:
            kw["error_y"] = dict(type="data", array=np.asarray(t["error_y"]),
                                 visible=True, thickness=1, width=2, color=col)
        if t.get("fill"):
            kw["fill"] = t["fill"]
            kw["fillcolor"] = t.get("fillcolor", "rgba(214,39,40,0.10)")
            kw["line"] = dict(width=0)
        if t.get("hovertemplate"):
            kw["hovertemplate"] = t["hovertemplate"]
        fig.add_trace(go.Scatter(**kw), secondary_y=bool(t.get("y2", False)))

    for v in vrects:
        fig.add_vrect(x0=v[0], x1=v[1], fillcolor=v[2] if len(v) > 2 else "orange",
                      opacity=v[3] if len(v) > 3 else 0.15, line_width=0, layer="below",
                      annotation_text=v[4] if len(v) > 4 else None,
                      annotation_position="top left")
    for h in hlines:
        fig.add_hline(y=h[0], line=dict(color=h[1] if len(h) > 1 else "grey",
                                        dash=h[2] if len(h) > 2 else "dot", width=1),
                      annotation_text=h[3] if len(h) > 3 else None,
                      annotation_position="right")
    for v in vlines:
        fig.add_vline(x=v[0], line=dict(color=v[1] if len(v) > 1 else "grey",
                                        dash=v[2] if len(v) > 2 else "dot", width=1),
                      annotation_text=v[3] if len(v) > 3 else None,
                      annotation_position="top")

    _layout(fig, xlab, ylab, title, height, legend, y2lab)
    if ylog:
        fig.update_yaxes(type="log")
    if xrange is not None:
        fig.update_xaxes(range=list(xrange))
    if yrange is not None:
        fig.update_yaxes(range=list(yrange), secondary_y=False)
    return fig


def heatmap(x, y, z, xlab, ylab, zlab, title=None, height=400,
            cmap="Inferno", reverse_y=False, hlines=(), yrange=None):
    """2-D map (spectrogram, T(r,t) ...)."""
    fig = go.Figure(go.Heatmap(
        x=np.asarray(x), y=np.asarray(y), z=np.asarray(z), colorscale=cmap,
        colorbar=dict(title=dict(text=zlab, side="right"), thickness=14),
        hovertemplate=f"{xlab}: %{{x:.4g}}<br>{ylab}: %{{y:.4g}}<br>{zlab}: %{{z:.4g}}<extra></extra>"))
    for h in hlines:
        fig.add_hline(y=h[0], line=dict(color=h[1] if len(h) > 1 else "cyan",
                                        dash="dash", width=1.2),
                      annotation_text=h[2] if len(h) > 2 else None,
                      annotation_position="top right")
    _layout(fig, xlab, ylab, title, height, legend=False)
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    if yrange is not None:
        fig.update_yaxes(range=list(yrange))
    return fig


# ------------------------------------------------------------------ composite figures
def profile_eval(res, fit_lo, fit_hi, t_app_gauss=None, height=420):
    """Two panels: the radial profile (+ Gaussian fit, pinhole shaded) and the
    collected spectrum with its Planck fit. Mirrors planck_plots.plot_profile_eval."""
    g = res.get("gauss")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                        subplot_titles=("Radial temperature profile",
                                        f"Collected spectrum (ξ={res['xi']:.2f}, "
                                        f"bias {res['bias_frac']*100:+.1f}%)"))
    r_um = res["r"] * 1e6
    fig.add_vrect(x0=0, x1=res["R"]*1e6, fillcolor="skyblue", opacity=0.20,
                  line_width=0, layer="below", row=1, col=1)
    fig.add_trace(go.Scatter(x=r_um, y=res["T_r"], name="simulated T(r)",
                             line=dict(color="black", width=2.5)), row=1, col=1)
    if g:
        fig.add_trace(go.Scatter(
            x=r_um, y=g["T_g_r"],
            name=f"Gaussian fit (σ={g['sigma']*1e6:.1f} µm, R/σ={g['R_over_sigma']:.2f})",
            line=dict(color="seagreen", width=1.8, dash="dash")), row=1, col=1)
    fig.add_hline(y=res["T_peak"], line=dict(color="grey", dash="dot", width=1),
                  annotation_text=f"peak {res['T_peak']:.0f} K", row=1, col=1)
    fig.add_hline(y=res["T_app"], line=dict(color="crimson", dash="dash", width=1.4),
                  annotation_text=f"apparent (numerical) {res['T_app']:.0f} K",
                  row=1, col=1)
    if t_app_gauss is not None:
        fig.add_hline(y=t_app_gauss,
                      line=dict(color="darkorange", dash="dashdot", width=1.4),
                      annotation_text=f"apparent (Gaussian/table) {t_app_gauss:.0f} K",
                      row=1, col=1)

    lam_um = res["lam_full"] * 1e6
    norm = np.nanmax(res["F_full"]) or 1.0
    fig.add_vrect(x0=fit_lo*1e6, x1=fit_hi*1e6, fillcolor="orange", opacity=0.18,
                  line_width=0, layer="below", row=1, col=2)
    fig.add_trace(go.Scatter(x=lam_um, y=res["F_full"]/norm,
                             name="collected (numerical, from data T(r))",
                             line=dict(color="black", width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=lam_um, y=planck(res["lam_full"], res["T_app"], res["A_app"])/norm,
        name=f"best-fit single Planck ({res['T_app']:.0f} K)",
        line=dict(color="crimson", width=1.8, dash="dash")), row=1, col=2)
    if g:
        fig.add_trace(go.Scatter(x=lam_um, y=g["F_full"]/norm,
                                 name="Gaussian-surrogate (analytic)",
                                 line=dict(color="seagreen", width=1.8, dash="dot")),
                      row=1, col=2)

    fig.update_xaxes(title_text="r (µm)", row=1, col=1, **_AXIS)
    fig.update_yaxes(title_text="T(r) (K)", row=1, col=1, **_AXIS)
    fig.update_xaxes(title_text="λ (µm)", range=[0, 3], row=1, col=2, **_AXIS)
    fig.update_yaxes(title_text="F / F_max", row=1, col=2, **_AXIS)
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=54, b=48),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode="pan",
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  font=dict(size=9)))
    return fig


def comparison(results, fit_lo, fit_hi, height=760):
    """2x2 universality figure (radial profiles, normalised profiles, master curve,
    bias surface). Mirrors planck_plots.plot_comparison."""
    groups = {}
    for res in results:
        groups.setdefault(res["group"], []).append(res)
    for gg in groups.values():
        gg.sort(key=lambda r: r["R_over_sigma"])
    names = list(groups)
    col_of = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.13, horizontal_spacing=0.09,
        subplot_titles=("Radial temperature distribution",
                        "Normalised profile (all families collapse)",
                        "Universal master curve vs ξ",
                        "Bias surface Φ(R/σ, ξ)"))

    for n in names:
        c = col_of[n]
        w = groups[n][-1]
        fig.add_trace(go.Scatter(x=w["r"]*1e6, y=w["T_r"], name=n, legendgroup=n,
                                 line=dict(color=c, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[r["R"]*1e6 for r in groups[n]],
            y=[T_radial(r["R"], r["T0"], r["sigma"]) for r in groups[n]],
            mode="markers", marker=dict(color=c, size=6), legendgroup=n,
            showlegend=False, name=n), row=1, col=1)
        fig.add_trace(go.Scatter(x=w["r"]/w["sigma"], y=w["T_r"]/w["T0"],
                                 line=dict(color=c, width=2), legendgroup=n,
                                 showlegend=False, name=n), row=1, col=2)
        ros = np.array([r["R_over_sigma"] for r in groups[n]])
        fig.add_trace(go.Scatter(x=ros, y=np.exp(-ros**2/2), mode="markers",
                                 marker=dict(color=c, size=6), legendgroup=n,
                                 showlegend=False, name=n), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=ros, y=[(r["bias_frac"])*100 for r in groups[n]],
            mode="lines+markers", line=dict(color=c, width=2),
            marker=dict(size=5), legendgroup=n, showlegend=False,
            name=f"{n} (ξ={groups[n][0]['xi']:.2f})"), row=2, col=2)

    ros_sat = max(r["R_over_sigma"] for r in results)
    T0d = np.linspace(2000., 12000., 60)
    xid = np.asarray(xi_window(T0d, fit_lo, fit_hi), float)
    bd = np.array([solve_bias(T, ros_sat, fit_lo, fit_hi)*100 for T in T0d])
    o = np.argsort(xid)
    fig.add_trace(go.Scatter(x=xid[o], y=bd[o], name=f"numeric (R/σ={ros_sat:g})",
                             line=dict(color="black", width=2.5)), row=2, col=1)
    xg = np.linspace(xid.min(), xid.max(), 200)
    fig.add_trace(go.Scatter(x=xg, y=(wien_saturated_ratio(xg)-1)*100,
                             name="analytic ξE₁(ξ)e^ξ",
                             line=dict(color="crimson", width=1.8, dash="dash")),
                  row=2, col=1)
    for n in names:
        sat = groups[n][-1]
        fig.add_trace(go.Scatter(x=[sat["xi"]], y=[sat["bias_frac"]*100],
                                 mode="markers", legendgroup=n, showlegend=False,
                                 marker=dict(color=col_of[n], size=10,
                                             line=dict(width=1, color="black")),
                                 name=n), row=2, col=1)

    for rr, cc, xl, yl in ((1, 1, "r (µm)", "T(r) (K)"),
                           (1, 2, "r/σ", "T(r)/T₀"),
                           (2, 1, "ξ = c₂/(λ_c T₀)", "saturated bias (%)"),
                           (2, 2, "R/σ", "apparent-T bias (%)")):
        fig.update_xaxes(title_text=xl, row=rr, col=cc, **_AXIS)
        fig.update_yaxes(title_text=yl, row=rr, col=cc, **_AXIS)
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=60, b=60),
                      plot_bgcolor="white", paper_bgcolor="white", dragmode="pan",
                      legend=dict(orientation="h", yanchor="top", y=-0.08,
                                  font=dict(size=9)))
    return fig
