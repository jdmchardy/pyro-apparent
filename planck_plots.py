"""
Shared comparison plotting for the Gaussian-hot-spot pyrometry model.

Draws the 2x2 "universality" figure onto a caller-supplied matplotlib Figure, so the
batch script (Agg backend) and the GUI (TkAgg backend) render the SAME plotting spaces.
No backend is selected here -- the caller sets it before importing this module.
"""
import numpy as np
from matplotlib import colormaps

from planck_model import xi_window, wien_saturated_ratio, solve_bias, T_radial, planck


def plot_time_series(fig, s, time_label="time"):
    """Plot the temperature histories from evaluate_profile_series() on one axes:
    actual peak T, fitted Gaussian peak T, apparent T, and T at the pinhole edge."""
    ax = fig.subplots(1, 1)
    t = s["times"]
    mlabel = {"table": "Gaussian, lookup table", "universal": "Gaussian universal",
              "gaussian": "Gaussian model", "numerical": "numerical"}.get(
                  s.get("method", "numerical"), "")
    ax.plot(t, s["T_peak"], "-o", color="k", ms=4, lw=1.8, label="actual peak $T$ (data)")
    ax.plot(t, s["T_gauss"], "--", color="seagreen", lw=1.8,
            label="fitted Gaussian peak $T_0$")
    ax.plot(t, s["T_app"], "-", color="crimson", lw=1.8,
            label=f"apparent $T$ ({mlabel})")
    ax.plot(t, s["T_edge"], ":", color="steelblue", lw=2.0,
            label=f"$T$ at pinhole edge ($r=R={s['R']*1e6:.0f}$ µm)")
    ax.set_xlabel(time_label); ax.set_ylabel("temperature (K)")
    ax.set_title("Temperature history through the pinhole")
    ax.legend(fontsize=8.5); ax.grid(alpha=.25)


def plot_profile_eval(fig, res, fit_lo, fit_hi):
    """Two-panel view for a simulated T(r): the profile, and its collected spectrum
    with the single-Planck apparent-T fit overlaid. `res` is from evaluate_profile()."""
    ax0, ax1 = fig.subplots(1, 2)

    g = res.get("gauss")

    # left: the simulated radial temperature profile (+ Gaussian fit inside R)
    ax0.axvspan(0, res["R"]*1e6, color="skyblue", alpha=.20, zorder=0,
                label=f"pinhole fit region ($r\\leq R$={res['R']*1e6:.0f} µm)")
    ax0.axvline(res["R"]*1e6, color="steelblue", ls=":", lw=1)
    ax0.plot(res["r"]*1e6, res["T_r"], 'k-', lw=2, label="simulated $T(r)$")
    if g:
        ax0.plot(res["r"]*1e6, g["T_g_r"], color="seagreen", ls="--", lw=1.6,
                 label=(f"Gaussian fit ($r\\leq R$)\n$\\sigma$={g['sigma']*1e6:.1f} µm, "
                        f"$R/\\sigma$={g['R_over_sigma']:.2f}"))
    ax0.axhline(res["T_peak"], color="grey", ls=":", lw=1,
                label=f"peak {res['T_peak']:.0f} K")
    ax0.axhline(res["T_app"], color="crimson", ls="--", lw=1.4,
                label=f"apparent {res['T_app']:.0f} K")
    ax0.set_xlabel(r"$r$ ($\mu$m)"); ax0.set_ylabel(r"$T(r)$ (K)")
    ax0.set_title("Simulated radial temperature profile")
    ax0.legend(fontsize=7.5); ax0.grid(alpha=.25)

    # right: collected spectrum (linear) + apparent-T fit + Gaussian universal prediction
    F, lam = res["F_full"], res["lam_full"]
    norm = F.max()
    ax1.plot(lam*1e6, F/norm, 'k-', lw=2, label="collected (simulated)")
    ax1.plot(lam*1e6, planck(lam, res["T_app"], res["A_app"])/norm, 'r--', lw=1.6,
             label=f"Planck fit {res['T_app']:.0f} K")
    if g:
        ax1.plot(lam*1e6, g["F_full"]/norm, color="seagreen", ls=":", lw=1.8,
                 label=f"Gaussian universal ({g['T_app']:.0f} K)")
    ax1.axvspan(fit_lo*1e6, fit_hi*1e6, color="orange", alpha=.2, label="fit window")
    ax1.set_xlim(0, 3)
    ax1.set_xlabel(r"$\lambda$ ($\mu$m)"); ax1.set_ylabel(r"$F/F_{\max}$ (linear)")
    ax1.set_title(f"Collected spectrum  ($\\xi$={res['xi']:.2f}, "
                  f"bias {res['bias_frac']*100:+.1f}%)")
    ax1.legend(fontsize=7.5); ax1.grid(alpha=.25)

    fig.suptitle("Simulated distribution vs Gaussian universal prediction", fontsize=12)


def group_results(results):
    """Group solved configs by their 'group' key, R/sigma-sorted, first-seen order."""
    groups = {}
    for res in results:
        groups.setdefault(res["group"], []).append(res)
    for g in groups.values():
        g.sort(key=lambda r: r["R_over_sigma"])
    return groups


def plot_comparison(fig, results, fit_lo, fit_hi):
    """Render the 4-panel comparison onto `fig`. Returns the grouped results."""
    groups = group_results(results)
    fam_names = list(groups)
    tab = colormaps["tab10"]
    fam_colors = {name: tab(i % 10) for i, name in enumerate(fam_names)}

    ax = fig.subplots(2, 2)

    # (0,0) radial temperature distribution, absolute -- one curve per family
    for name in fam_names:
        col = fam_colors[name]
        widest = groups[name][-1]                     # largest R -> longest r array
        ax[0, 0].plot(widest["r"]*1e6, widest["T_r"], color=col, lw=1.8, label=name)
        for res in groups[name]:
            ax[0, 0].plot(res["R"]*1e6, T_radial(res["R"], res["T0"], res["sigma"]),
                          'o', color=col, ms=4, alpha=.7)
    ax[0, 0].set_xlabel(r"$r$ ($\mu$m)"); ax[0, 0].set_ylabel(r"$T(r)$ (K)")
    ax[0, 0].set_title("Radial temperature distribution\n(markers = collection cutoffs $R$)")
    ax[0, 0].legend(fontsize=7.5); ax[0, 0].grid(alpha=.25)

    # (0,1) normalised radial profile T/T0 vs r/sigma -- universal Gaussian form
    for name in fam_names:
        col = fam_colors[name]
        widest = groups[name][-1]
        ax[0, 1].plot(widest["r"]/widest["sigma"], widest["T_r"]/widest["T0"],
                      color=col, lw=1.6, alpha=.8)
        for res in groups[name]:
            ros = res["R_over_sigma"]
            ax[0, 1].plot(ros, np.exp(-ros**2/2), 'o', color=col, ms=5)
    ax[0, 1].set_xlabel(r"$r/\sigma$"); ax[0, 1].set_ylabel(r"$T(r)/T_0$")
    ax[0, 1].set_title("Normalised radial profile\n(all families collapse; markers = $R/\\sigma$)")
    ax[0, 1].grid(alpha=.25)

    # (1,0) MASTER CURVE: saturated (R>>sigma) bias collapses onto one curve vs xi
    ros_sat = max(res["R_over_sigma"] for res in results)
    T0_dense = np.linspace(2000., 12000., 60)
    xi_dense = xi_window(T0_dense, fit_lo, fit_hi)
    bias_dense = np.array([solve_bias(T0, ros_sat, fit_lo, fit_hi)*100 for T0 in T0_dense])
    order = np.argsort(xi_dense)
    ax[1, 0].plot(xi_dense[order], bias_dense[order], 'k-', lw=2,
                  label=f"numeric master curve ($R/\\sigma$={ros_sat:g})")
    xi_grid = np.linspace(xi_dense.min(), xi_dense.max(), 200)
    ax[1, 0].plot(xi_grid, (wien_saturated_ratio(xi_grid) - 1)*100, 'r--', lw=1.6,
                  label=r"analytic Wien  $\xi E_1(\xi)e^{\xi}$")
    for name in fam_names:
        sat = groups[name][-1]
        ax[1, 0].plot(sat["xi"], sat["bias_frac"]*100, 'o', color=fam_colors[name],
                      ms=8, mec='k', mew=.6, zorder=3)
    ax[1, 0].set_xlabel(r"window Wien parameter  $\xi = c_2/(\lambda_{\rm fit} T_0)$")
    ax[1, 0].set_ylabel(r"saturated bias  $(T_{\rm app}/T_0 - 1)$ (%)")
    ax[1, 0].set_title("Universal master curve (all families collapse vs $\\xi$)")
    ax[1, 0].legend(fontsize=7.5); ax[1, 0].grid(alpha=.25)

    # (1,1) the 2-parameter surface: bias vs R/sigma, one line per family
    for name in fam_names:
        col = fam_colors[name]
        xs = [res["R_over_sigma"] for res in groups[name]]
        ys = [res["bias_frac"]*100 for res in groups[name]]
        lbl = f"{name}  ($\\xi$={groups[name][0]['xi']:.2f})"
        ax[1, 1].plot(xs, ys, '-o', color=col, lw=1.6, ms=4, label=lbl)
    ax[1, 1].axhline(0, color="grey", lw=.7, ls=":")
    ax[1, 1].set_xlabel(r"$R/\sigma$")
    ax[1, 1].set_ylabel(r"apparent-$T$ bias  $(T_{\rm app}/T_0 - 1)$ (%)")
    ax[1, 1].set_title(r"Bias surface $\Phi(R/\sigma,\,\xi)$  ($\sigma$-degenerate curves overlap)")
    ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.25)

    fig.suptitle("Apparent-temperature bias across families "
                 f"-- fit window {fit_lo*1e9:.0f}-{fit_hi*1e9:.0f} nm", fontsize=12)
    return groups
