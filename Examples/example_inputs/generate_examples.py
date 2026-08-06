"""Generate example input files: a COMSOL-style radial export for 100 heating pulses
with inter-pulse diffusion (20 um FWHM source), and matching synthetic experimental
SOP spectra."""
import os, datetime
import numpy as np
from planck_model import spectrum_from_profile

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_files")

# ---------------------------------------------------------------- physics
FWHM   = 20e-6                                  # heat-source FWHM  [m]
SIG0   = FWHM / (2*np.sqrt(2*np.log(2)))        # -> sigma0 = 8.493 um
ALPHA  = 2e-5                                   # thermal diffusivity [m^2/s]
T_AMB  = 300.0
N_PULSE = 100
PULSE_DT = 1e-6                                 # one pulse every 1 us
T_END   = N_PULSE * PULSE_DT                    # 100 us
DT_SNAP = 0.1e-6                                # COMSOL sampling: 0.1 us
T_TARGET_PEAK = 6500.0                          # scale so the run tops out near this

r  = np.linspace(0, 80e-6, 60)                  # radial nodes [m]
t  = np.arange(0.0, T_END + 0.5*DT_SNAP, DT_SNAP)
tp = np.arange(N_PULSE) * PULSE_DT              # pulse arrival times

def profile(t_now):
    """Superpose every pulse fired at tk <= t_now, each diffused for tau = t_now - tk.
    Variance grows as sig0^2 + 2*alpha*tau; amplitude ~ sig0^2/variance (2-D energy)."""
    tk = tp[tp <= t_now + 1e-15]
    if tk.size == 0:
        return np.zeros_like(r)
    v = SIG0**2 + 2*ALPHA*(t_now - tk)                       # (n_pulse,)
    return ((SIG0**2 / v)[None, :] * np.exp(-r[:, None]**2 / (2*v[None, :]))).sum(axis=1)

U = np.column_stack([profile(x) for x in t])                 # unit per-pulse rise
dT = (T_TARGET_PEAK - T_AMB) / U.max()
T = T_AMB + dT * U

# ---------------------------------------------------------------- 1) COMSOL-style export
os.makedirs(OUT, exist_ok=True)
stamp = datetime.datetime.now().strftime("%b %-d %Y, %H:%M")
comsol = os.path.join(OUT, "COMSOL_radial_T_100pulses_20umFWHM.txt")
with open(comsol, "w") as f:
    f.write("% Model:              synthetic_pulsed_heating.mph\n")
    f.write("% Version:            COMSOL 6.0.0.318\n")
    f.write(f"% Date:               {stamp}\n")
    f.write("% Dimension:          1\n")
    f.write(f"% Nodes:              {r.size}\n")
    f.write(f"% Expressions:        {t.size}\n")
    f.write("% Description:        Line graph\n")
    f.write("% R" + " " * 23 + ("Temperature" + " " * 14) * t.size + "\n")
    for i in range(r.size):
        row = [f"{r[i]:<25.17g}"] + [f"{T[i, j]:<25.17g}" for j in range(t.size)]
        f.write("".join(row).rstrip() + "\n")
print(f"[1] {os.path.basename(comsol)}")
print(f"    {r.size} radial nodes (0-{r.max()*1e6:.0f} um), {t.size} times "
      f"(0-{t.max()*1e6:.0f} us at dt={DT_SNAP*1e6:g} us)")
print(f"    T {T.min():.0f}-{T.max():.0f} K   [{os.path.getsize(comsol)/1e6:.2f} MB]")

# ---------------------------------------------------------------- 2) experimental SOP
R_PIN   = 15e-6                     # collection pinhole radius
SOP_DT  = 0.5e-6                    # detector samples every 0.5 us
GATE    = 0.5e-6                    # integrated over the full 0.5 us (no dead time)
NOISE   = 0.02
lam = np.linspace(560e-9, 800e-9, 260)

t_sop = np.arange(SOP_DT/2, T_END, SOP_DT)
rng = np.random.default_rng(4242)
cols = []
for tc in t_sop:                                     # average the LIGHT over each gate
    sub = t[(t >= tc - GATE/2) & (t < tc + GATE/2)]
    if sub.size == 0:
        sub = np.array([tc])
    acc = np.zeros_like(lam)
    for ts_ in sub:
        acc += spectrum_from_profile(lam, r, T_AMB + dT*profile(ts_), R_PIN)
    acc /= sub.size
    cols.append(acc * (1 + NOISE*rng.standard_normal(lam.size)))
I = np.column_stack(cols)

sop = os.path.join(OUT, "experimental_SOP_spectra_100pulses.csv")
hdr = ("times = " + " ".join(f"{x*1e6:g}" for x in t_sop) +
       "\nSynthetic EXPERIMENTAL SOP spectra for COMSOL_radial_T_100pulses_20umFWHM.txt"
       f"\n  pinhole radius R = {R_PIN*1e6:g} um, gate = {GATE*1e6:g} us, "
       f"noise = {NOISE*100:g}%"
       f"\n  source: {FWHM*1e6:g} um FWHM, {N_PULSE} pulses at {PULSE_DT*1e6:g} us, "
       f"alpha = {ALPHA:g} m^2/s"
       "\n  columns: wavelength_nm, then intensity (arb.) per time (us)")
np.savetxt(sop, np.column_stack([lam*1e9, I]), delimiter=",", header=hdr,
           comments="# ", fmt="%.6g")
print(f"[2] {os.path.basename(sop)}")
print(f"    {lam.size} wavelengths ({lam.min()*1e9:.0f}-{lam.max()*1e9:.0f} nm), "
      f"{t_sop.size} times ({t_sop.min()*1e6:g}-{t_sop.max()*1e6:g} us)")
print(f"    pinhole R={R_PIN*1e6:g} um, gate {GATE*1e6:g} us, {NOISE*100:g}% noise "
      f"[{os.path.getsize(sop)/1e6:.2f} MB]")
