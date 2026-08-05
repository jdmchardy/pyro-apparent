"""
Convert a COMSOL 1-D radial 'Line graph' export into the app's time-series CSV.

The COMSOL file has radius rows (column 0 = R in metres) and one temperature column
per time. If the column headers carry the times ('@ t=...'), they are used; otherwise
pass --t-end (seconds) and the times are set to 0..T_end in equal steps.

Usage:
    python comsol_to_app.py INPUT.txt OUTPUT.csv [--t-end 3.7e-5] [--time-unit us]

Example:
    python comsol_to_app.py \\
        test_data_Nicolas/Radial_T_profile_UBS_oppside_130GPa_coarse_mesh_...txt \\
        my_series.csv --t-end 3.7e-5 --time-unit us
"""
import argparse
from planck_model import parse_comsol_line_graph, write_series_csv

_UNIT = {"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="COMSOL line-graph .txt export")
    ap.add_argument("output", help="output app-format CSV")
    ap.add_argument("--t-end", type=float, default=None,
                    help="T_end in seconds; times = 0..T_end in equal steps "
                         "(ignored if the file already carries times)")
    ap.add_argument("--time-unit", default="us", choices=list(_UNIT),
                    help="unit the times are written in (default us)")
    a = ap.parse_args()

    t, r, T = parse_comsol_line_graph(a.input, t_end=a.t_end)
    scale = _UNIT[a.time_unit]
    note = f"radius_um, T[K] per time ({a.time_unit}); converted from COMSOL line graph"
    write_series_csv(a.output, t * scale, r, T, note=note)
    print(f"wrote {a.output}: {r.size} radii (0..{r.max()*1e6:.1f} um), "
          f"{t.size} times ({t.min()*scale:g}..{t.max()*scale:g} {a.time_unit}), "
          f"Tmax {T.max():.0f} K")


if __name__ == "__main__":
    main()
