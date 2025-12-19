# -*- coding: utf-8 -*-
"""
Analysis of chunked mparams JSONL outputs without loading everything into memory.

Key revisions (per request)
---------------------------
1) Exactly TWO plots are produced:
   (A) Conductivity histograms for p in P_HIST_VALUES = [0.01, 0.1, 0.25, 0.5]
   (B) min/median/max conductivity vs p for p in [0.01, 0.5]

2) Mobility is ALWAYS computed in the low-carrier-density limit.
   - All prior notions of "matching experimental mobility" are removed.
   - No experimental mobility data, no STD windows, no classification.

3) Conductivity uses:
      N(p) = p / [ r_nn * pi * (d/2)^2 ]
      mu_eff(p) = mu0 * (1 - p)
      sigma(p) = e * mu_eff(p) * N(p)
               = e * mu0 * (p*(1-p)) / volume

   where mu0 is the corrected low-density mobility from the rate and hop distance.

Data format
-----------
Each JSONL line is:
    [[Rnn_nm, lambda_eV, coupling_eV], ["crate", "reqrate"], flag]
We use crate (1/s) and r_nn (nm).

Notes
-----
- This script stores a single "base factor" per filtered point:
      A = e * mu0 / volume
  so sigma(p) = A * p * (1 - p).
  This keeps memory use to one float per filtered point and allows fast reuse
  across many p values.
"""

import os
import glob
import json
import math
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate

# -----------------------------------------------------------------------------
# Plot / font settings
# -----------------------------------------------------------------------------
matplotlib.rcParams["figure.dpi"] = 600
font = {"family": "normal", "weight": "bold", "size": 11}
matplotlib.rc("font", **font)

# -----------------------------------------------------------------------------
# Top-level configuration
# -----------------------------------------------------------------------------
CHUNK_PATTERN = "mparams_chunk*.jsonl"

# Geometry for carrier density
CHANNEL_DIAMETER_NM = 2.0  # d in nm

# p values for histogram plot
P_HIST_VALUES = [0.01, 0.1, 0.25, 0.5]

# p grid for min/median/max plot
P_STATS_MIN = 0.01
P_STATS_MAX = 0.5
P_STATS_N = 100  # number of p points for the min/median/max curve
P_STATS_GRID = np.linspace(P_STATS_MIN, P_STATS_MAX, P_STATS_N)

# Filtering ranges (unchanged from your script)
RNN_MIN, RNN_MAX = 0.5, 2.5            # nm
LAMBDA_MIN, LAMBDA_MAX = 0.15, 0.361   # eV
COUP_MIN, COUP_MAX = 0.001, 0.03       # eV

# Physical constants
kB = 1.380649e-23        # J/K
e_charge = 1.602176e-19  # Coulomb
T = 300.0                # K


# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Conductivity analysis from chunked JSONL outputs."
    )
    parser.add_argument(
        "-t", "--jobtype",
        required=True,
        choices=["j", "1D", "3D"],
        help="Correction mode: j = junction correction, 1D = true 1D, 3D = effective 3D",
    )
    parser.add_argument(
        "-d", "--directory",
        required=True,
        help="Directory containing chunked JSONL files",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Correction factor setup (low-density mobility always)
# -----------------------------------------------------------------------------
def get_structural_factor(structure_type: str) -> float:
    if structure_type == "3D":
        return 1.5 * 2.61
    if structure_type == "j":
        return 1.5
    if structure_type == "1D":
        return 1.0
    raise ValueError(f"Unknown structure_type: {structure_type}")


def corrected_low_density_mobility_from_rate(k_rate_s, r_nm, chain_err_adj_base):
    """
    Compute corrected low-density mobility mu0 (cm^2 / V s) from rate k (1/s) and r (nm).

    Naive 1D chain:
        mu_chain = (e / (kB T)) * k * r^2

    Convert nm^2 -> cm^2 via 1 nm = 1e-7 cm, so nm^2 -> cm^2 is / 1e14.

    Low-density corrected mobility:
        mu0 = mu_chain / chain_err_adj_base

    where chain_err_adj_base = 2 * structural_factor
    (no occupancy dependence; low-density limit).
    """
    mu_chain_nm2 = (e_charge / (kB * T)) * k_rate_s * (r_nm ** 2)
    mu_chain_cm2 = mu_chain_nm2 / 1e14
    mu0 = mu_chain_cm2 / chain_err_adj_base
    return mu0


def volume_cm3_per_site(r_nn_nm, d_nm):
    """
    volume = r_nn * pi * (d/2)^2 with r_nn and d in cm.
    """
    r_cm = r_nn_nm * 1e-7
    d_cm = d_nm * 1e-7
    return r_cm * math.pi * (d_cm / 2.0) ** 2


# -----------------------------------------------------------------------------
# Pass to build base factors A = e * mu0 / volume for each filtered point
# -----------------------------------------------------------------------------
def build_base_factors(files, chain_err_adj_base, d_nm):
    """
    Returns:
      base_factors: list of A_i = e * mu0_i / volume_i  (units S/cm divided by p(1-p))
      stats dict: total_points, filtered_points, parameter min/max over filtered points
    """
    total_points = 0
    filtered_points = 0

    rnn_min, rnn_max = float("inf"), float("-inf")
    lam_min, lam_max = float("inf"), float("-inf")
    coup_min, coup_max = float("inf"), float("-inf")

    base_factors = []

    for fname in files:
        print(f"[Base-factor pass] Processing {fname} ...")
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                rec = json.loads(line)
                (rnn_nm, lambda_eV, coupling_eV), (crate_str, _reqrate_str), _flag = rec
                total_points += 1

                # Parameter cuts
                if not (RNN_MIN <= rnn_nm <= RNN_MAX):
                    continue
                if not (LAMBDA_MIN <= lambda_eV <= LAMBDA_MAX):
                    continue
                if not (COUP_MIN <= coupling_eV <= COUP_MAX):
                    continue

                filtered_points += 1

                rnn_min = min(rnn_min, rnn_nm)
                rnn_max = max(rnn_max, rnn_nm)
                lam_min = min(lam_min, lambda_eV)
                lam_max = max(lam_max, lambda_eV)
                coup_min = min(coup_min, coupling_eV)
                coup_max = max(coup_max, coupling_eV)

                crate = float(crate_str)
                if not (crate > 0.0) or math.isnan(crate):
                    continue

                mu0 = corrected_low_density_mobility_from_rate(crate, rnn_nm, chain_err_adj_base)
                if not (mu0 > 0.0) or math.isnan(mu0):
                    continue

                vol = volume_cm3_per_site(rnn_nm, d_nm)
                # A = e * mu0 / volume, so sigma(p) = A * p * (1-p)
                A = e_charge * mu0 / vol
                if not (A > 0.0) or math.isnan(A) or math.isinf(A):
                    continue

                base_factors.append(A)

    stats = {
        "total_points": total_points,
        "filtered_points": filtered_points,
        "param_min_max": {
            "Inter-cofactor Distance (nm)": (rnn_min, rnn_max),
            "Reorganization Energy (eV)": (lam_min, lam_max),
            "Coupling (eV)": (coup_min, coup_max),
        },
    }
    return base_factors, stats


# -----------------------------------------------------------------------------
# Plot 1: Conductivity histograms for multiple p values
# -----------------------------------------------------------------------------
def plot_conductivity_histograms(base_factors, p_values, outname):
    """
    Overlaid conductivity histograms for selected p values.
    sigma_i(p) = A_i * p*(1-p)
    """
    if not base_factors:
        print("No base factors computed; skipping histogram plot.")
        return

    base = np.asarray(base_factors, dtype=float)

    # Build a common range over all selected p values for consistent bins
    all_sigmas = []
    for p in p_values:
        all_sigmas.append(base * (p * (1.0 - p)))
    all_sigmas = np.concatenate(all_sigmas)

    smin = float(np.min(all_sigmas))
    smax = float(np.max(all_sigmas))
    if smin == smax:
        smin *= 0.9
        smax *= 1.1

    nbins = 120
    edges = np.linspace(smin, smax, nbins + 1)

    fig, ax = plt.subplots(figsize=(9, 6))

    for p in p_values:
        sig = base * (p * (1.0 - p))
        ax.hist(sig, bins=edges, alpha=0.5, label=f"p={p:g}",histtype='step')

    ax.set_xlabel(r"Conductivity $\sigma$ (S/cm)")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(r"Conductivity distributions (log y): $\sigma = e \mu_0(1-p)\,N(p)$")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(outname, dpi=300)
    print(f"Saved conductivity histogram figure to {outname}")


# -----------------------------------------------------------------------------
# Plot 2: Min/median/max conductivity vs p
# -----------------------------------------------------------------------------
def plot_min_median_max_vs_p(base_factors, p_grid, outname):
    """
    For each p:
      sigma(p) = A * p*(1-p)
    Because p*(1-p) is a scalar multiplier, min/median/max scale accordingly:
      min_sigma(p)    = min(A) * p*(1-p)
      median_sigma(p) = median(A) * p*(1-p)
      max_sigma(p)    = max(A) * p*(1-p)
    """
    if not base_factors:
        print("No base factors computed; skipping min/median/max plot.")
        return

    base = np.asarray(base_factors, dtype=float)
    A_min = float(np.min(base))
    A_med = float(np.median(base))
    A_max = float(np.max(base))

    p_grid = np.asarray(p_grid, dtype=float)
    factor = p_grid * (1.0 - p_grid)

    sig_min = A_min * factor
    sig_med = A_med * factor
    sig_max = A_max * factor

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(p_grid, sig_min, label="min σ")
    ax.plot(p_grid, sig_med, label="median σ")
    ax.plot(p_grid, sig_max, label="max σ")

    ax.set_xlabel("Occupancy p")
    ax.set_ylabel(r"Conductivity $\sigma$ (S/cm)")
    ax.set_yscale("log")
    ax.set_title("Conductivity envelope vs occupancy (log y)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(outname, dpi=300)
    print(f"Saved min/median/max vs p figure to {outname}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    structure_type = args.jobtype
    workingdir = args.directory.rstrip("/")
    workingdir=""

    #if not os.path.isdir(workingdir):
    #    raise FileNotFoundError(f"Directory not found: {workingdir}")

    #os.chdir(workingdir)

    files = sorted(glob.glob(CHUNK_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matched pattern in {workingdir}: {CHUNK_PATTERN}")

    print(f"FOUND MODE: {structure_type}")
    structural_factor = get_structural_factor(structure_type)

    # Low-density mobility correction factor (no occupancy dependence)
    chain_err_adj_base = 2.0 * structural_factor
    print("Assuming naive 1D chain mobility exceeds corrected low-density mobility by factor:",
          chain_err_adj_base)
    print("mu0 = mu_chain / chain_err_adj_base  (low-density limit)")

    print(f"Found {len(files)} chunk file(s):")
    for f in files:
        print("  ", f)

    base_factors, stats = build_base_factors(files, chain_err_adj_base, CHANNEL_DIAMETER_NM)

    print("\n" + "%" * 40)
    print(f"Total points read: {stats['total_points']}")
    print(f"Points after parameter filters: {stats['filtered_points']}")
    print(f"Points used for conductivity (finite mu0 and A): {len(base_factors)}")
    print("%" * 40 + "\n")

    # Parameter range table over filtered points
    pmm = stats["param_min_max"]
    table = {
        "Parameter": [],
        "Minimum": [],
        "Maximum": [],
    }
    for k, (vmin, vmax) in pmm.items():
        table["Parameter"].append(k)
        table["Minimum"].append("N/A" if vmin == float("inf") else round(vmin, 6))
        table["Maximum"].append("N/A" if vmax == float("-inf") else round(vmax, 6))

    print("Parameter ranges over ALL filtered points:")
    print(tabulate(table, headers="keys"))
    print()

    # Plot 1: histograms at selected p values
    plot_conductivity_histograms(
        base_factors=base_factors,
        p_values=P_HIST_VALUES,
        outname="conductivity_histograms_multi_p_logy.png",
    )

    # Plot 2: min/median/max vs p
    plot_min_median_max_vs_p(
        base_factors=base_factors,
        p_grid=P_STATS_GRID,
        outname="conductivity_min_median_max_vs_p_logy.png",
    )


if __name__ == "__main__":
    main()
