# -*- coding: utf-8 -*-
"""
Analysis of chunked mparams JSONL outputs without loading everything into memory.

- Expects files like: mparams_chunk0.jsonl, mparams_chunk1.jsonl, ...
- Each line is a JSON record of the form:
    [[Rnn_nm, lambda_eV, coupling_eV], ["crate", "reqrate"], flag]

We do:
  1) First pass: count how many "good" points there are and compute stats,
     but do not store the points.
  2) Histogram pass: classify all filtered points as "matching" or
     "non-matching" relative to experiment and build histograms.

Mobility conventions
--------------------
- "Experimental mobility" μ_exp: comes from experiment (e.g. 0.09–0.27 cm^2/Vs).
- "Chain mobility" μ_chain: naive 1D chain result from the rate and hop distance.
- "Corrected mobility" μ_corr: the mobility we compare to experiment, after
  applying a geometry/network correction factor.

We define a correction factor `chain_err_adj` such that:
    μ_chain = chain_err_adj * μ_corr
or equivalently:
    μ_corr = μ_chain / chain_err_adj

Usage:
- When going from rate → mobility (forward direction), we:
    1. Compute μ_chain from the rate.
    2. Divide by chain_err_adj to get μ_corr.

This way, the correction factor is applied consistently and exactly once.
"""

import json
import math
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import statistics
from tabulate import tabulate
import argparse
import sys

# -----------------------------------------------------------------------------
# Global plotting toggles
# -----------------------------------------------------------------------------
# If True, use a logarithmic y-axis in the main mobility histogram
MAIN_YLOG = True
# If True, use more (narrower) bins in the main mobility histogram.
# If False, keep the current behavior (50 bins).
MAIN_FINE_BINS = True

SHOW_MIN_MAX_MATCH_ONLY=True

# Factor by which to multiply the default number of bins when MAIN_FINE_BINS is True
MAIN_FINE_BINS_FACTOR = 10

# -----------------------------------------------------------------------------
# Plot / font settings
# -----------------------------------------------------------------------------
matplotlib.rcParams['figure.dpi'] = 600
font = {'family': 'normal', 'weight': 'bold', 'size': 11}
matplotlib.rc('font', **font)

# -----------------------------------------------------------------------------
# Input configuration
# -----------------------------------------------------------------------------
# Pattern for chunked JSONL files produced by the new parallel code
CHUNK_PATTERN = "mparams_chunk*.jsonl"

# -----------------------------------------------------------------------------
# Experimental data and constants
# -----------------------------------------------------------------------------
expdata = [0.09, 0.11, 0.27, 0.27]  # experimental mobilities in cm^2/Vs
mu_stdev = statistics.stdev(expdata)
mu_exp_mean = sum(expdata) / len(expdata)

# Target experimental/corrected mobility (cm^2/Vs) used in the script
# (can be equal to or slightly adjusted from mu_exp_mean)
mu_exp_target = mu_exp_mean

# How many experimental standard deviations to use:
# - STD_SCREEN: for reporting "good" points in stats (unused in plotting now)
# - STD_RAINBOW: re-used here as the 0.5σ "matching" window for histograms
STD_SCREEN = 1.0   # ±1σ window for "good" points (stats only)
STD_RAINBOW = 1  # ±0.5σ band for "matching" in histograms
p=0.5
occ_factor=1-p
#low carrier density limit
occ_factor=1

mu_screen_halfwidth = mu_stdev * STD_SCREEN
mu_rainbow_halfwidth = mu_stdev * STD_RAINBOW

mu_low = mu_exp_target - mu_rainbow_halfwidth
mu_high = mu_exp_target + mu_rainbow_halfwidth

# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse plotting mode and working directory."
    )

    parser.add_argument(
        "-t", "--jobtype",
        required=True,
        choices=["j", "1D", "3D"],
        help="Plot mode: j = default junction correction, 1D = true 1D, 3D = effective 3D"
    )

    parser.add_argument(
        "-d", "--directory",
        required=True,
        help="Directory containing data for plotting"
    )

    args = parser.parse_args()

    return args.jobtype, args.directory


# Correction factor:
#   chain_err_adj = factor by which the naive 1D chain mobility (μ_chain)
#   overestimates the corrected mobility (μ_corr) that we compare to experiment.
#
#   μ_chain = chain_err_adj * μ_corr
#   μ_corr  = μ_chain / chain_err_adj
#
# Breakdown:
# - extra factor of 2 for the missing 1/2 in the 1D chain equation in this code
# - 3/2 from junctions
# - 2.61 from additional network / geometry corrections

structural_factor = 1
structure_type, workingdir = parse_args()
if workingdir.endswith("/"):
    workingdir = workingdir[:-1]

if structure_type == "3D":
    structural_factor = 1.5 * 2.61
elif structure_type == "j":
    structural_factor = 1.5
elif structure_type == "1D":
    structural_factor = 1

chain_err_adj = 2 * structural_factor/occ_factor
print("FOUND MODE ", structure_type)
print("Assuming naive 1D chain mobility exceeds corrected mobility by factor:",
      chain_err_adj)
print("μ_corr = μ_chain / chain_err_adj")

# Physical constants
kB = 1.380649e-23      # J/K
fund_charge = 1.602176e-19
T = 300.0

# Filtering ranges (same as in original script)
RNN_MIN, RNN_MAX = 0.5, 2.5           # nm
LAMBDA_MIN, LAMBDA_MAX = 0.15, 0.361  # eV
COUP_MIN, COUP_MAX = 0.001, 0.03      # eV


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _corrected_mobility_from_rate(k, r_nm):
    """
    Corrected mobility μ_corr in cm^2/(V·s) from rate k (1/s) and hop distance r (nm).

    Step 1: Compute the naive 1D chain mobility μ_chain:
        μ_chain = (e / (k_B T)) * k * r^2
      where r is in meters. Here r is given in nm, so we convert by hand.

    Step 2: Convert from m^2/(V·s) to cm^2/(V·s).

    Step 3: Apply the geometry/network correction factor:
        μ_corr = μ_chain / chain_err_adj

    The returned value μ_corr is the mobility that should be compared to
    experimental mobilities.
    """
    r = r_nm
    # 1D chain mobility (in nm^2 V^-1 s^-1 units)
    mu_chain = fund_charge / (kB * T) * k * (r ** 2)
    # Convert nm^2 -> cm^2: (1 nm = 1e-7 cm) → 1 nm^2 = 1e-14 cm^2
    mu_chain = mu_chain / 1e14
    # Apply correction factor ONCE to get corrected mobility
    mu_corr = mu_chain / chain_err_adj
    return mu_corr


# -----------------------------------------------------------------------------
# First pass: count-only and stats
# -----------------------------------------------------------------------------
def first_pass(files):
    """
    First streaming pass over all files.

    Computes:
    - total_points
    - filtered_points (after parameter cuts)
    - min/max for each parameter
    - count of points within 0.01 of mu_exp_target (corrected mobility)
    - count of points within STD_SCREEN * σ_exp of mu_exp_target
      (corrected mobility)
    """
    total_points = 0
    filtered_points = 0
    good_within_0p01 = 0
    good_within_screen = 0

    # Global min/max for parameters
    rnn_min, rnn_max = float("inf"), float("-inf")
    lam_min, lam_max = float("inf"), float("-inf")
    coup_min, coup_max = float("inf"), float("-inf")
    coupdist_min, coupdist_max = float("inf"), float("-inf")

    for fname in files:
        print(f"[First pass] Processing {fname} ...")
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                (rnn_nm, lambda_eV, coupling_eV), (crate_str, reqrate_str), flag = rec

                total_points += 1

                # Parameter cuts
                if not (RNN_MIN <= rnn_nm <= RNN_MAX):
                    continue
                if not (LAMBDA_MIN <= lambda_eV <= LAMBDA_MAX):
                    continue
                if not (COUP_MIN <= coupling_eV <= COUP_MAX):
                    continue

                filtered_points += 1

                # Min/max
                if(not SHOW_MIN_MAX_MATCH_ONLY):
                    rnn_min = min(rnn_min, rnn_nm)
                    rnn_max = max(rnn_max, rnn_nm)
                    lam_min = min(lam_min, lambda_eV)
                    lam_max = max(lam_max, lambda_eV)
                    coup_min = min(coup_min, coupling_eV)
                    coup_max = max(coup_max, coupling_eV)
                    coupdist_min=min(coupdist_min,coupling_eV*rnn_nm)
                    coupdist_max=max(coupdist_max,coupling_eV*rnn_nm)

                crate = float(crate_str)
                mu_corr = _corrected_mobility_from_rate(crate, rnn_nm)

                if abs(mu_corr - mu_exp_target) <= 0.01:
                    good_within_0p01 += 1
                if abs(mu_corr - mu_exp_target) <= mu_screen_halfwidth:
                    good_within_screen += 1
                    if(SHOW_MIN_MAX_MATCH_ONLY):
                        rnn_min = min(rnn_min, rnn_nm)
                        rnn_max = max(rnn_max, rnn_nm)
                        lam_min = min(lam_min, lambda_eV)
                        lam_max = max(lam_max, lambda_eV)
                        coup_min = min(coup_min, coupling_eV)
                        coup_max = max(coup_max, coupling_eV)
                        coupdist_min=min(coupdist_min,coupling_eV*rnn_nm)
                        coupdist_max=max(coupdist_max,coupling_eV*rnn_nm)

    param_min_max = {
        "Inter-cofactor Distance (nm)": (rnn_min, rnn_max),
        "Reorganization Energy (eV)": (lam_min, lam_max),
        "Coupling (eV)": (coup_min, coup_max),
        "coupling X distance (nm eV)":(coupdist_min, coupdist_max)
    }

    stats = {
        "total_points": total_points,
        "filtered_points": filtered_points,
        "param_min_max": param_min_max,
        "good_counts": {
            "within_0p01": good_within_0p01,
            "within_screen": good_within_screen,
        },
    }

    return stats


# -----------------------------------------------------------------------------
# Histogram pass & figure
# -----------------------------------------------------------------------------
def histogram_pass(files):
    """
    Histogram pass.

    Classifies points into:
      - matching: μ_corr within ±0.5σ (STD_RAINBOW) of μ_exp_target
      - non-matching: all other points that pass parameter cuts

    Returns a dict of arrays for mobility and Marcus parameters
    (Rnn, lambda, coupling) for each class + all μ_corr.
    """
    all_mu = []

    matching_mu = []
    nonmatching_mu = []

    match_rnn = []
    match_lambda = []
    match_coup = []  # coupling == H_DA

    nonmatch_rnn = []
    nonmatch_lambda = []
    nonmatch_coup = []

    for fname in files:
        print(f"[Histogram pass] Processing {fname} ...")
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                (rnn_nm, lambda_eV, coupling_eV), (crate_str, reqrate_str), flag = rec

                # Same parameter cuts
                if not (RNN_MIN <= rnn_nm <= RNN_MAX):
                    continue
                if not (LAMBDA_MIN <= lambda_eV <= LAMBDA_MAX):
                    continue
                if not (COUP_MIN <= coupling_eV <= COUP_MAX):
                    continue

                crate = float(crate_str)
                mu_corr = _corrected_mobility_from_rate(crate, rnn_nm)

                if not (mu_corr > 0.0) or math.isnan(mu_corr):
                    continue

                # All filtered μ_corr
                all_mu.append(mu_corr)

                if mu_low <= mu_corr <= mu_high:
                    # Matching (within ±0.5σ of experimental/mu_exp_target)
                    matching_mu.append(mu_corr)
                    match_rnn.append(rnn_nm)
                    match_lambda.append(lambda_eV)
                    match_coup.append(coupling_eV)
                else:
                    # Non-matching but still physically filtered
                    nonmatching_mu.append(mu_corr)
                    nonmatch_rnn.append(rnn_nm)
                    nonmatch_lambda.append(lambda_eV)
                    nonmatch_coup.append(coupling_eV)

    hist_data = {
        "all_mu": all_mu,
        "matching_mu": matching_mu,
        "nonmatching_mu": nonmatching_mu,
        "match_rnn": match_rnn,
        "match_lambda": match_lambda,
        "match_coup": match_coup,
        "nonmatch_rnn": nonmatch_rnn,
        "nonmatch_lambda": nonmatch_lambda,
        "nonmatch_coup": nonmatch_coup,
    }
    return hist_data


def make_histogram_figure(hist_data):
    """
    Build a 5-panel histogram figure:

    - Top row (1 wide panel):
        Histogram of ALL μ_corr values, with:
          * shaded ±0.5σ matching window
          * vertical line at μ_exp_target
          * overlaid histogram of matching μ_corr
          * single legend (key) here only.
    - Bottom row (4 panels):
        For each parameter (Rnn, λ, H_DA, R*H_DA), overlay matching vs
        non-matching distributions using the same bin edges.
    """
    all_mu = hist_data["all_mu"]
    matching_mu = hist_data["matching_mu"]
    nonmatching_mu = hist_data["nonmatching_mu"]

    if len(all_mu) == 0:
        print("No μ_corr points found after filtering; skipping histogram figure.")
        return

    if len(matching_mu) == 0:
        print("No matching μ_corr points found (within ±0.5σ); skipping histogram figure.")
        return

    if len(nonmatching_mu) == 0:
        print("No non-matching μ_corr points found; skipping histogram figure.")
        return

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 1.0])

    # ------------------------------------------------------------------
    # Main panel: histogram of ALL μ_corr + overlay of matching μ_corr
    # ------------------------------------------------------------------
    ax_main = fig.add_subplot(gs[0, :])

    # Shared bins for all μ and matching μ
    base_nbins_mu = 50  # original behavior
    nbins_mu = base_nbins_mu * MAIN_FINE_BINS_FACTOR if MAIN_FINE_BINS else base_nbins_mu

    mu_min = min(all_mu)
    mu_max = max(all_mu)
    mu_edges = np.linspace(mu_min, mu_max, nbins_mu + 1)

    # Shaded matching window (behind histograms)
    ax_main.axvspan(mu_low, mu_high, alpha=0.15,
                    label=fr"matching window (±{STD_RAINBOW}$\sigma$)")

    # Histogram of all points
    ax_main.hist(all_mu, bins=mu_edges, alpha=0.6, label="all points")

    # Histogram of matching points
    ax_main.hist(matching_mu, bins=mu_edges, alpha=0.7, label="matching (±1σ)")

    # Vertical line at experimental target
    ax_main.axvline(mu_exp_target, linestyle="--", linewidth=1.5,
                    label=r"$\mu_{\mathrm{exp,target}}$")

    if MAIN_YLOG:
        ax_main.set_yscale("log")

    ax_main.set_xlabel(r"Mobility $\mu$ (cm$^2$/Vs)")
    ax_main.set_ylabel("Count")
    ax_main.set_title(
        "Mobility distribution\n"
        f"(matching window ±{STD_RAINBOW}σ)"
    )
    ax_main.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Overlayed parameter histograms: Rnn, λ, H_DA, R*H_DA
    # ------------------------------------------------------------------
    nbins = 40

    # Rnn
    ax_rnn = fig.add_subplot(gs[1, 0])
    rnn_all = hist_data["match_rnn"] + hist_data["nonmatch_rnn"]
    rnn_min = min(rnn_all)
    rnn_max = max(rnn_all)
    rnn_edges = np.linspace(rnn_min, rnn_max, nbins + 1)

    ax_rnn.hist(hist_data["nonmatch_rnn"], bins=rnn_edges, alpha=0.6)
    ax_rnn.hist(hist_data["match_rnn"], bins=rnn_edges, alpha=0.7)
    ax_rnn.set_title("Rnn distribution")
    ax_rnn.set_xlabel("Rnn (nm)")
    ax_rnn.set_ylabel("Count")

    # lambda
    ax_lambda = fig.add_subplot(gs[1, 1])
    lam_all = hist_data["match_lambda"] + hist_data["nonmatch_lambda"]
    lam_min = min(lam_all)
    lam_max = max(lam_all)
    lam_edges = np.linspace(lam_min, lam_max, nbins + 1)

    ax_lambda.hist(hist_data["nonmatch_lambda"], bins=lam_edges, alpha=0.6)
    ax_lambda.hist(hist_data["match_lambda"], bins=lam_edges, alpha=0.7)
    ax_lambda.set_title(r"$\lambda$ distribution")
    ax_lambda.set_xlabel(r"$\lambda$ (eV)")

    # H_DA (coupling)
    ax_coup = fig.add_subplot(gs[1, 2])
    coup_all = hist_data["match_coup"] + hist_data["nonmatch_coup"]
    coup_min = min(coup_all)
    coup_max = max(coup_all)
    coup_edges = np.linspace(coup_min, coup_max, nbins + 1)

    ax_coup.hist(hist_data["nonmatch_coup"], bins=coup_edges, alpha=0.6)
    ax_coup.hist(hist_data["match_coup"], bins=coup_edges, alpha=0.7)
    ax_coup.set_title(r"$H_{\mathrm{DA}}$ distribution")
    ax_coup.set_xlabel(r"$H_{\mathrm{DA}}$ (eV)")

    # R * H_DA combined parameter
    ax_rH = fig.add_subplot(gs[1, 3])
    match_rH = [r * h for r, h in zip(hist_data["match_rnn"], hist_data["match_coup"])]
    nonmatch_rH = [r * h for r, h in zip(hist_data["nonmatch_rnn"], hist_data["nonmatch_coup"])]
    rH_all = match_rH + nonmatch_rH
    rH_min = min(rH_all)
    rH_max = max(rH_all)
    rH_edges = np.linspace(rH_min, rH_max, nbins + 1)

    ax_rH.hist(nonmatch_rH, bins=rH_edges, alpha=0.6)
    ax_rH.hist(match_rH, bins=rH_edges, alpha=0.7)
    ax_rH.set_title(r"$R \times H_{\mathrm{DA}}$ distribution")
    ax_rH.set_xlabel(r"$R \times H_{\mathrm{DA}}$ (nm·eV)")

    fig.tight_layout()
    outname = "mobility_param_histograms.png"
    fig.savefig(outname, dpi=300)
    print(f"Saved histogram figure to {outname}")


# -----------------------------------------------------------------------------
# Main analysis & plotting
# -----------------------------------------------------------------------------
def main():
    files = sorted(glob.glob(CHUNK_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {CHUNK_PATTERN}")

    print(f"Found {len(files)} chunk file(s):")
    for f in files:
        print("  ", f)

    # ---------------- First pass: counts, stats, no storage ----------------
    stats = first_pass(files)
    total = stats["total_points"]
    filtered = stats["filtered_points"]
    good_counts = stats["good_counts"]
    param_min_max = stats["param_min_max"]

    good_within_screen = good_counts["within_screen"]

    print("%" * 20)
    print(f"Total points read: {total}")
    print(f"Points after parameter filters: {filtered}")
    print("%" * 20)

    # Fractions of good points (relative to filtered set)
    if filtered > 0:
        frac_0p01 = good_counts["within_0p01"] / filtered
        frac_screen = good_within_screen / filtered
    else:
        frac_0p01 = frac_screen = 0.0

    print("Fraction within 0.01 of μ_exp_target (corrected):",
          frac_0p01, good_counts["within_0p01"], filtered)
    print(f"Fraction within ±{STD_SCREEN}σ_exp of μ_exp_target (corrected):",
          frac_screen, good_within_screen, filtered)

    # Parameter summary table
    
    params = {
        "Inter-cofactor Distance": [param_min_max["Inter-cofactor Distance (nm)"][0],
                                    param_min_max["Inter-cofactor Distance (nm)"][1]],
        "Reorganization Energy":   [param_min_max["Reorganization Energy (eV)"][0],
                                    param_min_max["Reorganization Energy (eV)"][1]],
        "Coupling":                [param_min_max["Coupling (eV)"][0],
                                    param_min_max["Coupling (eV)"][1]],
        "Coupling X Distance":     [param_min_max["coupling X distance (nm eV)"][0],
                                    param_min_max["coupling X distance (nm eV)"][1]]
    }
    table = {"Parameter": [], "Minimum": [], "Maximum": []}
    units = {"Inter-cofactor Distance": "(nm)",
             "Reorganization Energy": "(eV)",
             "Coupling": "(eV)",
             "Coupling X Distance":"(nm eV)"
             }

    for p in params:
        table["Parameter"].append(p + " " + units[p])
        table["Minimum"].append(round(params[p][0], 5))
        table["Maximum"].append(round(params[p][1], 5))

    print(tabulate(table, headers="keys"))

    # ---------------------------------------------------------------------
    # Histogram figure: μ_corr (all + matching) + parameter overlays
    # ---------------------------------------------------------------------
    hist_data = histogram_pass(files)
    make_histogram_figure(hist_data)


if __name__ == "__main__":
    main()
