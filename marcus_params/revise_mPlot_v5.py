# -*- coding: utf-8 -*-
"""
Analysis of chunked mparams JSONL outputs without loading everything into memory.

- Expects files like: mparams_chunk0.jsonl, mparams_chunk1.jsonl, ...
- Each line is a JSON record of the form:
    [[Rnn_nm, lambda_eV, coupling_eV], ["crate", "reqrate"], flag]

We do two streaming passes:
  1) First pass: count how many "good" points there are and compute stats,
     but do not store the points.
  2) Second pass: keep every n-th good point so that the total stored
     points is <= MAX_GOOD_POINTS_FOR_SCATTER.

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
- When inverting to get R*H given a target *corrected* mobility (the one that
  should match experiment), we:
    1. Multiply the target μ_corr by chain_err_adj to reconstruct μ_chain.
    2. Use the original inversion algebra (which is derived for μ_chain).

This way, the correction factor is applied consistently and exactly once in
each direction, and there is no double counting.
"""

import json
import math
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import statistics

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

# Max number of "good" points we keep for scatter plotting
MAX_GOOD_POINTS_FOR_SCATTER = 50000

# -----------------------------------------------------------------------------
# Experimental data and constants
# -----------------------------------------------------------------------------
expdata = [0.09, 0.11, 0.27, 0.27]  # experimental mobilities in cm^2/Vs
mu_stdev = statistics.stdev(expdata)
mu_exp_mean = sum(expdata) / len(expdata)


mu_exp_target = mu_exp_mean

# How many experimental standard deviations to use:
# - STD_SCREEN: for defining "good" points used in screening / scatter
# - STD_RAINBOW: for the shaded band in the rainbow plot
STD_SCREEN = 1.0   # ±1σ window for "good" points
STD_RAINBOW = 0.5  # ±0.5σ band in the rainbow plot
p=0.5
occ_factor=(1-p)
#low carrier density limit
occ_factor=1

mu_screen_halfwidth = mu_stdev * STD_SCREEN
mu_rainbow_halfwidth = mu_stdev * STD_RAINBOW

mu_low = mu_exp_target - mu_rainbow_halfwidth
mu_high = mu_exp_target + mu_rainbow_halfwidth

#parse arguments 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse plotting mode and working directory."
    )

    parser.add_argument(
        "-t", "--jobtype",
        required=True,
        choices=["j", "1D", "3D"],
        help="Plot mode: j = default rainbow, 1D = true 1D correction, 3D = effective 3D plot"
    )
    #default is j for "j"unction effect

    parser.add_argument(
        "-d", "--directory",
        required=True,
        help="Directory containing data for plotting"
    )

    args = parser.parse_args()

    return args.jobtype, args.directory

def rainbow_output_filename(type_option):
    """
    Map plot type to correct output image filename.
    """
    mapping = {
        "j":   "default_rainbow.png",
        "1D":  "true1D_rainbow.png",
        "3D":  "eff3D_rainbow.png",
    }

    if type_option not in mapping:
        raise ValueError(f"Unknown type option: {type_option}")

    return mapping[type_option]


# Correction factor:
#   chain_err_adj = factor by which the naive 1D chain mobility (μ_chain)
#   overestimates the corrected mobility (μ_corr) that we compare to experiment.
#
#   μ_chain = chain_err_adj * μ_corr
#   μ_corr  = μ_chain / chain_err_adj
#
# Breakdown:
# - the correction factor must always be multiplicative of 2 because we need an extra 2
# - because I have not included the factor of 1/2 in the 1D chain equation in this code
# - 3/2 from junctions
# - 2.61 from additional network / geometry corrections
import sys
#1.5 from junctions
#2.61 from 3D conduction channels
structural_factor=1
structure_type,workingdir=parse_args()
if(workingdir[-1]=="/"):
    workingdir=workingdir[:-1]
if(structure_type=="3D"):
    structural_factor=1.5 * 2.61
if(structure_type=="j"):
    structural_factor=1.5
if(structure_type=="1D"):
    structural_factor=1
#divide by occ factor since it is divided in the mobility calc so the net effect is multiplying
chain_err_adj = 2 * structural_factor/occ_factor
print("FOUND MODE ",structure_type)
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


def find_rH_given_lambda(lmbda, mu_corr_target):
    """
    Given reorganization energy lambda (eV) and a TARGET CORRECTED mobility
    μ_corr_target (cm^2/Vs, comparable to experimental values), compute the
    combination R*H (in nm·eV), from which the coupling in eV can be obtained
    as (R*H)/R.

    IMPORTANT:
    - The original inversion formula was derived for the *chain mobility*
      μ_chain (i.e., without geometry/network corrections).
    - Here we are given μ_corr_target (the corrected/experimental level).
    - We therefore reconstruct μ_chain_target via:
          μ_chain_target = chain_err_adj * μ_corr_target
      and feed μ_chain_target into the original inversion.

    This keeps the correction factor consistent with the forward usage:
      rate -> μ_chain -> μ_corr
    and avoids any double counting.
    """
    hbar = 1.054571817e-34
    kB_local = 1.380649e-23  # J/K
    fund_charge_local = 1.602176e-19
    eV_to_J = 1.60218e-19
    nm_to_m = 1e-9
    T_local = 300
    lmbda_si = lmbda * eV_to_J
    cm2_to_m2 = 1e-4

    # Reconstruct the underlying chain mobility (SI units) from the
    # corrected target mobility
    mu_chain_target = mu_corr_target * chain_err_adj  # cm^2/Vs
    mu_chain_si = mu_chain_target * cm2_to_m2         # m^2/Vs

    # Original algebra (for μ_chain), with μ_chain_si inserted
    x_si = math.sqrt(
        (mu_chain_si * hbar / (fund_charge_local * math.pi * 2))
        * kB_local * T_local
        * math.sqrt(4 * math.pi * lmbda_si * kB_local * T_local)
        * math.exp(lmbda_si / (4 * kB_local * T_local))
    )

    # Return in units of nm·eV (divide out meters and Joules)
    return x_si / (nm_to_m * eV_to_J)


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

    Returns:
        stats (dict)
    """
    total_points = 0
    filtered_points = 0
    good_within_0p01 = 0
    good_within_screen = 0

    # Global min/max for parameters
    rnn_min, rnn_max = float("inf"), float("-inf")
    lam_min, lam_max = float("inf"), float("-inf")
    coup_min, coup_max = float("inf"), float("-inf")

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
                rnn_min = min(rnn_min, rnn_nm)
                rnn_max = max(rnn_max, rnn_nm)
                lam_min = min(lam_min, lambda_eV)
                lam_max = max(lam_max, lambda_eV)
                coup_min = min(coup_min, coupling_eV)
                coup_max = max(coup_max, coupling_eV)

                crate = float(crate_str)
                # Corrected mobility (after applying chain_err_adj)
                mu_corr = _corrected_mobility_from_rate(crate, rnn_nm)

                if abs(mu_corr - mu_exp_target) <= 0.01:
                    good_within_0p01 += 1
                if abs(mu_corr - mu_exp_target) <= mu_screen_halfwidth:
                    good_within_screen += 1

    param_min_max = {
        "Inter-cofactor Distance (nm)": (rnn_min, rnn_max),
        "Reorganization Energy (eV)": (lam_min, lam_max),
        "Coupling (eV)": (coup_min, coup_max),
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
# Second pass: sample every n-th good point for plotting
# -----------------------------------------------------------------------------
def second_pass(files, stride):
    """
    Second streaming pass.

    Keeps every n-th "good" point (within STD_SCREEN * σ_exp in corrected
    mobility) so that the number of stored good points is
    <= MAX_GOOD_POINTS_FOR_SCATTER.

    Returns:
        good_points: list of (lambda_eV, rnn_nm * coupling_eV, log10(mu_corr))
    """
    good_points = []
    good_idx = 0  # counts how many "good within screen" points seen so far

    for fname in files:
        print(f"[Second pass] Processing {fname} ...")
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                (rnn_nm, lambda_eV, coupling_eV), (crate_str, reqrate_str), flag = rec

                # Apply same parameter cuts as first pass
                if not (RNN_MIN <= rnn_nm <= RNN_MAX):
                    continue
                if not (LAMBDA_MIN <= lambda_eV <= LAMBDA_MAX):
                    continue
                if not (COUP_MIN <= coupling_eV <= COUP_MAX):
                    continue

                crate = float(crate_str)
                mu_corr = _corrected_mobility_from_rate(crate, rnn_nm)

                if abs(mu_corr - mu_exp_target) <= mu_screen_halfwidth:
                    good_idx += 1
                    # Keep only every stride-th good point
                    if good_idx % stride == 0:
                        val_for_y = rnn_nm * coupling_eV
                        log_mu = math.log10(mu_corr) if mu_corr > 0 else float("nan")
                        good_points.append((lambda_eV, val_for_y, log_mu))

    return good_points


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

    # Decide stride so that stored good points <= MAX_GOOD_POINTS_FOR_SCATTER
    if good_within_screen == 0:
        stride = 1  # nothing to keep anyway
    else:
        stride = max(1, math.ceil(good_within_screen / MAX_GOOD_POINTS_FOR_SCATTER))

    print(f"Good points within screening window (±{STD_SCREEN}σ_exp): {good_within_screen}")
    print(f"MAX_GOOD_POINTS_FOR_SCATTER: {MAX_GOOD_POINTS_FOR_SCATTER}")
    print(f"Sampling stride (keep every n-th good point): n = {stride}")

    # ---------------- Second pass: sample points for scatter ----------------
    good_points = second_pass(files, stride)
    print(f"Stored {len(good_points)} good points for scatter plotting.")

    # Parameter summary table
    params = {
        "Inter-cofactor Distance": [param_min_max["Inter-cofactor Distance (nm)"][0],
                                    param_min_max["Inter-cofactor Distance (nm)"][1]],
        "Reorganization Energy":   [param_min_max["Reorganization Energy (eV)"][0],
                                    param_min_max["Reorganization Energy (eV)"][1]],
        "Coupling":                [param_min_max["Coupling (eV)"][0],
                                    param_min_max["Coupling (eV)"][1]],
    }
    table = {"Parameter": [], "Minimum": [], "Maximum": []}
    units = {"Inter-cofactor Distance": "(nm)",
             "Reorganization Energy": "(eV)",
             "Coupling": "(eV)"}

    for p in params:
        table["Parameter"].append(p + " " + units[p])
        table["Minimum"].append(round(params[p][0], 5))
        table["Maximum"].append(round(params[p][1], 5))

    print(tabulate(table, headers="keys"))

    # ---------------------------------------------------------------------
    # 2D scatter for GOOD points only (sampled)
    # ---------------------------------------------------------------------
    if good_points:
        lmbdas = [g[0] for g in good_points]
        r_times_c = [g[1] for g in good_points]
        log_mu = [g[2] for g in good_points]

        plt.figure()
        sc = plt.scatter(lmbdas, r_times_c, c=log_mu)
        plt.colorbar(sc, label="log10(μ_corr [cm^2/Vs])")
        plt.xlabel("Reorganization Energy (eV)")
        plt.ylabel("Rnn * Coupling (nm·eV)")
        plt.title(
            "Sampled Good Parameter Points\n"
            f"(within ±{STD_SCREEN}σ_exp of target corrected mobility)"
        )
        plt.tight_layout()
        plt.savefig("good_points_scatter.png", dpi=300)
    else:
        print("No good points found for scatter plot.")

    # ---------------------------------------------------------------------
    # "Rainbow" plot: Viable couplings vs lambda for various R
    # ---------------------------------------------------------------------
    plt.figure()
    ax = plt.gca()

    lmbda_rng = np.linspace(0.16, 0.36, 50)
    R_vals = [0.75, 1, 1.5, 2, 2.5]

    colorsets = [
        ["tab:blue", "cornflowerblue"],
        ["tab:orange", "sandybrown"],
        ["tab:green", "olivedrab"],
        ["tab:red", "salmon"],
        ["tab:purple", "thistle"],
    ]

    for i, R in enumerate(R_vals):
        # central curve: target corrected/experimental mobility
        y_mid = [(find_rH_given_lambda(l, mu_exp_target) / R) * 1000
                 for l in lmbda_rng]

        # lower and upper bounds for the rainbow band (using STD_RAINBOW)
        y1 = [(find_rH_given_lambda(l, mu_low) / R) * 1000 for l in lmbda_rng]
        y2 = [(find_rH_given_lambda(l, mu_high) / R) * 1000 for l in lmbda_rng]

        plt.plot(lmbda_rng, y_mid, label=str(R), color=colorsets[i][0])
        plt.plot(lmbda_rng, y1, '--', color=colorsets[i][-1])
        plt.plot(lmbda_rng, y2, '--', color=colorsets[i][-1])

        plt.fill(
            np.append(lmbda_rng, lmbda_rng[::-1]),
            np.append(y1, y2[::-1]),
            colorsets[i][-1],
            alpha=0.4
        )

    plt.xlabel("Reorganization Energy (eV)")
    plt.ylabel("Coupling (meV)")
    plt.title(
        "Viable Couplings and Reorganization Energies at\n"
        "Various Inter-cofactor Distances"
    )
    plt.legend(title="Cofactor\nCenter-to-Center\nDistance (nm)")
    plt.tick_params(which='minor', length=5, width=0.25)
    plt.minorticks_on()
    for j, lbl in enumerate(ax.xaxis.get_ticklabels()):
        if j % 2 != 0:
            lbl.set_visible(False)
    plt.grid(visible=True, which='minor', alpha=0.25)
    plt.grid(which='major', alpha=0.75)
    plt.tight_layout()
    filename=rainbow_output_filename(structure_type)
    plt.savefig(filename, dpi=300)


if __name__ == "__main__":
    main()
