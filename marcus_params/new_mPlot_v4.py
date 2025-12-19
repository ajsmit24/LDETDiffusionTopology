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
expdata = [0.09, 0.11, 0.27, 0.27]
mu_stdev = statistics.stdev(expdata)
av_mu_exp = sum(expdata) / len(expdata)

# Target mobility used in the script
av_mu_target = 0.185
mu_low = av_mu_target - mu_stdev * 0.5
mu_high = av_mu_target + mu_stdev * 0.5

# Error factor
#factor of 2 from extra two in mobility equation
#   This is essentially dividing by 1/2 in the 1D chain expression
#   but instead I equivelently multiply the target mobility by 2
#3/2 from junctions
chain_err_adj = 2*1.5*2.61
print("USING CF=2 for truly 1D chains")
#print("USING CF*=2.61 for inter-connected chains")
print("TOTAL CORRECTION FACTOR",chain_err_adj)

# Physical constants
kB = 1.380649e-23      # J/K
fund_charge = 1.602176e-19
T = 300.0

# Filtering ranges (same as in original script)
RNN_MIN, RNN_MAX = 0.5, 2.5         # nm
LAMBDA_MIN, LAMBDA_MAX = 0.15, 0.361  # eV
COUP_MIN, COUP_MAX = 0.001, 0.03    # eV


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def __mobility_from_rate(k, r_nm):
    """
    Mobility in cm^2/(V·s) from rate k (1/s) and distance r (nm).

    mu = (e / (k_B T)) * k * r^2
    r in nm, final mu converted to cm^2/Vs.
    """
    r = r_nm
    mu = fund_charge / (kB * T) * k * (r ** 2)  # nm^2 V^-1 s^-1
    mu = mu / 1e14  # nm^2 -> cm^2
    return mu


def find_rH_given_lambda(lmbda):
    """
    Original find_rH_given_lambda, using global av_mu_target.
    """
    hbar = 1.054571817e-34
    kB_local = 1.380649e-23  # J/K
    fund_charge_local = 1.602176e-19
    eV_to_J = 1.60218e-19
    nm_to_m = 1e-9
    T_local = 300
    lmbda_si = lmbda * eV_to_J
    cm2_to_m2 = 1e-4

    av_mu_si = av_mu_target * cm2_to_m2  # uses av_mu_target

    x_si = math.sqrt(
        (chain_err_adj * av_mu_si * hbar / (fund_charge_local * math.pi * 2))
        * kB_local * T_local
        * math.sqrt(4 * math.pi * lmbda_si * kB_local * T_local)
        * math.exp(lmbda_si / (4 * kB_local * T_local))
    )

    y = (1 / chain_err_adj) * (fund_charge_local / (kB_local * T_local)) \
        * (1 / math.sqrt(4 * math.pi * lmbda_si * kB_local * T_local)) \
        * math.exp(-1 * lmbda_si / (4 * kB_local * T_local)) \
        * ((0.01 * nm_to_m * eV_to_J) ** 2) * math.pi * 2 / hbar

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
    - count of points within 0.01 of av_mu_target
    - count of points within 1 sigma of av_mu_target

    Returns:
        stats (dict)
    """
    total_points = 0
    filtered_points = 0
    good_within_0p01 = 0
    good_within_stdev = 0

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
                mu_calc = __mobility_from_rate(crate, rnn_nm)

                if abs(mu_calc - av_mu_target) <= 0.01:
                    good_within_0p01 += 1
                if abs(mu_calc - av_mu_target) <= mu_stdev:
                    good_within_stdev += 1

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
            "within_stdev": good_within_stdev,
        },
    }

    return stats


# -----------------------------------------------------------------------------
# Second pass: sample every n-th good point for plotting
# -----------------------------------------------------------------------------
def second_pass(files, stride):
    """
    Second streaming pass.

    Keeps every n-th "good" point (within 1 sigma) so that the number
    of stored good points is <= MAX_GOOD_POINTS_FOR_SCATTER.

    Returns:
        good_points: list of (lambda_eV, rnn_nm * coupling_eV, log10(mu_calc))
    """
    good_points = []
    good_idx = 0  # counts how many "good within stdev" points seen so far

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
                mu_calc = __mobility_from_rate(crate, rnn_nm)

                if abs(mu_calc - av_mu_target) <= mu_stdev:
                    good_idx += 1
                    # Keep only every stride-th good point
                    if good_idx % stride == 0:
                        val_for_y = rnn_nm * coupling_eV
                        log_mu = math.log10(mu_calc) if mu_calc > 0 else float("nan")
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

    good_within_stdev = good_counts["within_stdev"]

    print("%" * 20)
    print(f"Total points read: {total}")
    print(f"Points after parameter filters: {filtered}")
    print("%" * 20)

    # Fractions of good points (relative to filtered set)
    if filtered > 0:
        frac_0p01 = good_counts["within_0p01"] / filtered
        frac_stdev = good_within_stdev / filtered
    else:
        frac_0p01 = frac_stdev = 0.0

    print("Fraction within 0.01 of av_mu_target:",
          frac_0p01, good_counts["within_0p01"], filtered)
    print("Fraction within 1 stdev of av_mu_target:",
          frac_stdev, good_within_stdev, filtered)

    # Decide stride so that stored good points <= MAX_GOOD_POINTS_FOR_SCATTER
    if good_within_stdev == 0:
        stride = 1  # nothing to keep anyway
    else:
        stride = max(1, math.ceil(good_within_stdev / MAX_GOOD_POINTS_FOR_SCATTER))

    print(f"Good points within 1σ: {good_within_stdev}")
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
        plt.colorbar(sc, label="log10(mu_calc [cm^2/Vs])")
        plt.xlabel("Reorganization Energy (eV)")
        plt.ylabel("Rnn * Coupling (nm·eV)")
        plt.title("Sampled Good Parameter Points (within 1σ of target mobility)")
        plt.tight_layout()
        plt.savefig("good_points_scatter.png", dpi=300)
    else:
        print("No good points found for scatter plot.")

    # ---------------------------------------------------------------------
    # "Rainbow" plot: Viable couplings vs lambda for various R (unchanged)
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

    i = 0
    for R in R_vals:
        global av_mu_target

        # central curve
        av_mu_target = 0.185
        y_mid = [(find_rH_given_lambda(l) / R) * 1000 for l in lmbda_rng]
        plt.plot(lmbda_rng, y_mid, label=str(R), color=colorsets[i][0])

        # lower bound: mu_low
        av_mu_target = mu_low
        y1 = [(find_rH_given_lambda(l) / R) * 1000 for l in lmbda_rng]

        # upper bound: mu_high
        av_mu_target = mu_high
        y2 = [(find_rH_given_lambda(l) / R) * 1000 for l in lmbda_rng]

        plt.plot(lmbda_rng, y1, '--', color=colorsets[i][-1])
        plt.plot(lmbda_rng, y2, '--', color=colorsets[i][-1])

        plt.fill(
            np.append(lmbda_rng, lmbda_rng[::-1]),
            np.append(y1, y2[::-1]),
            colorsets[i][-1],
            alpha=0.4
        )

        i += 1

    plt.xlabel("Reorganization Energy (eV)")
    plt.ylabel("Coupling (meV)")
    plt.title("Viable Couplings and Reorganization Energies at\nVarious Inter-cofactor Distances")
    plt.legend(title="Cofactor\nCenter-to-Center\nDistance (nm)")
    plt.tick_params(which='minor', length=5, width=0.25)
    plt.minorticks_on()
    [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0]
    plt.grid(visible=True, which='minor', alpha=0.25)
    plt.grid(which='major', alpha=0.75)
    plt.tight_layout()
    plt.savefig('rainbow_good.png', dpi=300)


if __name__ == "__main__":
    main()
