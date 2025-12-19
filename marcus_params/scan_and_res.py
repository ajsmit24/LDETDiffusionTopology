# -*- coding: utf-8 -*-
"""
Single-pass driver for hopping-rate parameter scan + analysis.

- Runs everything in ONE invocation:
    * defines parameter grid
    * two parallel passes over the grid
        1) count-only/statistics (no storage)
        2) sampling of "good" points for plotting
    * produces plots + printed stats at the end

- Only "good" points (within 1σ of target mobility) are stored,
  and even those are thinned so that at most MAX_GOOD_POINTS_FOR_SCATTER
  are kept.

- Parallelization is via multiprocessing on a single node;
  no Slurm array / chunk files needed.
"""

import os
import math
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
import statistics

print("TOP")

# ======================================================================
# Fundamental constants
# ======================================================================
kB = 1.380649e-23          # J/K
fund_charge = 1.602176e-19 # Coulomb (proton charge)
hbar = 1.054571817e-34     # J*s

# Experimental constants
# https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-020-76671-5/MediaObjects/41598_2020_76671_MOESM1_ESM.pdf
# mobility: cm^2 V^-1 s^-1
mu = (0.27 + 0.11) / 2
# convert to m^2 V^-1 s^-1
mu = mu / (100.0**2)

# Fixed parameters
T = 300.0
delG = 0.0

# ======================================================================
# Variable parameter ranges (in original units)
# ======================================================================
v_params = {
    "Rnn":      [0.5, 2.5],    # nm
    "Coupling": [0.001, 0.03], # eV
    "lambda":   [0.16, 0.36]   # eV
}

# ----------------------------------------------------------------------
# Parameter spacing (in original units)
# ----------------------------------------------------------------------
# These define the approximate spacing in each dimension.
# The actual linspace will include both endpoints, so the true step
# is (max - min) / (N - 1), with N determined below.
STEP_SIZE = {
    "Rnn":      0.001,   # nm
    "lambda":   0.0001,  # eV
    "Coupling": 0.0001,  # eV
}

DEFAULT_LINESTEPS = {
    "Rnn":      math.ceil((v_params["Rnn"][1]      - v_params["Rnn"][0])      / STEP_SIZE["Rnn"]),
    "lambda":   math.ceil((v_params["lambda"][1]   - v_params["lambda"][0])   / STEP_SIZE["lambda"]),
    "Coupling": math.ceil((v_params["Coupling"][1] - v_params["Coupling"][0]) / STEP_SIZE["Coupling"]),
}

# Unit conversions to SI
unit_convert = {
    "Rnn":      1e-9,        # nm -> m
    "Coupling": 1.60218e-19, # eV -> J
    "lambda":   1.60218e-19  # eV -> J
}

# ======================================================================
# Plot / font settings
# ======================================================================
matplotlib.rcParams['figure.dpi'] = 600
font = {'family': 'normal', 'weight': 'bold', 'size': 11}
matplotlib.rc('font', **font)

# ======================================================================
# Experimental data and "good" criteria (shared)
# ======================================================================
expdata = [0.09, 0.11, 0.27, 0.27]
mu_stdev = statistics.stdev(expdata)
av_mu_exp = sum(expdata) / len(expdata)

# Target mobility used in the script (central value)
av_mu_target = 0.185
mu_low = av_mu_target - mu_stdev * 0.5
mu_high = av_mu_target + mu_stdev * 0.5

# Error factor
chain_err_adj = 1.6 * 2

# Physical constants for mobility back-calculation
kB_mob = 1.380649e-23      # J/K
fund_charge_mob = 1.602176e-19
T_mob = 300.0

# Filtering ranges (same as in original analysis script)
RNN_MIN, RNN_MAX = 0.5, 2.5          # nm
LAMBDA_MIN, LAMBDA_MAX = 0.15, 0.361 # eV
COUP_MIN, COUP_MAX = 0.001, 0.03     # eV

# Max number of "good" points we keep for scatter plotting
MAX_GOOD_POINTS_FOR_SCATTER = 50000


# ======================================================================
# Physics helper functions
# ======================================================================
def calc_req_hopping(mu_val, r, T_val):
    """Required hopping rate from mobility, distance, and temperature."""
    return (kB * T_val * mu_val) / (fund_charge * r * r)


def calc_hop_from_params(lmbd, coupling, T_val):
    """Marcus hopping rate for given lambda and coupling."""
    pref = (2.0 * math.pi / hbar)
    gauss_pref = 1.0 / math.sqrt(4.0 * math.pi * lmbd * kB * T_val)
    exponent = -((delG + lmbd) ** 2) / (4.0 * lmbd * kB * T_val)
    return pref * gauss_pref * (coupling ** 2) * math.exp(exponent)


def mobility_from_rate(k, r_nm):
    """
    Mobility in cm^2/(V·s) from rate k (1/s) and distance r (nm).

    mu = (e / (k_B T)) * k * r^2
    r in nm, final mu converted to cm^2/Vs.
    """
    r = r_nm
    mu_val = fund_charge_mob / (kB_mob * T_mob) * k * (r ** 2)  # nm^2 V^-1 s^-1
    mu_val = mu_val / 1e14  # nm^2 -> cm^2
    return mu_val


def find_rH_given_lambda(lmbda):
    """
    Original find_rH_given_lambda, using global av_mu_target.
    """
    hbar_loc = 1.054571817e-34
    kB_local = 1.380649e-23  # J/K
    fund_charge_local = 1.602176e-19
    eV_to_J = 1.60218e-19
    nm_to_m = 1e-9
    T_local = 300
    lmbda_si = lmbda * eV_to_J
    cm2_to_m2 = 1e-4

    av_mu_si = av_mu_target * cm2_to_m2  # uses av_mu_target

    x_si = math.sqrt(
        (chain_err_adj * av_mu_si * hbar_loc / (fund_charge_local * math.pi * 2))
        * kB_local * T_local
        * math.sqrt(4 * math.pi * lmbda_si * kB_local * T_local)
        * math.exp(lmbda_si / (4 * kB_local * T_local))
    )

    y = (1 / chain_err_adj) * (fund_charge_local / (kB_local * T_local)) \
        * (1 / math.sqrt(4 * math.pi * lmbda_si * kB_local * T_local)) \
        * math.exp(-1 * lmbda_si / (4 * kB_local * T_local)) \
        * ((0.01 * nm_to_m * eV_to_J) ** 2) * math.pi * 2 / hbar_loc

    return x_si / (nm_to_m * eV_to_J)


# ======================================================================
# Worker + task generator for multiprocessing
# ======================================================================
def run(task):
    """
    Worker function.

    Parameters
    ----------
    task : dict
        {
          "Rnn_nm": float,
          "lambda_eV": float,
          "coupling_eV": float,
          "Rnn_SI": float,
          "lambda_SI": float,
          "coupling_SI": float,
        }

    Returns
    -------
    result : list
        [
          [Rnn_nm, lambda_eV, coupling_eV],
          [crate_str, reqrate_str],
          int(crate > reqrate)
        ]
    (same shape as original scanner for consistency)
    """
    Rnn_nm      = task["Rnn_nm"]
    lambda_eV   = task["lambda_eV"]
    coupling_eV = task["coupling_eV"]

    Rnn_SI      = task["Rnn_SI"]
    lambda_SI   = task["lambda_SI"]
    coupling_SI = task["coupling_SI"]

    # Compute rates
    reqrate = calc_req_hopping(mu, Rnn_SI, T)
    crate   = calc_hop_from_params(lambda_SI, coupling_SI, T)

    hops = ["{:e}".format(crate), "{:e}".format(reqrate)]
    flag = int(crate > reqrate)

    return [[Rnn_nm, lambda_eV, coupling_eV], hops, flag]


def task_generator(linespaces, start_index, end_index):
    """
    Yield tasks corresponding to global indices in [start_index, end_index).

    We treat the 3D grid (Rnn, lambda, Coupling) as a flattened 1D array.
    """
    R_vals = linespaces["Rnn"]      # in SI
    L_vals = linespaces["lambda"]   # in SI
    C_vals = linespaces["Coupling"] # in SI

    nr = len(R_vals)
    nl = len(L_vals)
    nc = len(C_vals)

    total = nr * nl * nc
    end_index = min(end_index, total)

    # Mapping from flat index -> (i, j, k)
    for g in range(start_index, end_index):
        i = g // (nl * nc)
        rem = g % (nl * nc)
        j = rem // nc
        k = rem % nc

        Rnn_SI = R_vals[i]
        lambda_SI = L_vals[j]
        coupling_SI = C_vals[k]

        # convert back to original units for storage
        Rnn_nm = Rnn_SI / unit_convert["Rnn"]
        lambda_eV = lambda_SI / unit_convert["lambda"]
        coupling_eV = coupling_SI / unit_convert["Coupling"]

        yield {
            "Rnn_nm": Rnn_nm,
            "lambda_eV": lambda_eV,
            "coupling_eV": coupling_eV,
            "Rnn_SI": Rnn_SI,
            "lambda_SI": lambda_SI,
            "coupling_SI": coupling_SI,
        }


# ======================================================================
# First pass: count-only and stats (no storage of points)
# ======================================================================
def first_pass(linespaces, nprocs, chunksize):
    """
    Parallel first pass over the full grid.

    Computes:
    - total_points (product of grid dimensions)
    - filtered_points (after parameter cuts)
    - min/max for each parameter (within filtered set)
    - count of points within 0.01 of av_mu_target
    - count of points within 1 sigma of av_mu_target

    Returns:
        stats (dict)
    """
    nr = len(linespaces["Rnn"])
    nl = len(linespaces["lambda"])
    nc = len(linespaces["Coupling"])
    total_points = nr * nl * nc

    filtered_points = 0
    good_within_0p01 = 0
    good_within_stdev = 0

    # Global min/max for parameters
    rnn_min, rnn_max = float("inf"), float("-inf")
    lam_min, lam_max = float("inf"), float("-inf")
    coup_min, coup_max = float("inf"), float("-inf")

    print("=== First pass: counting + stats (no points stored) ===")
    print(f"Total grid points: {total_points}")

    gen = task_generator(linespaces, 0, total_points)

    with Pool(processes=nprocs) as pool:
        for result in pool.imap_unordered(run, gen, chunksize=chunksize):
            (rnn_nm, lambda_eV, coupling_eV), (crate_str, reqrate_str), flag = result

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
            mu_calc = mobility_from_rate(crate, rnn_nm)

            if abs(mu_calc - av_mu_target) <= 0.01:
                good_within_0p01 += 1
            if abs(mu_calc - av_mu_target) <= mu_stdev:
                good_within_stdev += 1

    if filtered_points == 0:
        param_min_max = {}
    else:
        param_min_max = {
            "Inter-cofactor Distance (nm)": (rnn_min, rnn_max),
            "Reorganization Energy (eV)":   (lam_min, lam_max),
            "Coupling (eV)":                (coup_min, coup_max),
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


# ======================================================================
# Second pass: sample every n-th good point for plotting
# ======================================================================
def second_pass(linespaces, nprocs, chunksize, stride):
    """
    Parallel second pass over the full grid.

    Keeps every n-th "good" point (within 1 sigma) so that the number
    of stored good points is <= MAX_GOOD_POINTS_FOR_SCATTER.

    Returns:
        good_points: list of (lambda_eV, rnn_nm * coupling_eV, log10(mu_calc))
    """
    nr = len(linespaces["Rnn"])
    nl = len(linespaces["lambda"])
    nc = len(linespaces["Coupling"])
    total_points = nr * nl * nc

    good_points = []
    good_idx = 0  # counts how many "good within stdev" points seen so far

    print("=== Second pass: sampling 'good' points for plotting ===")
    print(f"Total grid points: {total_points}")
    print(f"Sampling stride: n = {stride}")

    gen = task_generator(linespaces, 0, total_points)

    with Pool(processes=nprocs) as pool:
        for result in pool.imap_unordered(run, gen, chunksize=chunksize):
            (rnn_nm, lambda_eV, coupling_eV), (crate_str, reqrate_str), flag = result

            # Apply same parameter cuts as first pass
            if not (RNN_MIN <= rnn_nm <= RNN_MAX):
                continue
            if not (LAMBDA_MIN <= lambda_eV <= LAMBDA_MAX):
                continue
            if not (COUP_MIN <= coupling_eV <= COUP_MAX):
                continue

            crate = float(crate_str)
            mu_calc = mobility_from_rate(crate, rnn_nm)

            if abs(mu_calc - av_mu_target) <= mu_stdev:
                good_idx += 1
                # Keep only every stride-th good point
                if good_idx % stride == 0:
                    val_for_y = rnn_nm * coupling_eV
                    log_mu = math.log10(mu_calc) if mu_calc > 0 else float("nan")
                    good_points.append((lambda_eV, val_for_y, log_mu))

    return good_points


# ======================================================================
# CLI + main driver
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-run parallel scan + analysis for hopping-rate grid."
    )

    # Grid resolution overrides
    parser.add_argument("--rnn-steps", type=int, default=None,
                        help=f"Number of linspace points for Rnn (default ~{DEFAULT_LINESTEPS['Rnn']}).")
    parser.add_argument("--lambda-steps", type=int, default=None,
                        help=f"Number of linspace points for lambda (default ~{DEFAULT_LINESTEPS['lambda']}).")
    parser.add_argument("--coupling-steps", type=int, default=None,
                        help=f"Number of linspace points for coupling (default ~{DEFAULT_LINESTEPS['Coupling']}).")

    # Multiprocessing config
    parser.add_argument("--nprocs", type=int, default=None,
                        help="Number of worker processes (default: SLURM_CPUS_PER_TASK or all cores).")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Chunksize for imap_unordered.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Determine linesteps using top-level defaults + overrides
    # ------------------------------------------------------------------
    linesteps = {
        "Rnn":      args.rnn_steps      or DEFAULT_LINESTEPS["Rnn"],
        "lambda":   args.lambda_steps   or DEFAULT_LINESTEPS["lambda"],
        "Coupling": args.coupling_steps or DEFAULT_LINESTEPS["Coupling"],
    }

    print("Linesteps:", linesteps)

    # ------------------------------------------------------------------
    # Build linespaces in SI units
    # ------------------------------------------------------------------
    v_params_SI = {}
    for k, v in v_params.items():
        v_params_SI[k] = [p * unit_convert[k] for p in v]

    linespaces = {
        "Rnn":      np.linspace(v_params_SI["Rnn"][0],      v_params_SI["Rnn"][1],      linesteps["Rnn"]),
        "lambda":   np.linspace(v_params_SI["lambda"][0],   v_params_SI["lambda"][1],   linesteps["lambda"]),
        "Coupling": np.linspace(v_params_SI["Coupling"][0], v_params_SI["Coupling"][1], linesteps["Coupling"]),
    }

    nr = linesteps["Rnn"]
    nl = linesteps["lambda"]
    nc = linesteps["Coupling"]
    total_points = nr * nl * nc

    print(f"Total parameter combinations: {total_points}")

    # ------------------------------------------------------------------
    # Number of processes
    # ------------------------------------------------------------------
    if args.nprocs is not None:
        nprocs = args.nprocs
    else:
        # Respect SLURM_CPUS_PER_TASK if set
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            try:
                nprocs = int(slurm_cpus)
            except ValueError:
                nprocs = cpu_count() or 1
        else:
            nprocs = cpu_count() or 1

    nprocs = max(1, nprocs)
    print(f"Using {nprocs} worker processes with chunksize={args.chunksize}")

    # ======================= FIRST PASS ==========================
    stats = first_pass(linespaces, nprocs, args.chunksize)
    total = stats["total_points"]
    filtered = stats["filtered_points"]
    good_counts = stats["good_counts"]
    param_min_max = stats["param_min_max"]

    good_within_stdev = good_counts["within_stdev"]
    good_within_0p01  = good_counts["within_0p01"]

    print("%" * 20)
    print(f"Total points (grid size): {total}")
    print(f"Points after parameter filters: {filtered}")
    print("%" * 20)

    # Fractions of good points (relative to filtered set)
    if filtered > 0:
        frac_0p01 = good_within_0p01 / filtered
        frac_stdev = good_within_stdev / filtered
    else:
        frac_0p01 = frac_stdev = 0.0

    print("Fraction within 0.01 of av_mu_target:",
          frac_0p01, good_within_0p01, filtered)
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

    # ======================= SECOND PASS =========================
    if good_within_stdev > 0:
        good_points = second_pass(linespaces, nprocs, args.chunksize, stride)
    else:
        good_points = []

    print(f"Stored {len(good_points)} good points for scatter plotting.")

    # ======================= REPORTING & PLOTS ===================
    # Parameter summary table
    if param_min_max:
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
    else:
        print("No filtered points → no parameter min/max to report.")

    # -----------------------------------------------------------------
    # 2D scatter for GOOD points only (sampled)
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # "Rainbow" plot: Viable couplings vs lambda for various R
    # -----------------------------------------------------------------
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
