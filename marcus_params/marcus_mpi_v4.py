# -*- coding: utf-8 -*-
"""
Parameter scan for hopping rates using multiprocessing and Slurm-friendly chunking.

- Parallelized with multiprocessing on a single node
- Designed to be run as a Slurm array job (each task handles a chunk of the 3D grid)
- Avoids building a giant list of all parameter combinations in memory
- Streams results to disk as JSON Lines (one record per line)

Usage examples
--------------
# Single-node, single-chunk test (default steps)
python scan_hopping_parallel.py

# Specify steps explicitly
python scan_hopping_parallel.py --rnn-steps 2000 --lambda-steps 200 --coupling-steps 290

# Slurm array (10 chunks, each with 8 CPUs)
# In your Slurm script:
#   #SBATCH --array=0-9
#   #SBATCH --cpus-per-task=8
#   srun python scan_hopping_parallel.py \
#       --num-chunks 10 \
#       --chunk-index ${SLURM_ARRAY_TASK_ID} \
#       --output mparams_chunk${SLURM_ARRAY_TASK_ID}.jsonl
"""

import os
import sys
import math
import json
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np

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
    "Rnn":      [0.5, 2.5],   # nm
    "Coupling": [0.001, 0.03],# eV
    "lambda":   [0.16, 0.36]  # eV
}

# Unit conversions to SI
unit_convert = {
    "Rnn":      1e-9,        # nm -> m
    "Coupling": 1.60218e-19, # eV -> J
    "lambda":   1.60218e-19  # eV -> J
}


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


# ======================================================================
# Worker function for multiprocessing
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


# ======================================================================
# Generator to iterate over a subrange of the 3D grid
# ======================================================================
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

    if end_index > total:
        end_index = total

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
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Parallel parameter scan for hopping rates."
    )

    # Optional overrides for steps
    parser.add_argument("--rnn-steps", type=int, default=None,
                        help="Number of linspace points for Rnn.")
    parser.add_argument("--lambda-steps", type=int, default=None,
                        help="Number of linspace points for lambda.")
    parser.add_argument("--coupling-steps", type=int, default=None,
                        help="Number of linspace points for coupling.")

    # Chunking for Slurm array
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Total number of chunks over the full grid.")
    parser.add_argument("--chunk-index", type=int, default=0,
                        help="Index of this chunk (0-based).")

    # Output
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output filename (JSON Lines).")

    # Multiprocessing config
    parser.add_argument("--nprocs", type=int, default=None,
                        help="Number of worker processes (default: SLURM_CPUS_PER_TASK or all cores).")
    parser.add_argument("--chunksize", type=int, default=1000,
                        help="Chunksize for imap_unordered.")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Determine linesteps (default: same logic as your original script)
    # ------------------------------------------------------------------
    default_linesteps = {
        "Rnn":      math.ceil((v_params["Rnn"][1]      - v_params["Rnn"][0])      / 0.001),
        "lambda":   math.ceil((v_params["lambda"][1]   - v_params["lambda"][0])   / 0.0001),
        "Coupling": math.ceil((v_params["Coupling"][1] - v_params["Coupling"][0]) / 0.0001),
    }

    linesteps = {
        "Rnn":      args.rnn_steps      or default_linesteps["Rnn"],
        "lambda":   args.lambda_steps   or default_linesteps["lambda"],
        "Coupling": args.coupling_steps or default_linesteps["Coupling"],
    }

    print("Linesteps:", linesteps)

    # ------------------------------------------------------------------
    # Build linespaces in SI units (small arrays, totally safe)
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
    # Chunking logic (for Slurm arrays)
    # ------------------------------------------------------------------
    num_chunks = max(1, args.num_chunks)
    chunk_index = args.chunk_index

    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError(f"chunk_index {chunk_index} is out of range for num_chunks={num_chunks}")

    chunk_size = math.ceil(total_points / num_chunks)
    start_index = chunk_index * chunk_size
    end_index = min(start_index + chunk_size, total_points)

    if start_index >= total_points:
        print(f"Chunk {chunk_index} has no work (start_index >= total_points). Exiting.")
        return

    print(f"Processing chunk {chunk_index+1}/{num_chunks}: "
          f"indices [{start_index}, {end_index}) ({end_index - start_index} points)")

    # ------------------------------------------------------------------
    # Output filename
    # ------------------------------------------------------------------
    if args.output is None:
        output_file = f"mparams_chunk{chunk_index}.jsonl"
    else:
        output_file = args.output

    # ------------------------------------------------------------------
    # Number of processes
    # ------------------------------------------------------------------
    if args.nprocs is not None:
        nprocs = args.nprocs
    else:
        # If running under Slurm, respect SLURM_CPUS_PER_TASK if set
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
    print(f"Writing results to: {output_file}")

    # ------------------------------------------------------------------
    # Run multiprocessing pool and stream results to disk
    # ------------------------------------------------------------------
    gen = task_generator(linespaces, start_index, end_index)

    with Pool(processes=nprocs) as pool, open(output_file, "w") as f:
        # imap_unordered yields results as they’re ready; we write them immediately
        for result in pool.imap_unordered(run, gen, chunksize=args.chunksize):
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
