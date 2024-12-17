# pilot_mfpt_cgScaling.py
import sys
import argparse
sys.path.insert(0, '../common/')
sys.path.insert(0, '/home/ajs193/cable_bacteria/diffusion_topology/common/')
import random
import utils
import CB_lattice_walker as cbw
import convergence_criteria as convcrit
from multiprocessing import Pool, cpu_count
import os
import json

def detect_cpus():
    # Detect number of CPUs from SLURM environment or default to all cores
    return int(os.environ.get("SLURM_NTASKS", cpu_count()))

def run_single_job(params):
    cg_scale, num_junctions, max_steps, log_file, r = params
    result_writer = utils.ResultWriter(log_file, frequency=1)
    random.seed(r)
    handler = cbw.Handler(
        nfiber=25,  # Fixed as 2 fibers for this test
        njuctions=num_junctions,
        numb_repeats=1,  # Single repeat per iteration
        writer=result_writer,
        max_steps=max_steps,
        j_inter_nodes=cg_scale
    )
    steps = handler.do_one_round()
    return steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="MFPT Coarse-Graining Scaling Simulation")
    parser.add_argument("-n", type=int, required=True, help="Value for j_inter_nodes")
    parser.add_argument("--njunctions", type=int, default=100, help="Number of junctions")
    parser.add_argument("--max_steps", type=int, default=int(1e20), help="Maximum steps for each run")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    cg_scale = args.n
    num_junctions = args.njunctions
    max_steps = args.max_steps

    # File names
    log_file = f"cg_scaling_mfpt_results_{cg_scale}.log"
    conv_log_file = f"cg_convergence_log_{cg_scale}.json"  # Convergence log file

    # Initialize result writer and logger
    result_writer = utils.ResultWriter(log_file, frequency=1)
    convergence_logger = utils.mlogger(conv_log_file)

    # Rolling average convergence criteria setup
    conv_checker = convcrit.Rolling_Av_Conv(property_key=["mfpt"], threshold_limit=0.05, usefile=False)

    # Results storage
    mfpt_values = []
    is_converged = False
    repeat_count = 0
    c = 0

    # Detect CPU count
    num_cpus = detect_cpus()
    print(f"Running with {num_cpus} parallel workers")

    print(f"Running MFPT calculations for j_inter_nodes={cg_scale}...")

    # Multiprocessing pool for parallel execution
    with Pool(num_cpus) as pool:
        while not is_converged:
            params = [(cg_scale, num_junctions, max_steps, log_file, c + i) for i in range(num_cpus)]
            c += len(params)
            steps_list = pool.map(run_single_job, params)
            print(steps_list)
            mfpt_values.extend(steps_list)
            repeat_count += len(steps_list)

            # Check convergence using rolling average
            data = [{"mfpt": step} for step in steps_list]
            is_converged, conv_data = conv_checker.check_conv(data, verbose=True)

            # Log verbose output to file
            convergence_logger.log(json.dumps({
                "cg_scale": cg_scale,
                "run_count": c,
                "convergence_data": conv_data
            }))

    # Final output of results
    avg_mfpt = sum(mfpt_values) / len(mfpt_values)
    print(f"\nFinal Results for j_inter_nodes={cg_scale}:")
    print(f"Mean First Passage Time={avg_mfpt}, Total Runs={repeat_count}")

    # Write to log file
    result_writer.write({"cg_scale": cg_scale, "mean_mfpt": avg_mfpt, "total_runs": repeat_count}, force=True)

if __name__ == "__main__":
    main()
