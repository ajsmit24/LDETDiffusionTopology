# pilot_mfpt_cgScaling.py
import sys
sys.path.insert(0, '../common/')
sys.path.insert(0, '/home/ajs193/cable_bacteria/diffusion_topology/common/')
import random
import utils
import CB_lattice_walker as cbw
import convergence_criteria as convcrit
from multiprocessing import Pool, cpu_count
import os


def detect_cpus():
    # Detect number of CPUs from SLURM environment or default to all cores
    return int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))

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

def main():
    # Fixed parameters
    num_junctions = 100
    max_steps = int(1e20)
    cg_scales = range(1, 251, 25)  # Coarse-graining scales (j_inter_nodes)
    log_file = "cg_scaling_mfpt_results.log"

    # Initialize result writer
    result_writer = utils.ResultWriter(log_file, frequency=1)

    # Rolling average convergence criteria setup
    conv_checker = convcrit.Rolling_Av_Conv(property_key=["mfpt"], threshold_limit=0.05, usefile=False)

    # Results storage
    results = {}

    # Detect CPU count
    num_cpus = detect_cpus()
    print(f"Running with {num_cpus} parallel workers")

    # Loop through different coarse-graining scales
    for cg_scale in cg_scales:
        print(f"Running MFPT calculations for j_inter_nodes={cg_scale}...")
        mfpt_values = []
        is_converged = False
        repeat_count = 0
        c=0
        # Multiprocessing pool for parallel execution
        with Pool(num_cpus) as pool:
            while not is_converged:
                params = [(cg_scale, num_junctions, max_steps, log_file,c+i) for i in range(num_cpus)]
                c+=len(params)
                steps_list = pool.map(run_single_job, params)
                mfpt_values.extend(steps_list)
                repeat_count += len(steps_list)

                # Check convergence using rolling average
                is_converged, conv_data = conv_checker.check_conv([
                    {"mfpt": step} for step in steps_list
                ], verbose=False)

        # Store results for the current scale
        avg_mfpt = sum(mfpt_values) / len(mfpt_values)
        results[cg_scale] = avg_mfpt
        print(f"j_inter_nodes={cg_scale}: MFPT={avg_mfpt}, Repeats={repeat_count}")

    # Final output of results
    print("\nFinal Results: Coarse-Graining Scaling Test")
    for scale, mfpt in results.items():
        print(f"j_inter_nodes={scale}, Mean First Passage Time={mfpt}")

    # Write to log file
    result_writer.write({"cg_scaling_results": results}, force=True)

if __name__ == "__main__":
    main()
