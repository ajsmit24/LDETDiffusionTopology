# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:13:17 2024

@author: smith
"""

import os
import subprocess

def create_slurm_job(template_file, output_dir, n_value):
    """
    Creates a SLURM job script from a template file by replacing placeholders.

    Parameters:
        template_file (str): Path to the template file.
        output_dir (str): Directory to store generated SLURM scripts.
        n_value (int): Value of n (coarse-graining scale) to insert into the script.
    Returns:
        str: Path to the generated SLURM script.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Output script file name
    output_file = os.path.join(output_dir, "job_n{n}.sh".replace("{n}", str(n_value)))

    # Read the template and replace placeholders
    with open(template_file, 'r') as f:
        template_content = f.read()

    # Replace placeholder for n value
    job_content = template_content.replace("{n}", str(n_value))

    # Write the modified content to the new file
    with open(output_file, 'w') as f:
        f.write(job_content)

    return output_file

def submit_slurm_job(script_path):
    """
    Submits the SLURM job script using sbatch.

    Parameters:
        script_path (str): Path to the SLURM script to be submitted.
    """
    try:
        result = subprocess.run(["sbatch", script_path], check=True, capture_output=True, text=True)
        print(f"Submitted job: {script_path}")
        print("Standard Output:")
        print(result.stdout)
        print("Standard Error:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job {script_path}: {e}")
        print("Error Output:")
        print(e.stderr)

def main():
    # Path to the SLURM script template
    template_file = "submit.tmplt"

    # Directory to store generated SLURM scripts
    output_dir = ""

    # Values of n to run the jobs for
    n_values = [1, 10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]

    # Iterate over n values, generate SLURM scripts, and submit jobs
    for n in n_values:
        print(f"Generating and submitting job for n={n}...")
        script_path = create_slurm_job(template_file, output_dir, n)
        submit_slurm_job(script_path)

if __name__ == "__main__":
    main()
