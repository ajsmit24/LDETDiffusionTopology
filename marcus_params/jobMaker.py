#!/usr/bin/env python3
import os
import subprocess
import re

PARTITION = "et2024"
MEM = "100G"
TIME = "20:00:00"
N_TASKS = 1

PYTHON_SCRIPT = "revise_mPlot_v5.py"

jobs = [
    {
        "label": "SMALL 1K, option j → default_rainbow.png",
        "job_name": "SMALLdef_j",
        "workdir": "mpSMALL_1K",
        "option": "j",
        "dest_png": "default_rainbow.png",
        "script_tag": "SMALL_1K_j",
    },
    {
        "label": "MED 1M, option j → default_rainbow.png",
        "job_name": "MEDdef_j",
        "workdir": "mpMED_1M",
        "option": "j",
        "dest_png": "default_rainbow.png",
        "script_tag": "MED_1M_j",
    },
    {
        "label": "BIG 1B, option j → default_rainbow.png",
        "job_name": "BIGdef_j",
        "workdir": "mpBIG_1B",
        "option": "j",
        "dest_png": "default_rainbow.png",
        "script_tag": "BIG_1B_j",
    },
    {
        "label": "MED 1M, option 1D → true1D_rainbow.png",
        "job_name": "MEDtrue_1D",
        "workdir": "mpMED_1M",
        "option": "1D",
        "dest_png": "true1D_rainbow.png",
        "script_tag": "MED_1M_1D",
    },
    {
        "label": "MED 1M, option 3D → eff3D_rainbow.png",
        "job_name": "MEDeff_3D",
        "workdir": "mpMED_1M",
        "option": "3D",
        "dest_png": "eff3D_rainbow.png",
        "script_tag": "MED_1M_3D",
    },
]


def make_slurm_script(job):
    """
    Create a Slurm bash script for a given job definition.
    No leading indentation on any line so #SBATCH works correctly.
    """
    script_name = f"job_{job['script_tag']}.sh"
    script_path = os.path.abspath(script_name)

    lines = [
        "#!/bin/sh",
        f"#SBATCH -p {PARTITION}",
        f"#SBATCH --mem={MEM}",
        f"#SBATCH -t {TIME}",
        f"#SBATCH -n {N_TASKS}",
        f"#SBATCH -J {job['job_name']}",
        f"#SBATCH -o {job['script_tag']}_%j.out",
        f"#SBATCH -e {job['script_tag']}_%j.err",
        "",
        f"(cd {job['workdir']}; python ../{PYTHON_SCRIPT} -t {job['option']} -d {job['workdir']})",
        "",
    ]

    script_text = "\n".join(lines)

    with open(script_path, "w") as f:
        f.write(script_text)

    os.chmod(script_path, 0o750)
    return script_path


def submit_slurm_script(script_path):
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
    except FileNotFoundError:
        print("ERROR: sbatch not found. Must be on cluster.")
        return None, "", ""

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    job_id = match.group(1) if match else None

    return job_id, stdout, stderr


def main():
    for job in jobs:
        print("=" * 60)
        print(f"Preparing job: {job['label']}")

        script_path = make_slurm_script(job)
        print(f"  Script created: {script_path}")

        job_id, stdout, stderr = submit_slurm_script(script_path)

        if job_id:
            print(f"  Submitted as job ID: {job_id}")
        else:
            print("  Could not detect job ID.")

        print("  sbatch stdout:", stdout)
        if stderr:
            print("  sbatch stderr:", stderr)

    print("=" * 60)
    print("All submissions attempted.")


if __name__ == "__main__":
    main()
