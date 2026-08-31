import subprocess
import time
import re
import sys

# ================= CONFIGURATION =================
PARAM_FILE = "dgyre_new_3rougher_1cd_1wind_ny1024"
SUBMIT_SCRIPT = "Run.sh"  # Your qsub/sbatch batch script
CHECK_INTERVAL = 60          # Seconds between status checks
INITIAL_JOB_ID = "123456"    # Replace with your currently running Job ID
MAX_TMAX = 220752000       # Set your target max tmax (e.g., 10 years = 10 * 31536000)
SECONDS_PER_YEAR = 31536000.0
# =================================================

def check_job_status(job_id):
    """Query qstat to check if the job is queued, running, or completed."""
    try:
        output = subprocess.check_output(["qstat", str(job_id)], stderr=subprocess.STDOUT, text=True)
        if str(job_id) in output:
            if " R " in output or " RUN " in output:
                return "RUNNING"
            elif " Q " in output or " PEND " in output:
                return "QUEUED"
            return "IN_QUEUE"
    except subprocess.CalledProcessError:
        # Job missing from qstat output means it finished
        return "FINISHED"
    return "FINISHED"

def update_parameters(filename):
    """Updates restart=1, doubles startIdx, and adds 1 year to tmax."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    updated_tmax = None

    for line in lines:
        parts = line.strip().split()
        if not parts:
            new_lines.append(line)
            continue

        key = parts[0]
        if key == "restart":
            new_lines.append("restart 1\n")
        elif key == "startIdx":
            val = int(float(parts[1]))
            new_lines.append(f"startIdx {val * 2}\n")
        elif key == "tmax":
            current_tmax = float(parts[1])
            updated_tmax = current_tmax + SECONDS_PER_YEAR
            new_lines.append(f"tmax {updated_tmax:.10f}\n")
        else:
            new_lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(new_lines)

    return updated_tmax

def submit_new_job():
    """Submits the job via qsub and returns the new Job ID."""
    # Adjust command to 'sbatch' or 'qsub' based on your cluster syntax
    result = subprocess.check_output(["sh", SUBMIT_SCRIPT], text=True)
    match = re.search(r'\d+', result)
    if match:
        return match.group(0)
    else:
        raise RuntimeError(f"Could not parse new Job ID from output: {result}")

def main():
    current_job_id = INITIAL_JOB_ID
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Monitoring started for initial Job ID: {current_job_id}")

    while True:
        status = check_job_status(current_job_id)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] Job {current_job_id} Status: {status}")

        if status == "FINISHED":
            print(f"[{timestamp}] Job {current_job_id} completed. Updating parameter file '{PARAM_FILE}'...")
            
            new_tmax = update_parameters(PARAM_FILE)
            print(f"[{timestamp}] Updated parameter file: restart=1, startIdx doubled, new tmax={new_tmax}")

            if new_tmax >= MAX_TMAX:
                print(f"[{timestamp}] Maximum target tmax ({MAX_TMAX}) reached. Stopping loop.")
                break

            print(f"[{timestamp}] Submitting next iteration...")
            current_job_id = submit_new_job()
            print(f"[{timestamp}] New Job submitted successfully with Job ID: {current_job_id}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
