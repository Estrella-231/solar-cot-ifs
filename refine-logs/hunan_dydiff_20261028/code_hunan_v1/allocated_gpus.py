"""Print only PBS-assigned physical GPU IDs; fail on unexpected occupants."""
import os
import re
import subprocess

expected = int(os.environ["EXPECTED_GPUS"])
raw = subprocess.check_output([
    "/opt/gridview/pbs/dispatcher/bin/qstat", "-f", os.environ["PBS_JOBID"]], text=True)
match = re.search(r"^\s*exec_gpus\s*=\s*(.*?)(?=^\s*[A-Za-z_][\w.]*\s*=|\Z)",
                  raw, flags=re.MULTILINE | re.DOTALL)
if match is None:
    raise RuntimeError("PBS did not report exec_gpus")
allocation = re.sub(r"\s+", "", match.group(1))
gpus = sorted(set(int(s) for s in re.findall(r"gpu/(\d+)", allocation)))
if len(gpus) != expected:
    raise RuntimeError(f"expected {expected} allocated GPUs, got {gpus}")
for gpu in gpus:
    active = subprocess.check_output([
        "nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,process_name",
        "--format=csv,noheader"], text=True).strip()
    if active:
        raise RuntimeError(f"allocated GPU {gpu} already has compute processes: {active}")
print(",".join(map(str, gpus)))
