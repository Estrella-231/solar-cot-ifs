"""Run the portable package's read-only FY4B 105E preflight."""
import json

from Preflight import run_preflight

print(json.dumps({"satellite": "FY4B_105E", "problems": run_preflight(["FY4B_105E"])}, indent=2))
