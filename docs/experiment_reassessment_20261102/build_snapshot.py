"""Summarize saved validation scores; no model inference or training."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def read(relative):
    p = ROOT / relative
    return json.loads(p.read_text(encoding="utf-8")), {"path": relative, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

hb_same, s1 = read("refine-logs/hubei_same_time_geometry_20261102/INDEPENDENT_FP64_RESULTS.json")
hb_fore, s2 = read("refine-logs/hubei_threeframe_dydiff_forecast_20261102/INDEPENDENT_FP64_RECHECK.json")
hb_runs, s3 = read("refine-logs/hubei_threeframe_dydiff_forecast_20261102/recovered/evaluation/RESULTS.json")
hn, s4 = read("refine-logs/fulltrain_forecast_cot_20261101/observed_trained_heads_evaluation_v1/RESULTS.json")

def seed_summary(cells, a, ac):
    seeds = sorted({c["seed"] for c in cells})
    result = []
    for seed in seeds:
        selected = [c for c in cells if c["seed"] == seed]
        x = statistics.mean(c[a] for c in selected)
        y = statistics.mean(c[ac] for c in selected)
        result.append({"seed":seed,"no_cot_rmse":x,"cot_rmse":y,"gain_percent":100*(x-y)/x})
    return {"per_seed_region_equal_means":result,
            "no_cot_seed_sd":statistics.stdev(c["no_cot_rmse"] for c in result),
            "cot_seed_sd":statistics.stdev(c["cot_rmse"] for c in result)}

rows = [{"region":"Hubei","mode":"observed_same_time","no_cot_rmse":hb_same["A_rmse"],"cot_rmse":hb_same["AC_rmse"],"gain_percent":hb_same["gain_pct"],"delta_ci95":hb_same["delta_ci95"],"rows":hb_same["n_validation_station_times"],"dates":len(hb_same["target_dates"]),"seed_summary":seed_summary(hb_same["cells"],"A_rmse","AC_rmse")}]
for region, data, cells in [
    ("Hubei",hb_fore["comparisons"],hb_runs["comparisons"]["full"]["cells"]),
    ("Hunan",hn["comparisons"],hn["comparisons"]["full"]["runs"]),
]:
    d=data["full"]; a="A_rmse" if region=="Hubei" else "no_cot_rmse"; ac="AC_rmse" if region=="Hubei" else "cot_rmse"
    rows.append({"region":region,"mode":"DyDiff_forecast","no_cot_rmse":d[a],"cot_rmse":d[ac],"gain_percent":d["reduction_percent"],"delta_ci95":d["delta_ci95"],"rows":d["rows"],"dates":d.get("target_dates",d.get("target_bjt_dates")),"seed_summary":seed_summary(cells,a,ac),"all_forecast_leads":data["lead45_240"]})

for row in rows:
    assert abs(row["gain_percent"] - 100*(row["no_cot_rmse"]-row["cot_rmse"])/row["no_cot_rmse"]) < 1e-9

payload={"scope":"saved validation summaries; no new inference, training or bootstrap; no test/buffer payload read","sources":[s1,s2,s3,s4],"rows":rows,"warning":"Different regions and same-time/forecast cohorts are not causal comparisons of training domain or extrapolation."}
(OUT/"CURRENT_RESULTS.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
with (OUT/"current_results.csv").open("w",encoding="utf-8",newline="") as stream:
    fields=["region","mode","no_cot_rmse","cot_rmse","gain_percent","rows","dates"]
    writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
for r in rows:
    print(r["region"],r["mode"],round(r["no_cot_rmse"],3),round(r["cot_rmse"],3),round(r["gain_percent"],3),"seed_SD",round(r["seed_summary"]["no_cot_seed_sd"],3),round(r["seed_summary"]["cot_seed_sd"],3))
