from pathlib import Path
p=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/agri_direct_cot_20261020/EXPERIMENT_TRACKER.md')
marker='## DENSE_REFERENCE_CPU_COMPLETE_20261021'
entry='## DENSE_REFERENCE_CPU_COMPLETE_20261021\n\n- Luna\u5b9e\u65f6\u6838\u9a8c215358.tc6000=C\u3001exit_status=0\uff0c4,025/4,025\u672a\u6765\u9a8c\u8bc1\u65f6\u523b\u79bb\u7ebf\u5c0f\u56fe\u63d0\u53d6\u5b8c\u6210\uff0c\u7ea6394.9\u79d2\u3002complete\u6536\u636e\u5df2\u53d6\u56de\u672c\u5730results/dense_observed_reference_complete_20261021.json\uff1b\u5b8c\u6574x_raw/mask/rows\u6570\u7ec4\u6b64\u65f6\u4ecd\u5728\u670d\u52a1\u5668\uff0c\u672a\u58f0\u79f0\u672c\u5730\u5df2\u6709\u5b8c\u6574\u955c\u50cf\u3002\u672a\u8bad\u7ec3\u3001\u672a\u8bfbGHI/CPP\u6216\u65e7test/buffer\uff1b\u672a\u6765\u89c2\u6d4b\u4ec5\u79bb\u7ebf\u53c2\u8003\u3002\u4e0b\u4e00\u6b65\u4e0b\u8f7d\u5e76\u9010\u6587\u4ef6SHA\u6838\u9a8c\uff0c\u518d\u4ee5\u51bb\u7ed3R\u751f\u6210\u5bc6\u96c6\u53c2\u8003\u3001\u590d\u7528\u5df2\u51bb\u7ed3\u4e8b\u4ef6\u8bca\u65ad\u89c4\u5219\uff0c\u4e0d\u80fd\u66ff\u6362CPP\u53c2\u8003\u6210\u7ee9\u6216\u5f53\u4f5cGHI\u6536\u76ca\u3002\n'
if marker not in p.read_text(encoding="utf-8"):
 with p.open("a",encoding="utf-8") as f:f.write("\n"+entry)
print("PASS_CPU_COMPLETION_TRACKER_APPEND")
