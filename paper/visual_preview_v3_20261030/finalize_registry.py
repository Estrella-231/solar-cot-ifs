from pathlib import Path
import json
import re

root = Path(__file__).resolve().parent
paper = root.parent
asset_root = paper / "figures/paper_visual_preview_v3_20261030"
cfg = json.loads((root / "FIGURE_ASSETS.json").read_text(encoding="utf-8"))
paths = {
    "fig:regions": "data_scope/figure.pdf",
    "fig:architecture": "generated/architecture_final.png",
    "fig:framework": "generated/causal_workflow_final.png",
    "fig:events": "real_case/fig_cloud_source_mechanism.pdf",
    "fig:replacement": "source_replacement/source_replacement.pdf",
    "fig:probe": "probe/cot_conditional_information.pdf",
    "fig:three_station_event": "real_case/fig_three_station_source_response.pdf",
}
for label, relative in paths.items():
    source = asset_root / relative
    assert source.exists(), source
    cfg["figures"][label]["source"] = source.relative_to(paper).as_posix()
    if source.suffix == ".pdf":
        png = source.with_suffix(".png")
        if png.exists():
            cfg["figures"][label]["png_source"] = png.relative_to(paper).as_posix()

def latex(text):
    text = text.replace("%", r"\%").replace("&", r"\&")
    text = text.replace("×", r"$\times$").replace("–", "--").replace("−", "$-$")
    text = text.replace("**", "")
    return text

event = (asset_root / "real_case/CAPTIONS.md").read_text(encoding="utf-8")
for label, section in [("fig:events", event.split("## Supplement:")[0]), ("fig:three_station_event", event.split("## Supplement:")[1])]:
    en = re.search(r"\*\*English\.\*\* ([^\n]+)", section).group(1)
    zh = re.search(r"\*\*中文。\*\* ([^\n]+)", section).group(1)
    cfg["captions_en"][label] = latex(en)
    cfg["captions_zh"][label] = zh
replacement = (asset_root / "source_replacement/CAPTIONS.md").read_text(encoding="utf-8")
cfg["captions_en"]["fig:replacement"] = latex(re.search(r"\*\*EN\.\*\* ([^\n]+)", replacement).group(1))
cfg["captions_zh"]["fig:replacement"] = re.search(r"\*\*中文。\*\* ([^\n]+)", replacement).group(1)
probe = (asset_root / "probe/CAPTION.md").read_text(encoding="utf-8")
cfg["captions_en"]["fig:probe"] = latex(probe.split("## English")[1].split("## 中文")[0].strip())
cfg["captions_zh"]["fig:probe"] = probe.split("## 中文")[1].split("## Interpretation")[0].strip().replace("**", "")
(root / "FIGURE_ASSETS.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Final figure registry ready: " + str(len(cfg["figures"])) + " assets")
