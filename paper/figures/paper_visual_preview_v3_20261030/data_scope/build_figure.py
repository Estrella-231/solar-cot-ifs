"""Truthful station-coordinate locator and deliberately unfilled date coverage.

No observation, test, buffer, satellite, or CPP numerical payload is read.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, ConnectionPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "scripts/add_hubei_interval_clear_sky.py",
    ROOT / "scripts/attach_hubei_daylight_mask.py",
    ROOT / "docs/COT_REAL_FORECAST_PAIR_INVENTORY_20260908.md",
    ROOT / "docs/ACTUAL_DATA_INVENTORY_HUNAN_HUBEI_20261016.md",
]
POINTS = [
    {"station":"Niushou", "region":"Hubei", "latitude":32.0800, "longitude":112.0400},
    {"station":"Wangzhai", "region":"Hubei", "latitude":32.0400, "longitude":112.1200},
    {"station":"Mizhuang", "region":"Hubei", "latitude":32.1200, "longitude":112.2000},
    {"station":"Sili", "region":"Hunan", "latitude":25.9478611111, "longitude":112.347916667},
    {"station":"Zhujia", "region":"Hunan", "latitude":29.566666667, "longitude":112.45},
]
BLUE = "#34789C"
TEAL = "#168C80"
INK = "#202C35"
MUTED = "#657480"
GRID = "#E6EBEF"

with plt.rc_context({
    "font.family":"serif", "font.serif":["Times New Roman", "DejaVu Serif"], "font.size":8.0,
    "axes.titlesize":9, "axes.labelsize":8,
    "xtick.labelsize":7, "ytick.labelsize":7,
    "axes.linewidth":0.7, "axes.edgecolor":"#B3BFC8",
    "text.color":INK, "axes.labelcolor":INK,
    "xtick.color":MUTED, "ytick.color":MUTED,
    "pdf.fonttype":42, "ps.fonttype":42,
}):
    fig = plt.figure(figsize=(177/25.4, 122/25.4), facecolor="white")
    ax = fig.add_axes([.086,.15,.375,.75])
    detail = fig.add_axes([.595,.625,.365,.275])
    support = fig.add_axes([.545,.125,.425,.39])
    ax.set_title("(a) Study locations", loc="left", fontweight="bold", pad=12)
    ax.set(xlim=(109.8,115.0), ylim=(25.15,33.05), xlabel="Longitude", ylabel="Latitude")
    # Geographic coordinate plane, no fabricated administrative boundaries.
    ax.set_aspect(1/0.866, adjustable="box")
    ax.set_xticks([110,112,114]); ax.set_yticks([26,28,30,32])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:.0f}°E"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y,p:f"{y:.0f}°N"))
    ax.grid(color=GRID, linewidth=.6, zorder=0)
    for p in POINTS:
        hb = p["region"] == "Hubei"
        ax.scatter(p["longitude"],p["latitude"],s=20 if hb else 42,
                   marker="o" if hb else "^",c=BLUE if hb else TEAL,
                   edgecolors="white",linewidths=.7,zorder=5)
    ax.annotate("Hubei\n3 stations",(112.12,32.08),(110.05,32.63),
                ha="left",va="center",fontsize=8.4,color=BLUE,fontweight="bold",
                arrowprops={"arrowstyle":"-","color":BLUE,"lw":.8})
    ax.annotate("Zhujia",(112.45,29.566666667),(113.02,29.75),
                color=TEAL,fontweight="bold",fontsize=8.2,
                arrowprops={"arrowstyle":"-","color":TEAL,"lw":.65})
    ax.annotate("Sili",(112.347916667,25.9478611111),(113.02,26.25),
                color=TEAL,fontweight="bold",fontsize=8.2,
                arrowprops={"arrowstyle":"-","color":TEAL,"lw":.65})
    ax.text(110.1,27.8,"Hunan\n2 stations",color=TEAL,fontsize=8.4,fontweight="bold")
    ax.add_patch(Rectangle((111.82,31.86),.6,.45,fill=False,lw=.85,edgecolor=BLUE))
    ax.text(.0,-.17,"Coordinate locator; no province boundaries shown.",transform=ax.transAxes,
            fontsize=6.5,color=MUTED)

    detail.set_title("(b) Hubei station detail",loc="left",fontweight="bold",pad=12)
    detail.set(xlim=(111.99,112.25),ylim=(32.005,32.155))
    detail.set_aspect(1/0.848,adjustable="box")
    detail.set_xticks([112.0,112.1,112.2]); detail.set_yticks([32.04,32.08,32.12])
    detail.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f"{x:.1f}°E"))
    detail.yaxis.set_major_formatter(FuncFormatter(lambda y,p:f"{y:.2f}°N"))
    detail.grid(color=GRID,linewidth=.6,zorder=0)
    offsets={"Niushou":(0,9),"Wangzhai":(0,-15),"Mizhuang":(0,9)}
    for p in POINTS[:3]:
        detail.scatter(p["longitude"],p["latitude"],s=30,c=BLUE,edgecolors="white",linewidths=.7,zorder=5)
        detail.annotate(p["station"],(p["longitude"],p["latitude"]),xytext=offsets[p["station"]],
                        textcoords="offset points",ha="center",color=BLUE,fontsize=7.3,fontweight="bold")
    # Detail connector is a schematic linkage, not a measured geographic boundary.
    fig.add_artist(ConnectionPatch(xyA=(112.42,32.31),coordsA=ax.transData,
                    xyB=(0,1),coordsB=detail.transAxes,arrowstyle="-",lw=.65,
                    color="#A7B5C1",linestyle=(0,(3,3)),zorder=1))

    support.set_axis_off(); support.set_xlim(0,1);support.set_ylim(0,1)
    support.text(0,1.04,"(c) Observation support",fontsize=9,fontweight="bold",va="bottom")
    items=[("Satellite","FY-4B AGRI · 13 channels"),
           ("Ground","GHI · 15-min interval means"),
           ("Cloud reference","CPP cloud optical thickness")]
    for i,(label,desc) in enumerate(items):
        y=.88-i*.17
        support.text(.0,y,label,fontsize=7.5,color=MUTED,va="center")
        support.text(.37,y,desc,fontsize=7.5,va="center")
        support.plot([0,1],[y-.075,y-.075],color=GRID,lw=.65)
    support.add_patch(FancyBboxPatch((0,.035),1,.30,boxstyle="round,pad=.012,rounding_size=.015",
        facecolor="#F7F9FA",edgecolor="#B9C5CE",linewidth=.8,linestyle=(0,(3,2))))
    support.text(.04,.258,"FINAL DAILY COVERAGE",fontsize=7.2,fontweight="bold",color=MUTED)
    support.text(.04,.16,"Reserved for verified train / validation dates",fontsize=7.1,color=MUTED)
    support.text(.04,.08,"Per-day availability and sample counts pending",fontsize=6.8,color=MUTED)
    for ext in ("png","pdf","svg"):
        fig.savefig(OUT/f"figure.{ext}",dpi=450,facecolor="white")
    plt.close(fig)

provenance={
    "purpose":"177-mm provisional manuscript figure with verified station-coordinate constants and explicitly pending final coverage",
    "regions_independently_trained":True,
    "map_type":"Longitude/latitude coordinate locator, equirectangular local aspect correction; no basemap or administrative boundaries",
    "station_coordinates":POINTS,
    "coordinate_provenance_note":"Hubei constants agree between both active label-geometry scripts. Hunan constants agree with the archived station-patch inventory and existing Fig1. This is a configuration locator, not a new geolocation survey.",
    "date_coverage_status":"pending; final corrected model sample manifests were not bound in this figure task",
    "date_coverage_policy":"No first-last interval is rendered as continuous availability. Historical inventory endpoint dates and sample counts are intentionally omitted.",
    "numerical_payload_read":False,
    "test_or_buffer_payload_read":False,
    "sources":[{"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in SOURCES],
    "physical_dimensions_mm":[177,122],
    "raster_dpi":450,
    "status":"layout preview; final calendar remains a placeholder"
}
(OUT/"sources.json").write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding="utf-8")
(OUT/"caption.md").write_text("""# Figure 1 caption / 图 1 图注

**English.** Study stations and observation sources. (a) Configured station coordinates for the three Hubei stations and the two Hunan stations. (b) Enlarged view of the Hubei stations. Panels (a–b) are coordinate locators without administrative boundaries. (c) Satellite, ground irradiance, and cloud-reference observations used by the regional experiments. The reserved coverage panel will be populated from the final verified daily sample manifests; it does not assert continuous data availability. The two regions are trained independently.

**中文。** 研究站点与观测来源。（a）湖北三站与湖南两站的配置经纬度；（b）湖北三站的放大位置图。a–b 为经纬度定位图，不包含行政边界。（c）区域实验使用的卫星、地面辐照度和云参考观测。日期覆盖区域暂留位，后续依据最终核验的逐日样本清单补齐；当前图不表示数据连续可用。两个区域独立训练。

**Preview status / 预览说明。** Verified coordinates and source types; final split calendar and sample counts are not yet inserted. No numerical observation or test/buffer payload was opened.
""",encoding="utf-8")
print(OUT/"figure.png")
