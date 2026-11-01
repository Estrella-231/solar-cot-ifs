"""Generate an honest status index from verified bibliography and PDF inventories."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
VISUAL = {
    "10.1016/j.apenergy.2023.121979", "10.1016/j.enconman.2024.118598",
    "10.5194/amt-17-1851-2024", "10.1029/2025gl119553",
    "10.1016/j.solener.2017.11.049", "10.1002/solr.202400475",
    "10.5194/amt-17-6003-2024", "10.1029/2024jd041032",
    "10.1038/s41467-023-44666-1", "10.1016/j.engappai.2022.105493",
    "10.5194/acp-25-16347-2025", "10.5194/amt-18-3917-2025",
    "10.1002/met.2051", "10.1080/01431161.2025.2471596",
    "10.3390/rs11141703", "10.5194/acp-20-1131-2020",
    "10.5194/amt-10-409-2017", "10.5194/amt-17-261-2024",
    "10.5194/amt-17-6697-2024", "10.5194/amt-18-6681-2025",
    "10.5194/amt-18-7315-2025", "10.5194/amt-19-899-2026",
    "10.3390/s20092606", "10.3390/rs15041141",
    "10.3390/rs17050737", "10.1127/metz/2016/0781",
    "10.3390/rs17162861",
    "10.3390/rs10030411", "10.3390/rs13081418", "10.3390/s21051859",
    "10.1016/j.rse.2019.111371",
    "10.1016/j.apenergy.2024.124387",
    "10.1016/j.apenergy.2024.124186",
    "10.1016/j.solener.2023.111820",
}
MATRIX = {
    "10.3390/rs17162861", "10.1016/j.apenergy.2025.127284",
    "10.1016/j.renene.2024.121055", "10.3390/s20092606",
    "10.1016/j.solener.2023.111820", "10.1016/j.solener.2021.12.066",
    "10.1002/met.2051", "10.1016/j.atmosenv.2024.121023",
    "10.1029/2005jd006668", "10.5194/amt-19-899-2026",
} | VISUAL


def esc(s):
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def main():
    catalog = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    check = json.loads((HERE / "zotero_check.json").read_text(encoding="utf-8"))
    pdfs = json.loads((HERE / "pdf_manifest.json").read_text(encoding="utf-8"))
    by_doi = {x["doi"].lower(): x for x in check["rows"]}
    by_pdf = {x["doi"].lower(): x for x in pdfs}
    records = catalog["records"]
    lines = ["# 卫星云场、COT 与辐照度：文献索引", "",
             "2026-10-31。按 DOI 对 Crossref 题录核验并与本机 Zotero 去重。",
             f"本表 {len(records)} 条；Zotero 中 {check['in_library']} 条，其中所选集合 {check['in_collection']} 条。",
             f"本地可打开 PDF {sum(x['status']=='downloaded' for x in pdfs)} 篇；Zotero 有 PDF 附件的篇数为本次检查快照 {check['with_pdf_attachment']}，批量查找仍可能继续变化。",
             "PDF 与附件分别统计；有 DOI 或摘要不代表读过全文。下表‘图页’只指已查看的指定原图页面，不代表通读全部图页。", "",
             "| 年份 | 文献（DOI） | 期刊 | Zotero 条目 | PDF | 阅读状态 |", "|---|---|---|---|---|---|"]
    for x in records:
        doi=x["doi"].lower(); z=by_doi.get(doi,{})
        p=by_pdf.get(doi,{})
        if p.get("status")=="downloaded":
            pdf="本地可打开"
        elif p.get("status")=="unavailable":
            pdf="开放地址获取失败"
        else:
            pdf="待寻找"
        if z.get("pdf_attachments"):
            pdf += "；Zotero 有附件"
        depth="图页" if doi in VISUAL else ("专题对照" if doi in MATRIX else "题录")
        title=esc(x["title"].replace("<scp>","").replace("</scp>",""))
        lines.append(f"| {x['year']} | [{title}](https://doi.org/{x['doi']}) | {esc(x['venue'])} | {z.get('zotero_key','未找到')} | {pdf} | {depth} |")
    lines.extend(["", "图页笔记：[FIGURE_LANGUAGE_REVIEW.md](FIGURE_LANGUAGE_REVIEW.md)。",
                  "直接相关方法对照：[RESEARCH_MATRIX.md](RESEARCH_MATRIX.md)。",
                  "原有 Zotero 条目的扩展阅读：[ADDITIONAL_LIBRARY_REVIEW.md](ADDITIONAL_LIBRARY_REVIEW.md)。",
                  "SHADECast 的 Zotero 条目 YDYRDZS8 已按正式版修正四位作者、年份 2025 和卷号 377；重复条目未合并。",
                  "程序化原始记录：[catalog.json](catalog.json)、[pdf_manifest.json](pdf_manifest.json)、[zotero_check.json](zotero_check.json)。", ""])
    (HERE / "LITERATURE_INDEX.md").write_text("\n".join(lines),encoding="utf-8")
    print(f"index rows={len(records)} local_pdf={sum(x['status']=='downloaded' for x in pdfs)} zotero_pdf={check['with_pdf_attachment']}")


if __name__ == "__main__":
    main()
