"""Build a Crossref-verified literature catalogue and duplicate-safe RIS batch.

Reads Zotero through its local read-only API; does not write to Zotero.
The RIS batch is separately imported using the Zotero connector helper.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import requests

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
SEED = PAPER / "bibliography_expansion_20261030" / "MANUSCRIPT_REFERENCES.json"
AGRI_FORECAST = "卫星辐照度/云场预报"
CLOUD = "云物理/COT与辐射估计"
EXTRA = {
    AGRI_FORECAST: [
        "10.3390/rs17162861", "10.1016/j.apenergy.2025.127284",
        "10.3390/s20092606", "10.1002/solr.202400475",
        "10.1016/j.apenergy.2024.124186", "10.1016/j.solener.2021.12.066",
        "10.1016/j.solener.2023.111820", "10.1016/j.solener.2023.01.037",
        "10.1016/j.apenergy.2021.117834", "10.1002/met.2051",
        "10.1016/j.eswa.2023.122925", "10.1016/j.solener.2024.112319",
        "10.1038/s41467-023-44666-1", "10.1016/j.apenergy.2024.124387",
        "10.1016/j.engappai.2022.105493", "10.1016/j.rser.2022.112680",
    ],
    CLOUD: [
        "10.5194/amt-17-6003-2024", "10.5194/amt-17-6697-2024",
        "10.5194/acp-20-1131-2020", "10.5194/acp-25-16347-2025",
        "10.3390/rs11141703", "10.3390/rs13081418", "10.3390/rs10030411",
        "10.3390/rs15041141", "10.1175/JAMC-D-20-0205.1",
        "10.1016/j.atmosenv.2024.121023", "10.1029/2005JD006668",
        "10.1029/2008JD010579", "10.1029/2005JD006792", "10.1029/2018JD028905",
        "10.34133/remotesensing.1044", "10.3390/rs15235454", "10.3390/s21051859",
        "10.1029/2020JD032683", "10.1029/2024JD041032", "10.5194/amt-19-899-2026",
        "10.5194/amt-18-6681-2025", "10.5194/amt-17-261-2024",
    ],
}
PDF_OVERRIDES = {
    "10.1016/j.apenergy.2023.121979": "https://www.li-realab.info/publication/solar-forecast/chen-2023-resourcing/chen-2023-resourcing.pdf",
    "10.1016/j.enconman.2024.118598": "https://www.li-realab.info/publication/solar-forecast/chen-2024-dlchain/chen-2024-DLchain.pdf",
    "10.1002/solr.202400475": "https://publica-rest.fraunhofer.de/server/api/core/bitstreams/93fbd665-f6ed-4edb-802b-ae2d0bf1e966/content",
    "10.1029/2024jd041032": "https://arxiv.org/pdf/2405.19336",
    "10.1080/01431161.2025.2471596": "https://arxiv.org/pdf/2409.16316",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def zotero_top() -> list[dict]:
    items = []
    for start in range(0, 10000, 100):
        url = f"http://127.0.0.1:23119/api/users/0/items/top?limit=100&start={start}"
        with urlopen(url, timeout=15) as r:
            batch = json.load(r)
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


def get_json(url: str) -> dict:
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": "Irradiance-literature-research/1.0"}, timeout=25)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(0.7 * (attempt + 1))
    raise RuntimeError("unreachable")


def crossref(doi: str) -> dict:
    x = get_json("https://api.crossref.org/works/" + quote(doi, safe=""))["message"]
    dates = x.get("published", x.get("issued", {})).get("date-parts", [[None]])[0]
    authors = [" ".join(filter(None, (a.get("given", ""), a.get("family", "")))) for a in x.get("author", [])]
    authors_ris = [", ".join(filter(None, (a.get("family", ""), a.get("given", ""))))
                   or a.get("name", "") for a in x.get("author", [])]
    return {"doi": x.get("DOI", doi), "title": (x.get("title") or [""])[0],
            "year": dates[0], "authors": authors, "authors_ris": authors_ris,
            "venue": (x.get("container-title") or [""])[0],
            "type": x.get("type", ""), "url": x.get("URL", "https://doi.org/" + doi),
            "volume": x.get("volume", ""), "issue": x.get("issue", ""),
            "pages": x.get("page", x.get("article-number", "")),
            "publisher": x.get("publisher", "")}


def openalex(doi: str) -> dict:
    try:
        x = get_json("https://api.openalex.org/works/https://doi.org/" + quote(doi, safe="/"))
        loc = x.get("best_oa_location") or {}
        return {"pdf_url": loc.get("pdf_url") or "", "oa_version": loc.get("version") or "",
                "oa_landing": loc.get("landing_page_url") or ""}
    except Exception as e:
        return {"pdf_url": "", "oa_version": "", "oa_landing": "", "oa_error": str(e)[:180]}


def ris_row(x: dict) -> str:
    kinds = {"journal-article": "JOUR", "proceedings-article": "CPAPER", "book-chapter": "CHAP"}
    rows = [f"TY  - {kinds.get(x['type'], 'GEN')}", f"TI  - {x['title']}"]
    rows.extend(f"AU  - {a}" for a in x["authors_ris"] if a)
    for k, tag in [("year", "PY"), ("venue", "JO"), ("volume", "VL"), ("issue", "IS"),
                   ("pages", "SP"), ("doi", "DO"), ("url", "UR")]:
        if x.get(k):
            rows.append(f"{tag}  - {x[k]}")
    rows.extend(["ER  -", ""])
    return "\n".join(rows)


def main() -> None:
    original = json.loads(SEED.read_text(encoding="utf-8"))
    queue: dict[str, dict] = {}
    for x in original:
        if x.get("doi"):
            queue[x["doi"].lower()] = {"category": "已有论文引用", "expected_title": x["title"]}
    for category, dois in EXTRA.items():
        for doi in dois:
            queue.setdefault(doi.lower(), {"category": category, "expected_title": ""})
    library = zotero_top()
    doi_map = {x["data"].get("DOI", "").lower(): x for x in library if x["data"].get("DOI")}
    title_map = {norm(x["data"].get("title", "")): x for x in library if x["data"].get("title")}
    records, failures = [], []
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(crossref, doi): doi for doi in queue}
        for f in cf.as_completed(futures):
            doi = futures[f]
            try:
                item = f.result()
                item.update(queue[doi])
                item["title_match"] = not item["expected_title"] or norm(item["expected_title"]) == norm(item["title"])
                found = doi_map.get(doi) or title_map.get(norm(item["title"]))
                item["zotero_key_before"] = found["key"] if found else ""
                records.append(item)
            except Exception as e:
                failures.append({"doi": doi, "error": str(e)[:300]})
    with cf.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(openalex, x["doi"]): x for x in records}
        for f in cf.as_completed(futures):
            item = futures[f]
            item.update(f.result())
            override = PDF_OVERRIDES.get(item["doi"].lower())
            if override:
                item["pdf_url"] = override
                item["pdf_source"] = "author/official open copy"
            elif item["pdf_url"]:
                item["pdf_source"] = "OpenAlex best OA location; availability unverified"
    records.sort(key=lambda x: (x["category"], x["year"] or 0, x["title"]))
    pending = [x for x in records if not x["zotero_key_before"] and x["title_match"]]
    out = {"scope": "FY-4B/AGRI, COT, satellite irradiance estimation and forecasting, evaluation",
           "read_depth": "Crossref bibliographic metadata; OpenAlex OA metadata. No full-text claim.",
           "zotero_before_count": len(library), "candidate_count": len(queue),
           "verified_metadata_count": len(records), "existing_library_count": len(records)-len(pending),
           "new_import_count": len(pending), "failures": failures, "records": records}
    (HERE / "catalog.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "new_records.ris").write_text("\n".join(ris_row(x) for x in pending), encoding="utf-8")
    with (HERE / "catalog.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["category", "year", "title", "authors", "venue", "doi", "url", "pdf_url", "oa_version", "zotero_key_before", "title_match"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for x in records:
            row = dict(x)
            row["authors"] = "; ".join(x["authors"])
            w.writerow(row)
    print(json.dumps({k: out[k] for k in ["candidate_count", "verified_metadata_count", "existing_library_count", "new_import_count", "failures"]}, ensure_ascii=False))
    print("OA PDF candidates:", sum(bool(x["pdf_url"]) for x in records))
    print("Seed title mismatches:", [(x["doi"], x["expected_title"], x["title"]) for x in records if not x["title_match"]])


if __name__ == "__main__":
    main()
