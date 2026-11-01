"""Fetch discoverable public PDFs; verify actual PDF bytes and extract figure-page index."""
import concurrent.futures as cf
import hashlib
import json
import re
from pathlib import Path

import fitz
import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "pdfs"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic literature review; open-access only)"}
FALLBACKS = {
    "10.1002/met.2051": "https://smhi.diva-portal.org/smash/get/diva2%3A1658507/FULLTEXT01.pdf",
    "10.1016/j.solener.2017.11.049": "https://repository.library.noaa.gov/view/noaa/49543/noaa_49543_DS1.pdf",
    "10.3390/s20092606": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7248847/pdf/sensors-20-02606.pdf",
    "10.3390/rs11141703": "https://mdpi-res.com/remotesensing/remotesensing-11-01703/article_deploy/remotesensing-11-01703-v2.pdf?attachment=1",
}


def filename(doi):
    return re.sub(r"[^A-Za-z0-9.-]+", "_", doi) + ".pdf"


def fetch(item):
    doi, url = item["doi"], item["pdf_url"]
    dest = OUT / filename(doi)
    result = {"doi": doi, "title": item["title"], "url": url}
    try:
        if not dest.exists():
            choices = [url]
            if FALLBACKS.get(doi.lower()):
                choices.insert(0, FALLBACKS[doi.lower()])
            errors = []
            raw = None
            for candidate in choices:
                try:
                    r = requests.get(candidate, headers=HEADERS, timeout=45)
                    r.raise_for_status()
                    if len(r.content) < 10000 or not r.content[:1024].lstrip().startswith(b"%PDF-"):
                        raise ValueError(f"not a PDF response ({len(r.content)} bytes)")
                    raw = r.content
                    result["actual_url"] = candidate
                    break
                except Exception as e:
                    errors.append(str(e)[:150])
            if raw is None:
                raise ValueError("; ".join(errors))
            with fitz.open(stream=raw, filetype="pdf") as doc:
                if not doc.page_count:
                    raise ValueError("PDF has no pages")
            dest.write_bytes(raw)
        raw = dest.read_bytes()
        with fitz.open(dest) as doc:
            index = []
            for i, page in enumerate(doc):
                text = page.get_text("text")
                labels = sorted(set(re.findall(r"(?im)\b(?:Fig(?:ure)?\.?|图)\s*(\d{1,2})\b", text)))
                if labels:
                    index.append({"pdf_page": i + 1, "figure_mentions": labels})
            result.update(status="downloaded", path=str(dest), pages=doc.page_count,
                          sha256=hashlib.sha256(raw).hexdigest(), figure_index=index,
                          first_page_text=doc[0].get_text("text")[:3500])
    except Exception as e:
        result.update(status="unavailable", error=str(e)[:300])
    return result


def main():
    OUT.mkdir(exist_ok=True)
    cat = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    candidates = [x for x in cat["records"] if x.get("pdf_url")]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, candidates))
    results.sort(key=lambda x: x["doi"])
    (HERE / "pdf_manifest.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(x["status"] == "downloaded" for x in results)
    print(json.dumps({"pdf_candidates": len(results), "valid_downloads": ok,
                      "unavailable": len(results)-ok}, ensure_ascii=False))


if __name__ == "__main__":
    main()
