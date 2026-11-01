"""Count bibliography entries, not citation occurrences or citation impact."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import sys

import pymupdf
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def lines(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            yield line["bbox"], "".join(span["text"] for span in line["spans"])


def record(path, ident, main_end, reference_start, url, entries):
    doc = pymupdf.open(path)
    return {
        "id": ident,
        "source_path": str(path),
        "source_url": url,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pdf_pages": len(doc),
        "main_text_page_span": [1, main_end],
        "references_page_span": [reference_start, len(doc)],
        "reference_count": len(entries),
        "counts_by_pdf_page": dict(Counter(e["pdf_page"] for e in entries)),
        "entries": entries,
    }


def numbered(path, ident, main_end, reference_start, url):
    doc = pymupdf.open(path)
    found = []
    for page_num in range(reference_start, len(doc) + 1):
        for bbox, text in lines(doc[page_num - 1]):
            match = re.match(r"^\[(\d+)\]\s*(.*)", text)
            if match:
                found.append({
                    "ordinal": int(match[1]), "pdf_page": page_num,
                    "bbox": [round(x, 3) for x in bbox],
                    "first_author_or_organization": match[2].split(",")[0].split(".")[0][:40],
                    "first_line_sha256": hashlib.sha256(text.encode()).hexdigest(),
                })
    found.sort(key=lambda e: e["ordinal"])
    assert [e["ordinal"] for e in found] == list(range(1, len(found) + 1))
    result = record(path, ident, main_end, reference_start, url, found)
    result["method"] = "PDF numbered bibliography starts; every ordinal occurs once and is contiguous from 1."
    return result


reports = []
reports.append(numbered(
    ROOT / "narrative_review_20261015/chen_2023_spectral_irradiance.pdf",
    "chen_2023", 12, 16,
    "https://www.li-realab.info/publication/solar-forecast/chen-2023-resourcing/chen-2023-resourcing.pdf",
))
reports.append(numbered(
    ROOT / "wording_review_20261030/chen_2024_dlchain.pdf",
    "chen_2024", 15, 16,
    "https://www.li-realab.info/publication/solar-forecast/chen-2024-dlchain/chen-2024-DLchain.pdf",
))

amt_path = ROOT / "narrative_review_20261015/papachristopoulou_2024_cloud_aerosol.pdf"
doc = pymupdf.open(amt_path)
found = []
for pn in range(24, 28):
    for bbox, text in lines(doc[pn - 1]):
        x, y = bbox[:2]
        if not 60 < y < 730 or (pn == 24 and x < 200 and y < 360):
            continue
        if min(abs(x - 46.49), abs(x - 307.28)) < .2:
            found.append({"ordinal": len(found) + 1, "pdf_page": pn,
                          "bbox": [round(v, 3) for v in bbox],
                          "first_author_or_organization": text.split(",")[0].split(":")[0],
                          "first_line_sha256": hashlib.sha256(text.encode()).hexdigest()})
url = "https://amt.copernicus.org/articles/17/1851/2024/"
response = requests.get(url, timeout=40)
response.raise_for_status()
response.encoding = "utf-8"
refs = BeautifulSoup(response.text, "html.parser").select("p.ref")
ids = [ref.get("id") for ref in refs]
assert ids == [f"bib1.bib{i}" for i in range(1, len(refs) + 1)]
assert len(found) == len(refs)
for entry, ref in zip(found, refs):
    reftext = ref.get_text(" ", strip=True)
    assert reftext.startswith(entry["first_author_or_organization"])
    entry["html_id"] = ref.get("id")
    entry["html_text_sha256"] = hashlib.sha256(reftext.encode()).hexdigest()
report = record(amt_path, "papachristopoulou_2024", 20, 24, url, found)
report["method"] = "Independent PDF hanging-indent starts and official HTML p.ref elements; contiguous HTML IDs and first-author order matched."
report["html_reference_count"] = len(refs)
report["html_sha256"] = hashlib.sha256(response.content).hexdigest()
reports.append(report)

cloud_path = next(Path("C:/Users/29728/xwechat_files/wxid_s79spdfmjaq912_2146/msg/file/2026-10").glob("*GRL*(1).pdf"))
doc = pymupdf.open(cloud_path)
found = []
for pn in [9, 10, 11]:
    for bbox, text in lines(doc[pn - 1]):
        x, y = bbox[:2]
        if abs(x - 167.24) < .2 and 80 < y < 740 and (pn > 9 or y > 700):
            found.append({"ordinal": len(found) + 1, "pdf_page": pn,
                          "bbox": [round(v, 3) for v in bbox],
                          "first_author": text.split(",")[0],
                          "first_line_sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "year": re.search(r"\((\d{4})\)", text)[1]})
report = record(cloud_path, "zhang_2026_cloud_fno", 9, 9,
                "https://doi.org/10.1029/2025GL119553", found)
report["method"] = "User-provided published PDF; reference start lines at x=167.24 pt; continuation lines indented to x=175.01 pt. Header/footer and left-side acknowledgments excluded; every start has author and parenthesized year."
reports.append(report)

payload = {"counted_at": "2026-10-30", "quantity": "bibliography_entries", "papers": reports}
(OUT / "REFERENCE_COUNTS.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
for r in reports:
    print(r["id"], r["reference_count"], r["counts_by_pdf_page"])
