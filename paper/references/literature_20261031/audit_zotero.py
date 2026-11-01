"""Read-only verification of catalogue records, collection membership, and PDF attachments."""
import concurrent.futures as cf
import json
from pathlib import Path
from urllib.request import urlopen

from build_catalog import norm, zotero_top

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:23119/api/users/0"
COLLECTION = "2QTWAQ7P"


def get(path):
    with urlopen(BASE + path, timeout=20) as response:
        return json.load(response)


def children(row):
    if not row["zotero_key"]:
        return row
    try:
        items = get(f"/items/{row['zotero_key']}/children?limit=100")
        row["pdf_attachments"] = [x["key"] for x in items if x["data"].get("contentType") == "application/pdf"]
    except Exception as exc:
        row["attachment_check_error"] = str(exc)[:200]
    return row


def main():
    catalog = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    library = zotero_top()
    collection = []
    for start in range(0, 1000, 100):
        batch = get(f"/collections/{COLLECTION}/items/top?limit=100&start={start}")
        collection.extend(batch)
        if len(batch) < 100:
            break
    keys = {x["key"] for x in collection}
    # Multiple Zotero entries can share a DOI. Prefer the copy in the requested
    # collection, rather than whichever duplicate happens to be returned last.
    by_doi, by_title = {}, {}
    for item in sorted(library, key=lambda item: item["key"] not in keys):
        data = item["data"]
        if data.get("DOI"):
            by_doi.setdefault(data["DOI"].lower(), item)
        if data.get("title"):
            by_title.setdefault(norm(data["title"]), item)
    rows = []
    for x in catalog["records"]:
        found = by_doi.get(x["doi"].lower()) or by_title.get(norm(x["title"]))
        key = found["key"] if found else ""
        rows.append({"doi": x["doi"], "title": x["title"], "zotero_key": key,
                     "in_requested_collection": key in keys, "pdf_attachments": []})
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(children, rows))
    summary = {"candidate_count": len(rows), "in_library": sum(bool(x["zotero_key"]) for x in rows),
               "in_collection": sum(x["in_requested_collection"] for x in rows),
               "with_pdf_attachment": sum(bool(x["pdf_attachments"]) for x in rows),
               "attachment_check_errors": sum(bool(x.get("attachment_check_error")) for x in rows),
               "collection_key": COLLECTION, "rows": rows}
    (HERE / "zotero_check.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
