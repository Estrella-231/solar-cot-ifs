"""Read workbook properties and header rows only; never decode radiation data cells."""
import argparse
import json
from pathlib import Path

from openpyxl import load_workbook


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workbooks", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    results = []
    for path in a.workbooks:
        wb = load_workbook(path, read_only=True, data_only=False)
        props = wb.properties
        item = {
            "file": str(path),
            "properties": {k: getattr(props, k, None) for k in ("title", "subject", "creator", "keywords", "description", "category", "identifier", "created", "modified")},
            "sheets": [],
            "scope": "workbook properties and first row labels only; no observation values read",
        }
        for ws in wb.worksheets:
            headers = [ws.cell(1, c).value for c in range(1, min(ws.max_column, 80) + 1)]
            comments = {str(c): getattr(ws.cell(1, c), "comment", None).text
                        for c in range(1, min(ws.max_column, 80) + 1)
                        if getattr(ws.cell(1, c), "comment", None)}
            item["sheets"].append({"name": ws.title, "max_column": ws.max_column, "header_row_1": headers, "header_comments": comments})
        wb.close()
        results.append(item)
    out = {"state": "HEADER_ONLY_METADATA_AUDIT", "observation_values_decoded": 0, "workbooks": results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
