"""Metadata-only XLSX structure probe; never prints radiation cell values."""
import argparse
import zipfile
import xml.etree.ElementTree as ET

NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    args = ap.parse_args()
    with zipfile.ZipFile(args.xlsx) as zf:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {x.attrib["Id"]: x.attrib["Target"] for x in rels}
        for sh in wb.find("m:sheets", NS):
            target = relmap[sh.attrib["{" + NS["r"] + "}id"]]
            path = target.lstrip("/") if target.startswith("/") else "xl/" + target
            if not path.startswith("xl/"):
                path = "xl/" + path
            print("SHEET", sh.attrib["name"], path)
            with zf.open(path) as f:
                for _, row in ET.iterparse(f, events=("end",)):
                    if row.tag.endswith("}row"):
                        rn = int(row.attrib.get("r", "0"))
                        if rn <= 4:
                            cells = []
                            for c in row:
                                if c.tag.endswith("}c"):
                                    ref = c.attrib.get("r", "")
                                    if ref.rstrip("0123456789") in {"F", "Q"}:
                                        v = c.find("m:v", NS)
                                        inline = c.find("m:is", NS)
                                        # For F only report cell type/style, never its value.
                                        if ref.startswith("F"):
                                            cells.append((ref, "type=" + c.attrib.get("t", "n"), "style=" + c.attrib.get("s", "")))
                                        elif v is not None:
                                            cells.append((ref, "time_type=" + c.attrib.get("t", "n"), "time_value=" + v.text))
                                        elif inline is not None:
                                            txt = "".join(t.text or "" for t in inline.iter() if t.tag.endswith("}t"))
                                            cells.append((ref, "time_inline", "time_text=" + txt))
                            if cells:
                                print(" ROW", rn, cells)
                        row.clear()
                        if rn >= 4:
                            break


if __name__ == "__main__":
    main()
