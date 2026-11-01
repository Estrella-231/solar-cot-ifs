"""Render selected original PDF pages for visual figure review."""
from pathlib import Path
import fitz

HERE = Path(__file__).resolve().parent
TASKS = {
    "10.1002_solr.202400475.pdf": [5, 9, 10, 13],
    "10.5194_amt-17-6003-2024.pdf": [5, 11],
    "10.1029_2024jd041032.pdf": [8, 12, 15],
    "10.1038_s41467-023-44666-1.pdf": [3, 5],
    "10.1016_j.engappai.2022.105493.pdf": [30, 31],
    "10.5194_acp-25-16347-2025.pdf": [8, 13],
    "10.5194_amt-18-3917-2025.pdf": [8, 10],
    "10.1002_met.2051.pdf": [9],
    "10.1080_01431161.2025.2471596.pdf": [8],
    "10.3390_rs11141703.pdf": [9],
    "10.5194_acp-20-1131-2020.pdf": [7],
    "10.5194_amt-10-409-2017.pdf": [9],
    "10.5194_amt-17-261-2024.pdf": [12],
    "10.5194_amt-17-6697-2024.pdf": [4, 8],
    "10.5194_amt-18-6681-2025.pdf": [11],
    "10.5194_amt-18-7315-2025.pdf": [10],
    "10.5194_amt-19-899-2026.pdf": [13],
}
out = HERE / "rendered_pages"
out.mkdir(exist_ok=True)
for filename, pages in TASKS.items():
    with fitz.open(HERE / "pdfs" / filename) as doc:
        for page in pages:
            path = out / f"{Path(filename).stem}_p{page}.png"
            doc[page-1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False).save(path)
            print(path)
