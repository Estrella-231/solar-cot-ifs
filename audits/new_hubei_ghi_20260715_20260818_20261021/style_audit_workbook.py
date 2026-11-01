"""Make a consistently styled delivery copy, preserving all audited cell values."""
import hashlib
import json
from pathlib import Path
import openpyxl
from openpyxl.styles import Font

root=Path(__file__).parent/'results_v1'
source=root/'新观测质量核查.xlsx'
dest=root/'新观测质量核查_交付.xlsx'
wb=openpyxl.load_workbook(source)
before={ws.title:[[c.value for c in row] for row in ws] for ws in wb}
for ws in wb:
    for row in ws:
        for c in row:
            c.font=Font(name='Arial',size=10,bold=(c.row==1 and ws.title!='Read me'),
                        color='FFFFFF' if c.row==1 and ws.title!='Read me' else '000000')
wb.save(dest)
check=openpyxl.load_workbook(dest,read_only=True,data_only=False)
assert {ws.title:[[c.value for c in row] for row in ws] for ws in check}==before
assert all(c.data_type not in ('f','e') for ws in check for row in ws for c in row)
check.close()
(root/'delivery_style_receipt.json').write_text(json.dumps({
    'state':'PASS_VALUES_PRESERVED_STYLE_ONLY_DELIVERY','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'delivery_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'all_cell_values_equal':True,'formula_count':0,'error_cells':0},indent=2)+'\n',encoding='utf-8')
print('PASS: consistent Arial delivery; all cell values preserved, no formulas/error cells.')
