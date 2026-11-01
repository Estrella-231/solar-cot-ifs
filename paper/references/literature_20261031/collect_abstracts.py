"""Collect source-provided abstracts without labeling metadata as a full reading."""
import concurrent.futures
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

import requests

HERE = Path(__file__).resolve().parent
records = json.loads((HERE / 'catalog.json').read_text(encoding='utf-8'))['records']

def fetch(record):
    doi = record['doi']
    try:
        response = requests.get('https://api.crossref.org/works/' + quote(doi, safe='/'),
                                headers={'User-Agent': 'AcademicLiteratureReview/1.0'}, timeout=20)
        response.raise_for_status()
        message = response.json()['message']
        abstract = html.unescape(re.sub(r'<[^>]+>', ' ', message.get('abstract', '')))
        abstract = re.sub(r'\s+', ' ', abstract).strip()
        return {'doi': doi, 'title': message.get('title', [''])[0],
                'abstract': abstract, 'source': 'https://api.crossref.org/works/' + quote(doi, safe='/'),
                'status': 'abstract' if abstract else 'not_supplied'}
    except Exception as exc:
        return {'doi': doi, 'status': 'error', 'error': str(exc)}

with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    result = list(pool.map(fetch, records))
(HERE / 'crossref_abstracts.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print({s: sum(row['status'] == s for row in result) for s in ('abstract', 'not_supplied', 'error')})
