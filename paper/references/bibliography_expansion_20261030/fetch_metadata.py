"""Retrieve publisher-deposited Crossref metadata; no scientific claim inference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import requests
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
DOIS = {
    'huang2019': '10.1016/j.rse.2019.111371',
    'yang2020': '10.1016/j.solener.2020.04.019',
    'chen2023': '10.1016/j.apenergy.2023.121979',
    'chen2024': '10.1016/j.enconman.2024.118598',
    'liao2024': '10.1016/j.renene.2024.121055',
    'pap2024': '10.5194/amt-17-1851-2024',
    'zhang2026': '10.1029/2025GL119553',
    'song2025': '10.1029/2025GL116237',
    'schuurman2025': '10.1080/01431161.2025.2471596',
    'nakajima1990': '10.1175/1520-0469(1990)047<1878:DOTOTA>2.0.CO;2',
    'xia2025': '10.3390/rs17050737',
    'miller2018': '10.1016/j.solener.2017.11.049',
    'nielsen2021': '10.1016/j.solener.2021.09.073',
}

def fetch(pair):
    key, doi = pair
    r = requests.get('https://api.crossref.org/works/' + quote(doi, safe=''), timeout=40)
    r.raise_for_status()
    m = r.json()['message']
    return key, {'doi': doi, 'title': m['title'][0], 'authors': m.get('author', []),
                 'journal': m.get('container-title', []), 'volume': m.get('volume'),
                 'issue': m.get('issue'), 'pages': m.get('page'),
                 'article_number': m.get('article-number'), 'issued': m.get('issued'),
                 'published': m.get('published'), 'URL': m.get('URL'),
                 'source': 'Crossref publisher-deposited metadata; fetched 2026-10-30'}

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        data = dict(pool.map(fetch, DOIS.items()))
    (HERE / 'ROOT_METADATA.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    for k, v in data.items():
        print(k, v['title'], v['volume'], v['pages'] or v['article_number'], v['issued'])
