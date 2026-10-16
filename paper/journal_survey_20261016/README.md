# COT paper journal survey — 2026-10-16

Open `index.html` for the offline Chinese report: 22 candidate journals, field/tier/CAS filters, per-journal positioning, source links, and submission strategy. This is a broad candidate review, not an exhaustive global journal census.

- `build_report.py`: stdlib-only report/data generator. Run with Python 3.
- `journals.json`: source-linked journal records.
- `check_layout.cjs`: local Playwright/Edge rendering and filter check (runtime path is machine-specific).
- `layout_check.json`: observed UI checks; 22 entries, 5 priority candidates, unique-title search, 7 CAS-1 candidates, no mobile viewport overflow or JavaScript errors.
- `preview_desktop.png`, `preview_mobile.png`: rendered previews.

CAS values use explicitly labelled public secondary reports of the 2025 upgraded major-category partition. They were not authenticated against the licensed official database. JCR values retain category/year caveats; SJR, CiteScore, JCI and the 2026 Xinrui partition are not substituted. Unresolved metadata is marked pending. Publisher access failures are disclosed in the HTML. All reputation/fit assessments are editorial judgments, not acceptance forecasts or institutional recognition guarantees.

The primary sources include ISES, IEEE GRSS, MDPI, RMetS, Elsevier's journal store and published articles in Solar Energy, Applied Energy and AMT. Ranking sources are linked at each entry. No manuscript body or experimental assets were changed; no test data was opened.
