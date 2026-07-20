"""Check internal links in rendered _book HTML pages.

For each given page, every relative <a href> is resolved against _book/ and
verified: the target file must exist and, when a #fragment is present, the
target page must contain that id. Usage:

    python scripts/check_html_links.py [page.html ...]   (default: all pages)
"""
import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "_book"

pages = [ROOT / p for p in sys.argv[1:]] or sorted(ROOT.rglob("*.html"))
bad = 0
for page in pages:
    html = page.read_text(encoding="utf-8", errors="replace")
    for href in re.findall(r'<a[^>]+href="([^"#]*)(#[^"]*)?"', html):
        target, frag = href
        if target.startswith(("http", "mailto:", "javascript:")):
            continue
        dest = (page.parent / target).resolve() if target else page
        if target and not dest.exists():
            print(f"DEAD FILE  {page.relative_to(ROOT)} -> {target}")
            bad += 1
            continue
        if frag and dest.suffix == ".html":
            frag_id = frag[1:]
            dest_html = dest.read_text(encoding="utf-8", errors="replace")
            if f'id="{frag_id}"' not in dest_html and f"id='{frag_id}'" not in dest_html:
                print(f"DEAD FRAG  {page.relative_to(ROOT)} -> {target}{frag}")
                bad += 1
print(f"checked {len(pages)} pages: {bad} dead links")
