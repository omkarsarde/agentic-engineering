# Deployment — how the book is published

Live at **<https://book.omkarsarde.com>** (GitHub Pages, custom domain since 2026-07-27).

## Pipeline

Every push to `main` runs `.github/workflows/publish.yml`:

1. Validate the manuscript (`python scripts/validate_book.py --strict`).
2. Test every executable artifact (`python -m pytest -q`).
3. Render the HTML site (`quarto render --to html`) into `_book/`.
4. Write `_book/CNAME` containing `book.omkarsarde.com` — the CNAME file must ship
   inside the rendered output, or the next publish force-push detaches the custom
   domain from the Pages settings.
5. Publish `_book/` to the `gh-pages` branch (Pages source = `gh-pages`, root).

A failed run leaves the previous `gh-pages` untouched, so a red build never takes
the live site down — it just stops updating it.

## Custom domain

- DNS: `book` CNAME → `omkarsarde.github.io` in Cloudflare, **DNS-only (grey cloud),
  permanently** — proxying the record breaks GitHub's TLS certificate issuance.
- Pages settings: custom domain `book.omkarsarde.com` with **Enforce HTTPS** on
  (Let's Encrypt certificate, auto-renewed by GitHub).
- The original `omkarsarde.github.io/agentic-engineering` URL 301-redirects to the
  custom domain automatically; existing links keep working.

## Formats

CI publishes HTML only. The EPUB is rendered locally and attached to a GitHub
Release; the landing page links `releases/latest` so the download never 404s
between releases. PDF is a manual build requiring a TeX distribution.

## Local rendering

Install Quarto, install `requirements.txt`, and run:

```powershell
python scripts/validate_book.py --strict
python -m pytest -q
quarto render --to html
```

The deployable static site appears in `_book/`. It contains ordinary HTML, CSS,
JavaScript, SVG, search data, and assets, so any static host can serve it. There is
no application server or database bill; hosting cost is $0.

## Why the source is `.qmd`

Quarto Markdown remains readable in a text editor and in source control, while adding
book navigation, cross-references, citations, equations, callouts, executable code
blocks, multiple output formats, and Mermaid rendering. A plain `.md` file would work
for prose, but `.qmd` gives the book-level publishing contract without forcing content
into HTML.

The visual stack is intentionally layered:

- Mermaid for labeled architectures, lifecycles, state machines, and sequences;
- generated SVG for quantitative plots and comparisons;
- tables for exact mappings;
- runnable Python fixtures for behavior that a static picture cannot prove;
- ordinary raster images only when the content is inherently pictorial.

Every figure has a caption and text alternative. Diagrams are used to remove
explanatory burden, not to decorate pages.
