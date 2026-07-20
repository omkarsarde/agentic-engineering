# Agentic Engineering

This directory is a clean-room rewrite of the book. It does not modify or replace the legacy manuscript in `genai-gym/book`.

## Render locally

Install [Quarto](https://quarto.org/docs/get-started/) and a Python environment containing the packages in `requirements.txt`, then run:

```powershell
quarto preview
```

Build the deployable website with:

```powershell
quarto render --to html
```

The HTML site is written to `_book/`. A full `quarto render` also requests the configured PDF and EPUB formats; PDF rendering requires a TeX distribution. The CI pipeline renders the HTML site and the EPUB together, so the download link on the landing page always matches the published content. PDF remains a manual build requiring a TeX distribution. GitHub Pages deployment is defined in `.github/workflows/publish.yml` for the case where this directory becomes the repository root. See [DEPLOYMENT.md](DEPLOYMENT.md) for the $0 hosting paths, nested-repository setup, and the rationale for Quarto Markdown and the visual stack.

## Validate source

```powershell
python scripts/validate_book.py --strict
```

The validator checks navigation, chapter apparatus, route-B backfills, visual counts, forbidden phrases, duplicate headings, and broken local links. Executable reference artifacts are tested separately under `tests/`.

## Source hierarchy

1. `_quarto.yml` defines the public book order.
2. `EDITORIAL-CONTRACT.md` defines the teaching and code contract.
3. `VISUAL-SYSTEM.md` defines figure selection and accessibility.
4. Appendix C owns volatile facts; the spine owns durable mechanisms.
