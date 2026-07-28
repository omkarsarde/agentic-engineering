# Agentic Engineering

Source for the book, published at **<https://book.omkarsarde.com>**.

## Render locally

Install [Quarto](https://quarto.org/docs/get-started/) and a Python environment containing the packages in `requirements.txt`, then run:

```powershell
quarto preview
```

Build the deployable website with:

```powershell
quarto render --to html
```

The HTML site is written to `_book/`. CI renders and publishes the HTML site only; the EPUB is rendered locally and attached to a GitHub Release, which the landing page links via `releases/latest`. PDF remains a manual build requiring a TeX distribution. See [DEPLOYMENT.md](DEPLOYMENT.md) for how publishing and the custom domain work, and for the rationale for Quarto Markdown and the visual stack.

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
