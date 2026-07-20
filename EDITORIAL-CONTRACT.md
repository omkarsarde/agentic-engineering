# Editorial contract (v2)

Binding for every chapter and appendix. Where this contract and any older document
disagree, this contract wins. The section-level rhythm is specified in
`_redo/analysis/d2l-style-template.md`; this file states the book-level rules.

## Reader and promise

The reader knows neural-network basics, backpropagation, and Python. Nothing else is
assumed. Every LLM, retrieval, agent, systems, evaluation, and security term is defined in
plain language at first use. The book teaches the way d2l.ai teaches machine learning:
the reader watches working code produce real output, narrated by a knowledgeable "we",
and leaves able to build, measure, and defend each mechanism.

One linear reading path. A reader already fluent in model internals may start at
Chapter 12; the preface says this once. No other routes, boxes, or per-chapter
prerequisites exist.

## Chapter shape

1. **Opening** (no heading): 1–3 paragraphs. First sentence names the concrete thing this
   chapter builds or measures. Enumerate the parts inline. Anchor to prior chapters with
   inline references, never a prerequisites block.
2. **5–9 teaching sections** (H2), each one primary idea, each following the beat:
   motivate → intuition in prose → formal object (numbered, every symbol defined) →
   implement in small executed cells → show the output → interpret it honestly.
   Plain functional section titles.
3. **`## Summary`** — 60–110 words, synthesizing, one forward pointer. Prose or 3 bullets.
4. **`## Exercises`** — 5–8 numbered items, investigative tone, anchored to knobs in this
   chapter's artifact, at least one multi-part, no solutions.

The chapter's single integrated build is the running artifact the sections construct
incrementally. There is no separate "Build" section: by the last teaching section the
artifact exists, has run, and its measurements have been shown.

~4,000–7,000 prose words per chapter (floor 3,000; soft cap 8,000), excluding code and
output. 400–900 prose words per section.

## Code

- Teaching code lives **inline in executable ```` ```{python} ```` cells**, developed
  incrementally: 5–12 cells per section-bearing arc, each 2–20 lines, prose between every
  pair of cells stating why the next cell comes. Never a wall of code, never a fragment of
  an external file, never an elision comment (`# ... added later ...`).
- **Every runnable cell shows its result** — printed values, generated text (imperfect
  output shown honestly), plots, measured numbers — rendered by Quarto at build time.
- Cells whose definitions are reused or tested start with `# @save` (after any `#|`
  options). `scripts/tangle.py` extracts saved cells in order to `code/chNN/_generated.py`;
  tests import only generated modules. Nothing else lives under `code/chNN/`.
- **Docstrings teach.** Public classes and functions carry Google-style docstrings —
  one-line summary, `Args:`, `Returns:`, `Raises:` when relevant — written to teach the
  concept the object embodies. Trivial helpers may carry a single line.
- **Deterministic, offline, fast.** Pin seeds; no network; CPU-only; a chapter's cells
  execute in ≤ 90 seconds total. Real-API calls appear as `#| eval: false` cells paired
  with one captured transcript presented as data and labeled as a recorded run.
- Systems chapters substitute a small inline simulation or measurement for "train a
  model": build the mechanism in fragments, drive it with a synthetic workload, print the
  measured numbers, vary one knob, re-measure. A worked trace of one concrete input is the
  fallback when nothing is measurable.

## Figures

At least 3 per chapter, no ceiling; every one earns its place. Prefer plots produced by
the chapter's own cells. Concept diagrams (mermaid/dot/SVG) are for comparing
alternatives, locating a component in a larger system, or tracing a flow prose cannot
carry. Every figure is numbered, captioned with the question it answers, referenced from
prose, and legible in grayscale. Diagram source is version-controlled.

## Voice

First-person plural "we", present tense, intuition before rigor, short declaratives to
land long explanations, lightly opinionated ("inelegant but ubiquitous" is on-voice).
Forbidden: "obviously", "clearly", "trivially", "of course"; gatekeeping; interview
stakes as motivation; contract-jargon ("code-owned gate", "typed endings", "canonical
home") outside a definition site. Cross-references inline and numbered
("recall from @sec-ch09"). Citations inline (Author, Year), load-bearing only, resolving
to `references.bib`. No footnotes, no "Notes and sources" section.

## Deleted apparatus (must not appear anywhere)

"What you need going in" · route/backfill boxes · "Contents" link lists · "What you will
build" callouts · artifact-checkpoint tables · "Acceptance checks" · "Honesty note" ·
"What endures, what changes" · durability tags in titles · learning-objectives boxes ·
troubleshooting sidebars · per-section setup instructions. Honest caveats live as
ordinary sentences where they matter.

## Dated facts

Volatile facts (model names, versions, prices, leaderboards, SKUs, legal dates) live in
`.landscape-2026` callouts — ISO verification date, primary-source links, a closing
"Verify live" instruction, deletable without breaking the chapter — or in Appendix C's
registry for cross-chapter operational pins. The spine teaches mechanisms.

## Interview boxes

At most one short "In the interview" callout per chapter: one question, the crisp answer,
the trap. The integrated 45-minute design arguments live only in Appendix B's design
studies; Chapter 32 exercises them.

## Ownership boundaries

Unchanged from v1: KV arithmetic Ch 3 · behavior specs Ch 7 · hallucination/calibration/
abstention Ch 9 · customization/merging Ch 11 · compaction/context Ch 13 · core RAG
metrics Ch 14 · agentic retrieval Ch 15 · raw loop Ch 16 · skills/approval mechanics
Ch 17 · memory policy/deletion Ch 18 · MCP security/checkpointers Ch 19 · statistical
eval/judge calibration Ch 22 · action security Ch 24 · cost/durable effects/delivery
Ch 26 · observability/SLO/incidents/HITL-ops/privacy-ops Ch 27 · product/people/AIMS
Ch 28 · media provenance Ch 30. Elsewhere: recap in one paragraph, point to the owner.
