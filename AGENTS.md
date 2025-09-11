# Agent Communication Guidelines

- Always explain concepts in simple terms, like you are talking to a thoughtful 15-year-old.
- Prefer clear, plain words over jargon; introduce any needed term with a short, concrete example.
- When numbers or thresholds are involved, state what they mean in practice (what passes, what fails) and why.
- When describing formulas, give the plain-English meaning first, then the formula.
- Use short steps and bullets for procedures; keep each step action-oriented.
- Call out assumptions and defaults explicitly so behavior is predictable.

This applies to all written answers and code review comments produced for this project.

## Session Rule — writtenThesis format lock (permanent)

- Do not change the LaTeX template structure or style files in `writtenThesis/`.
- Allowed edits: only the textual content of chapter files and supporting content files:
  - `writtenThesis/abs.tex`, `acros.tex`, `chap-intro.tex`, `chap-art.tex`, `chap-meth.tex`, `chap-development.tex`, `chap-results.tex`, `chap-conclusion.tex`, `appendix1.tex`, and `refs.bib`.
- Forbidden edits without explicit user approval: `thesis.tex` preamble structure, `upthesis.sty`, package list, cover templates, or any formatting/layout parameters.
- Assumption: language is UK English unless the user instructs otherwise.
- Goal: preserve the faculty’s default thesis format; only supply content.

