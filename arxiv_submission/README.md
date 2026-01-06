# arXiv Submission Folder

This folder is organized for source-based arXiv submission.

## Structure
- `source/`: upload-ready LaTeX source bundle
- `docs/`: submission notes/checklist
- `build/`: local compile outputs from the source bundle

## Upload target
Upload the **contents of `source/`** (or `arxiv_source.zip`).

## Local compile test
```bash
cd source
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
