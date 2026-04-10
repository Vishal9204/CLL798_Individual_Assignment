# Complexity Science Project: Software Dependency Networks

This repository contains a complete submission package for a complexity-science project on **software dependency networks**. The system is interpreted as a network of software libraries where:

- **Noise** is a random developer update to one upstream library.
- **Connectivity** is the dependency graph linking libraries to one another.
- **Avalanche** is the downstream cascade of libraries that break or must update after compatibility buffers are exceeded.

The project includes:

- A detailed Python simulation and analysis pipeline.
- Generated CSV outputs and publication-style figures.
- A journal-style LaTeX manuscript in [`report/manuscript.tex`](/Users/vishalgupta/Documents/New%20project/report/manuscript.tex).
- A matching PDF report in [`report/software_dependency_networks_manuscript.pdf`](/Users/vishalgupta/Documents/New%20project/report/software_dependency_networks_manuscript.pdf).
- A prompt log in [`PROMPTS_USED.md`](/Users/vishalgupta/Documents/New%20project/PROMPTS_USED.md).

## Project Structure

- [`run_pipeline.py`](/Users/vishalgupta/Documents/New%20project/run_pipeline.py): full end-to-end workflow.
- [`build_report_pdf.py`](/Users/vishalgupta/Documents/New%20project/build_report_pdf.py): rebuild the PDF from generated results.
- [`src/software_dependency_networks/model.py`](/Users/vishalgupta/Documents/New%20project/src/software_dependency_networks/model.py): network generator and avalanche simulation.
- [`src/software_dependency_networks/pipeline.py`](/Users/vishalgupta/Documents/New%20project/src/software_dependency_networks/pipeline.py): experiment sweep, plots, summaries, and LaTeX macros.
- [`src/software_dependency_networks/reporting.py`](/Users/vishalgupta/Documents/New%20project/src/software_dependency_networks/reporting.py): PDF report renderer.
- [`results/summary_by_connectivity.csv`](/Users/vishalgupta/Documents/New%20project/results/summary_by_connectivity.csv): key statistics by connectivity regime.
- [`results/avalanche_events.csv`](/Users/vishalgupta/Documents/New%20project/results/avalanche_events.csv): per-shock avalanche records.

## How to Run

From the project root:

```bash
python3 run_pipeline.py
```

This command will:

1. simulate the dependency networks across multiple connectivity regimes,
2. generate CSV outputs,
3. save the figures inside `report/figures/`,
4. update `report/generated_summary.tex`, and
5. render the final PDF report.

If you only want to rerender the PDF after the results already exist:

```bash
python3 build_report_pdf.py
```

## Simulation Logic

The simulation uses a directed acyclic graph. A directed edge `i -> j` means package `j` depends on package `i`.

- Each package stores the upstream versions it last validated against.
- A random update increments the version of one source package.
- A downstream package breaks if the accumulated upstream version drift is greater than or equal to its compatibility buffer.
- When a package breaks, it updates, resets its expected upstream versions, and may trigger further downstream failures.

This produces the sandpile-like pattern needed for complexity science:

- **slow drive** through one random update at a time,
- **fast relaxation** through cascades,
- **broad event-size distribution** over many avalanches.

## Submission Checklist

- `.tex` manuscript: [`report/manuscript.tex`](/Users/vishalgupta/Documents/New%20project/report/manuscript.tex)
- PDF manuscript: [`report/software_dependency_networks_manuscript.pdf`](/Users/vishalgupta/Documents/New%20project/report/software_dependency_networks_manuscript.pdf)
- Code and project files: this repository
- Prompt list: [`PROMPTS_USED.md`](/Users/vishalgupta/Documents/New%20project/PROMPTS_USED.md)

## Git Upload

This folder can be pushed to GitHub or any Git-based platform. A typical workflow is:

```bash
git init
git add .
git commit -m "Add complexity science project on software dependency networks"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```

After pushing, submit the repository link along with the PDF and the `.tex` file.
