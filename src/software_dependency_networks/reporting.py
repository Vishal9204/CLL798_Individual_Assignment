from __future__ import annotations

import json
import textwrap
from pathlib import Path


def build_report_pdf(
    project_root: Path,
    metrics: dict[str, object] | None = None,
    output_path: Path | None = None,
) -> Path:
    """Render a submission-ready PDF report using matplotlib's PDF backend."""

    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    if metrics is None:
        metrics_path = project_root / "results" / "report_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

    if output_path is None:
        output_path = project_root / "report" / "software_dependency_networks_manuscript.pdf"

    selected = metrics["selected_rows"]
    derived = metrics["derived"]
    config = metrics["config"]
    figures_dir = project_root / "report" / "figures"
    prompts_path = project_root / "PROMPTS_USED.md"
    prompts_lines = [
        line.strip("- ").strip()
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("- ")
    ]

    title = metrics["title"]

    def add_text_page(pdf: PdfPages, page_title: str, paragraphs: list[str]) -> None:
        figure = plt.figure(figsize=(8.27, 11.69))
        axis = figure.add_axes([0.08, 0.05, 0.84, 0.9])
        axis.axis("off")
        y_position = 0.98
        axis.text(
            0.0,
            y_position,
            page_title,
            fontsize=18,
            fontweight="bold",
            va="top",
        )
        y_position -= 0.055

        for paragraph in paragraphs:
            wrapped = textwrap.fill(paragraph, width=104)
            axis.text(
                0.0,
                y_position,
                wrapped,
                fontsize=10.5,
                va="top",
                linespacing=1.45,
            )
            line_count = wrapped.count("\n") + 1
            y_position -= 0.028 * line_count + 0.028

        axis.text(
            0.0,
            0.015,
            "Generated locally from the included code and simulation outputs.",
            fontsize=8.5,
            color="#555555",
        )
        pdf.savefig(figure)
        plt.close(figure)

    def add_figure_page(
        pdf: PdfPages,
        page_title: str,
        image_path: Path,
        caption: str,
    ) -> None:
        figure = plt.figure(figsize=(8.27, 11.69))
        axis = figure.add_axes([0.08, 0.12, 0.84, 0.78])
        image = mpimg.imread(image_path)
        axis.imshow(image)
        axis.axis("off")
        figure.suptitle(page_title, fontsize=17, fontweight="bold", y=0.97)
        figure.text(
            0.08,
            0.075,
            textwrap.fill(caption, width=112),
            fontsize=10.5,
            va="bottom",
        )
        pdf.savefig(figure)
        plt.close(figure)

    abstract_page = [
        (
            f"{title} studies a stylized ecosystem of interconnected software libraries as a complex adaptive system. "
            "In the model, noise is a random developer update to one upstream library, connectivity is the dependency structure linking packages, "
            "and an avalanche is the number of downstream libraries that must update after compatibility buffers are exceeded."
        ),
        (
            f"The simulation uses {config['node_count']} libraries, {config['replicates']} replicate networks per connectivity regime, "
            f"and {config['steps'] - config['burn_in']} recorded shocks after burn-in for each replicate. "
            f"Across the sweep, mean in-degree rises from about {selected['low']['actual_mean_indegree']:.2f} to "
            f"{selected['high']['actual_mean_indegree']:.2f} and the mean avalanche size grows by a factor of "
            f"{derived['mean_avalanche_growth_factor']:.2f}."
        ),
        (
            f"Large cascades also become more common: the probability of an avalanche of size at least "
            f"{config['large_avalanche_threshold']} increases by {derived['large_avalanche_probability_change_points']:.2f} percentage points "
            f"between the lowest and highest connectivity conditions, and the largest observed cascade in the highest-connectivity regime "
            f"reaches {int(derived['high_connectivity_max_avalanche'])} libraries."
        ),
        (
            "The results show a broad avalanche-size distribution and a persistent band of near-failure libraries, which is consistent with "
            "self-organized near-critical behaviour: the network is driven slowly by isolated updates and relaxes quickly through irregular cascades."
        ),
    ]

    methods_page = [
        (
            "Why this is a good complexity-science system: software package ecosystems exhibit many interacting components, heterogeneity, "
            "path dependence, local compatibility rules, and global consequences that are not obvious from any single package."
        ),
        (
            "Model details: packages are added in age order to form a directed acyclic dependency graph. Preferential attachment makes popular "
            "libraries accumulate more dependents, producing hub-like structures similar to real package ecosystems. Every package stores the versions "
            "of its upstream dependencies that it last validated against."
        ),
        (
            "Noise, avalanche, and connectivity mapping: a random source package receives a version increment of one. A downstream package breaks when the "
            "accumulated upstream version drift reaches its compatibility buffer. When it breaks, it updates, resets its expected upstream versions, and can "
            "trigger further downstream failures. Connectivity is controlled by the target mean number of dependencies per package."
        ),
        (
            "Analytical outputs: the code records avalanche size, maximum cascade depth, mean stress ratio, and the vulnerable fraction of libraries that are "
            "within one extra upstream version change of failure. These observables are used to compare connectivity regimes and to comment on self-organized criticality."
        ),
    ]

    results_page = [
        (
            f"In the low-connectivity regime (mean in-degree about {selected['low']['actual_mean_indegree']:.2f}), the mean avalanche size is "
            f"{selected['low']['mean_avalanche_size']:.2f} and the 95th percentile is {selected['low']['p95_avalanche_size']:.2f}. "
            f"Large cascades are relatively rare, with probability {100.0 * selected['low']['large_avalanche_probability']:.2f}%."
        ),
        (
            f"In the middle regime (mean in-degree about {selected['mid']['actual_mean_indegree']:.2f}), avalanches become broader and the network "
            f"maintains a vulnerable fraction near {100.0 * selected['mid']['mean_vulnerable_fraction']:.2f}%. "
            "This indicates that a substantial minority of libraries are continually poised close to their compatibility thresholds."
        ),
        (
            f"In the high-connectivity regime (mean in-degree about {selected['high']['actual_mean_indegree']:.2f}), the mean avalanche size rises to "
            f"{selected['high']['mean_avalanche_size']:.2f}, the 95th percentile reaches {selected['high']['p95_avalanche_size']:.2f}, "
            f"and the largest single cascade affects {int(selected['high']['max_avalanche_size'])} libraries."
        ),
        (
            "Taken together, the figures show the classic slow-drive and fast-relaxation pattern associated with critical systems: isolated updates steadily "
            "reload the network, while occasional cascades discharge the accumulated compatibility stress over many downstream packages."
        ),
    ]

    discussion_page = [
        (
            "Comment on self-organized criticality: this model does not prove exact critical exponents for a real software ecosystem, but it does display the main "
            "qualitative ingredients of self-organized critical behaviour. The system is driven slowly, relaxes rapidly, exhibits broad cascade distributions, and "
            "settles into a persistent near-threshold state without per-event retuning of parameters."
        ),
        (
            "Interpretation: software dependency networks become more fragile as connectivity increases because every update has more possible downstream paths. "
            "At the same time, the network never becomes completely quiescent; instead it hovers in a regime where rare but very large cascades remain possible."
        ),
        (
            "Limitations: real package ecosystems include semantic versioning, optional dependencies, maintainers, patch backports, and social coordination, none of "
            "which are modelled here. The simulation is therefore a stylized complexity-science analogue rather than a literal predictor of PyPI or npm behaviour."
        ),
        "Prompts used in preparing this project:",
    ]
    discussion_page.extend(f"{index}. {prompt}" for index, prompt in enumerate(prompts_lines, start=1))
    discussion_page.extend(
        [
            (
                "Acknowledgement of help: the project used AI assistance for ideation, coding support, report drafting, and packaging. "
                "All generated claims in the report are tied to locally reproducible simulation outputs included in the repository."
            ),
            (
                "Submission note: the repository includes the LaTeX manuscript, a matching PDF report, the Python source code, generated figures, and CSV outputs "
                "so the entire submission can be rerun and inspected."
            ),
        ]
    )

    with PdfPages(output_path) as pdf:
        add_text_page(pdf, title, abstract_page)
        add_text_page(pdf, "Model and Experimental Design", methods_page)
        add_figure_page(
            pdf,
            "Figure 1. Example Update Cascade",
            figures_dir / "network_avalanche_example.png",
            (
                "A single upstream update (blue node) can trigger a downstream avalanche (red nodes) because every dependent library carries a finite "
                "compatibility buffer. This visual corresponds directly to the assignment definitions of noise, connectivity, and avalanche."
            ),
        )
        add_figure_page(
            pdf,
            "Figure 2. Avalanche-Size Distribution",
            figures_dir / "avalanche_distribution_ccdf.png",
            (
                "The complementary cumulative distribution broadens as connectivity rises, showing that denser dependency structures create a heavier tail of "
                "possible cascade sizes."
            ),
        )
        add_figure_page(
            pdf,
            "Figure 3. Connectivity and Cascade Risk",
            figures_dir / "connectivity_vs_cascade_risk.png",
            (
                "Increasing mean in-degree amplifies both the typical avalanche size and the probability of large cascades, while a persistent vulnerable fraction "
                "of libraries remains close to failure thresholds."
            ),
        )
        add_figure_page(
            pdf,
            "Figure 4. Slow Loading and Fast Relaxation",
            figures_dir / "criticality_dynamics.png",
            (
                "For a representative medium-connectivity network, the vulnerable fraction and stress ratio remain in a steady band while avalanches appear as "
                "intermittent spikes, which is the characteristic signature of near-critical dynamics."
            ),
        )
        add_text_page(pdf, "Results and Discussion", results_page + discussion_page)

    return output_path
