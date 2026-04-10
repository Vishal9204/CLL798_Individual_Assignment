from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .model import DependencyNetworkSimulation, NetworkConfig, empirical_ccdf


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment sweep configuration."""

    connectivity_values: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    replicates: int = 5
    node_count: int = 450
    steps: int = 5500
    burn_in: int = 1000
    sample_interval: int = 10
    base_buffer: float = 1.5
    buffer_slope: float = 0.4
    large_avalanche_threshold: int = 20
    seed: int = 42
    example_node_count: int = 36
    example_min_size: int = 8
    example_max_size: int = 18


def run_experiment(project_root: Path, config: ExperimentConfig) -> dict[str, object]:
    """Run the connectivity sweep, save all analysis artefacts, and return metrics."""

    results_dir = project_root / "results"
    figures_dir = project_root / "report" / "figures"
    report_dir = project_root / "report"

    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    event_frames: list[pd.DataFrame] = []
    snapshot_frames: list[pd.DataFrame] = []

    for connectivity_index, mean_dependencies in enumerate(config.connectivity_values):
        for replicate in range(config.replicates):
            run_seed = config.seed + connectivity_index * 100 + replicate
            simulation = DependencyNetworkSimulation.from_config(
                NetworkConfig(
                    node_count=config.node_count,
                    mean_dependencies=mean_dependencies,
                    base_buffer=config.base_buffer,
                    buffer_slope=config.buffer_slope,
                    steps=config.steps,
                    burn_in=config.burn_in,
                    seed=run_seed,
                    sample_interval=config.sample_interval,
                )
            )

            events, snapshots = simulation.run()
            event_frame = pd.DataFrame(events)
            snapshot_frame = pd.DataFrame(snapshots)

            event_frame["target_mean_dependencies"] = mean_dependencies
            event_frame["actual_mean_indegree"] = simulation.actual_mean_indegree
            event_frame["actual_mean_outdegree"] = simulation.actual_mean_outdegree
            event_frame["replicate"] = replicate

            snapshot_frame["target_mean_dependencies"] = mean_dependencies
            snapshot_frame["actual_mean_indegree"] = simulation.actual_mean_indegree
            snapshot_frame["replicate"] = replicate

            event_frames.append(event_frame)
            snapshot_frames.append(snapshot_frame)

    events_df = pd.concat(event_frames, ignore_index=True)
    snapshots_df = pd.concat(snapshot_frames, ignore_index=True)

    summary_df = _summarise_events(
        events_df=events_df,
        large_avalanche_threshold=config.large_avalanche_threshold,
    )

    selected_rows = _select_low_mid_high(summary_df)
    example_event = _create_example_visualisation(
        figures_dir=figures_dir,
        target_mean_dependencies=float(selected_rows["mid"]["target_mean_dependencies"]),
        config=config,
    )

    _plot_avalanche_distribution(
        events_df=events_df,
        selected_rows=selected_rows,
        output_path=figures_dir / "avalanche_distribution_ccdf.png",
    )
    _plot_connectivity_risk(
        summary_df=summary_df,
        output_path=figures_dir / "connectivity_vs_cascade_risk.png",
    )
    _plot_criticality_dynamics(
        events_df=events_df,
        snapshots_df=snapshots_df,
        target_mean_dependencies=float(selected_rows["mid"]["target_mean_dependencies"]),
        output_path=figures_dir / "criticality_dynamics.png",
    )

    events_df.to_csv(results_dir / "avalanche_events.csv", index=False)
    snapshots_df.to_csv(results_dir / "state_timeseries.csv", index=False)
    summary_df.to_csv(results_dir / "summary_by_connectivity.csv", index=False)

    metrics = _build_metrics(
        config=config,
        summary_df=summary_df,
        selected_rows=selected_rows,
        example_event=example_event,
    )
    with (results_dir / "report_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    with (results_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    _write_generated_summary_tex(
        output_path=report_dir / "generated_summary.tex",
        metrics=metrics,
    )

    return metrics


def _summarise_events(
    events_df: pd.DataFrame,
    large_avalanche_threshold: int,
) -> pd.DataFrame:
    grouped = events_df.groupby("target_mean_dependencies", sort=True)
    summary_df = grouped.agg(
        actual_mean_indegree=("actual_mean_indegree", "mean"),
        mean_avalanche_size=("avalanche_size", "mean"),
        median_avalanche_size=("avalanche_size", "median"),
        p95_avalanche_size=("avalanche_size", lambda x: float(np.percentile(x, 95))),
        nonzero_probability=("avalanche_size", lambda x: float((x > 0).mean())),
        large_avalanche_probability=(
            "avalanche_size",
            lambda x: float((x >= large_avalanche_threshold).mean()),
        ),
        max_avalanche_size=("avalanche_size", "max"),
        mean_max_depth=("max_depth", "mean"),
        mean_vulnerable_fraction=("vulnerable_fraction", "mean"),
        mean_stress_ratio=("mean_stress_ratio", "mean"),
    ).reset_index()
    return summary_df


def _select_low_mid_high(summary_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    ordered = summary_df.sort_values("target_mean_dependencies").reset_index(drop=True)
    mid_index = len(ordered) // 2
    selected = {
        "low": ordered.iloc[0].to_dict(),
        "mid": ordered.iloc[mid_index].to_dict(),
        "high": ordered.iloc[-1].to_dict(),
    }
    return {
        key: {
            metric: float(value) if isinstance(value, (np.floating, np.integer)) else value
            for metric, value in row.items()
        }
        for key, row in selected.items()
    }


def _create_example_visualisation(
    figures_dir: Path,
    target_mean_dependencies: float,
    config: ExperimentConfig,
) -> dict[str, float | int]:
    import matplotlib.pyplot as plt
    import networkx as nx

    example_sim = DependencyNetworkSimulation.from_config(
        NetworkConfig(
            node_count=config.example_node_count,
            mean_dependencies=target_mean_dependencies,
            base_buffer=config.base_buffer,
            buffer_slope=config.buffer_slope,
            steps=2000,
            burn_in=0,
            seed=config.seed + 999,
            sample_interval=1,
        )
    )
    event = example_sim.find_interesting_avalanche(
        min_size=config.example_min_size,
        max_size=config.example_max_size,
    )

    graph = nx.DiGraph()
    graph.add_nodes_from(range(example_sim.node_count))
    graph.add_edges_from(example_sim.edges())
    layout = nx.spring_layout(graph, seed=7, k=0.6)

    avalanche_nodes = set(event["avalanche_nodes"])
    source_node = int(event["shock_node"])
    colors = []
    sizes = []
    for node in graph.nodes:
        if node == source_node:
            colors.append("#1f77b4")
        elif node in avalanche_nodes:
            colors.append("#d62728")
        else:
            colors.append("#d7d7d7")
        sizes.append(180 + 20 * graph.out_degree(node))

    figure, axis = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_edges(
        graph,
        pos=layout,
        ax=axis,
        edge_color="#c7c7c7",
        arrows=False,
        width=1.0,
        alpha=0.75,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=layout,
        ax=axis,
        node_color=colors,
        node_size=sizes,
        linewidths=0.8,
        edgecolors="#555555",
    )
    axis.set_title(
        "Illustrative cascade in a software dependency network",
        fontsize=14,
        pad=12,
    )
    axis.text(
        0.01,
        0.02,
        (
            f"Blue: noisy source library {source_node} | "
            f"Red: downstream libraries that failed and updated | "
            f"Avalanche size = {event['avalanche_size']}"
        ),
        transform=axis.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(figures_dir / "network_avalanche_example.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    return {
        "shock_node": int(event["shock_node"]),
        "avalanche_size": int(event["avalanche_size"]),
        "total_updates": int(event["total_updates"]),
        "max_depth": int(event["max_depth"]),
        "actual_mean_indegree": example_sim.actual_mean_indegree,
    }


def _plot_avalanche_distribution(
    events_df: pd.DataFrame,
    selected_rows: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 6.2))
    palette = {"low": "#4c78a8", "mid": "#f58518", "high": "#54a24b"}

    for key in ("low", "mid", "high"):
        target_value = selected_rows[key]["target_mean_dependencies"]
        subset = events_df.loc[
            events_df["target_mean_dependencies"] == target_value, "avalanche_size"
        ]
        sizes, ccdf = empirical_ccdf(subset.tolist())
        axis.loglog(
            sizes,
            ccdf,
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=4.0,
            label=(
                f"{key.title()} connectivity "
                f"(mean in-degree ≈ {selected_rows[key]['actual_mean_indegree']:.2f})"
            ),
            color=palette[key],
        )

    axis.set_xlabel("Avalanche size, s")
    axis.set_ylabel("P(S ≥ s)")
    axis.set_title("Avalanche-size distribution broadens as connectivity increases")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_connectivity_risk(summary_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))

    x_values = summary_df["actual_mean_indegree"].to_numpy()
    axes[0].plot(
        x_values,
        summary_df["mean_avalanche_size"],
        color="#1f77b4",
        marker="o",
        linewidth=2,
        label="Mean avalanche size",
    )
    axes[0].plot(
        x_values,
        summary_df["p95_avalanche_size"],
        color="#d62728",
        marker="s",
        linewidth=2,
        label="95th percentile avalanche size",
    )
    axes[0].set_xlabel("Actual mean in-degree")
    axes[0].set_ylabel("Cascade magnitude")
    axes[0].set_title("Connectivity amplifies typical and extreme cascades")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        x_values,
        100.0 * summary_df["large_avalanche_probability"],
        color="#f58518",
        marker="o",
        linewidth=2,
        label="P(avalanche size ≥ 20)",
    )
    axes[1].plot(
        x_values,
        100.0 * summary_df["mean_vulnerable_fraction"],
        color="#54a24b",
        marker="^",
        linewidth=2,
        label="Vulnerable fraction",
    )
    axes[1].set_xlabel("Actual mean in-degree")
    axes[1].set_ylabel("Percentage of events / nodes")
    axes[1].set_title("Large cascades rise while vulnerability remains persistent")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _plot_criticality_dynamics(
    events_df: pd.DataFrame,
    snapshots_df: pd.DataFrame,
    target_mean_dependencies: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    mid_events = events_df[
        (events_df["target_mean_dependencies"] == target_mean_dependencies)
        & (events_df["replicate"] == 0)
    ].copy()
    mid_snapshots = snapshots_df[
        (snapshots_df["target_mean_dependencies"] == target_mean_dependencies)
        & (snapshots_df["replicate"] == 0)
    ].copy()

    mid_snapshots["rolling_vulnerability"] = (
        mid_snapshots["vulnerable_fraction"].rolling(window=12, min_periods=1).mean()
    )
    mid_snapshots["rolling_stress"] = (
        mid_snapshots["mean_stress_ratio"].rolling(window=12, min_periods=1).mean()
    )

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)

    axes[0].plot(
        mid_snapshots["step"],
        100.0 * mid_snapshots["rolling_vulnerability"],
        color="#54a24b",
        linewidth=2,
        label="Rolling vulnerable fraction",
    )
    axes[0].plot(
        mid_snapshots["step"],
        100.0 * mid_snapshots["rolling_stress"],
        color="#1f77b4",
        linewidth=2,
        label="Rolling mean stress ratio",
    )
    axes[0].set_ylabel("Percent")
    axes[0].set_title(
        "Slow loading keeps the network near a persistent vulnerable band"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        mid_events["step"],
        mid_events["avalanche_size"],
        color="#d62728",
        linewidth=1.0,
    )
    axes[1].set_xlabel("Recorded update event after burn-in")
    axes[1].set_ylabel("Avalanche size")
    axes[1].set_title("Fast relaxation appears as intermittent cascade spikes")
    axes[1].grid(True, alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def _build_metrics(
    config: ExperimentConfig,
    summary_df: pd.DataFrame,
    selected_rows: dict[str, dict[str, float]],
    example_event: dict[str, float | int],
) -> dict[str, object]:
    low = selected_rows["low"]
    mid = selected_rows["mid"]
    high = selected_rows["high"]

    return {
        "title": "Avalanche Dynamics in Software Dependency Networks",
        "config": asdict(config),
        "summary_rows": summary_df.to_dict(orient="records"),
        "selected_rows": selected_rows,
        "example_event": example_event,
        "derived": {
            "mean_avalanche_growth_factor": high["mean_avalanche_size"]
            / low["mean_avalanche_size"],
            "large_avalanche_probability_change_points": 100.0
            * (high["large_avalanche_probability"] - low["large_avalanche_probability"]),
            "high_connectivity_max_avalanche": high["max_avalanche_size"],
            "mid_connectivity_vulnerable_fraction_percent": 100.0
            * mid["mean_vulnerable_fraction"],
        },
    }


def _write_generated_summary_tex(output_path: Path, metrics: dict[str, object]) -> None:
    config = metrics["config"]
    selected = metrics["selected_rows"]
    derived = metrics["derived"]

    def numeric(value: float) -> str:
        return f"{value:.2f}"

    lines = [
        "% Auto-generated by run_pipeline.py",
        f"\\newcommand{{\\NodeCount}}{{{int(config['node_count'])}}}",
        f"\\newcommand{{\\TotalSteps}}{{{int(config['steps'])}}}",
        f"\\newcommand{{\\BurnIn}}{{{int(config['burn_in'])}}}",
        f"\\newcommand{{\\RecordedShocks}}{{{int(config['steps'] - config['burn_in'])}}}",
        f"\\newcommand{{\\ReplicateCount}}{{{int(config['replicates'])}}}",
        (
            "\\newcommand{\\ConnectivityTargets}{"
            + ", ".join(f"{value:.1f}" for value in config["connectivity_values"])
            + "}"
        ),
        f"\\newcommand{{\\LargeAvalancheThreshold}}{{{int(config['large_avalanche_threshold'])}}}",
        f"\\newcommand{{\\ConnectivityLow}}{{{numeric(selected['low']['actual_mean_indegree'])}}}",
        f"\\newcommand{{\\ConnectivityMid}}{{{numeric(selected['mid']['actual_mean_indegree'])}}}",
        f"\\newcommand{{\\ConnectivityHigh}}{{{numeric(selected['high']['actual_mean_indegree'])}}}",
        f"\\newcommand{{\\MeanAvalancheLow}}{{{numeric(selected['low']['mean_avalanche_size'])}}}",
        f"\\newcommand{{\\MeanAvalancheMid}}{{{numeric(selected['mid']['mean_avalanche_size'])}}}",
        f"\\newcommand{{\\MeanAvalancheHigh}}{{{numeric(selected['high']['mean_avalanche_size'])}}}",
        f"\\newcommand{{\\P95AvalancheLow}}{{{numeric(selected['low']['p95_avalanche_size'])}}}",
        f"\\newcommand{{\\P95AvalancheMid}}{{{numeric(selected['mid']['p95_avalanche_size'])}}}",
        f"\\newcommand{{\\P95AvalancheHigh}}{{{numeric(selected['high']['p95_avalanche_size'])}}}",
        f"\\newcommand{{\\LargeProbLow}}{{{numeric(100.0 * selected['low']['large_avalanche_probability'])}}}",
        f"\\newcommand{{\\LargeProbMid}}{{{numeric(100.0 * selected['mid']['large_avalanche_probability'])}}}",
        f"\\newcommand{{\\LargeProbHigh}}{{{numeric(100.0 * selected['high']['large_avalanche_probability'])}}}",
        (
            f"\\newcommand{{\\VulnerableMid}}{{"
            f"{numeric(100.0 * selected['mid']['mean_vulnerable_fraction'])}}}"
        ),
        f"\\newcommand{{\\MeanGrowth}}{{{numeric(derived['mean_avalanche_growth_factor'])}}}",
        (
            f"\\newcommand{{\\LargeProbChange}}{{"
            f"{numeric(derived['large_avalanche_probability_change_points'])}}}"
        ),
        (
            f"\\newcommand{{\\HighMaxAvalanche}}{{"
            f"{int(derived['high_connectivity_max_avalanche'])}}}"
        ),
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
