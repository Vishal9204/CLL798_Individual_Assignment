from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for a single dependency-network simulation."""

    node_count: int = 450
    mean_dependencies: float = 3.0
    base_buffer: float = 1.5
    buffer_slope: float = 0.4
    steps: int = 5500
    burn_in: int = 1000
    seed: int = 0
    sample_interval: int = 10


class DependencyNetworkSimulation:
    """Simulate update cascades on a directed software dependency network.

    The model treats each library as a node in a directed acyclic graph (DAG).
    A directed edge i -> j means that library j depends on library i.

    Each library j stores the version of every upstream dependency that it last
    validated against. Random updates increase the version of a selected source
    library by one. A downstream library breaks and updates when the accumulated
    upstream version drift exceeds its compatibility buffer.
    """

    def __init__(
        self,
        config: NetworkConfig,
        dependencies: list[set[int]],
        dependents: list[set[int]],
        buffers: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.node_count = config.node_count
        self.dependencies = dependencies
        self.dependents = dependents
        self.buffers = buffers.astype(int)
        self.rng = rng

        self.versions = np.zeros(self.node_count, dtype=int)
        self.expected_versions = [
            {parent: 0 for parent in parents} for parents in self.dependencies
        ]
        self.in_degrees = np.array([len(parents) for parents in self.dependencies])
        self.out_degrees = np.array([len(children) for children in self.dependents])
        self.actual_mean_indegree = float(self.in_degrees.mean())
        self.actual_mean_outdegree = float(self.out_degrees.mean())

    @classmethod
    def from_config(cls, config: NetworkConfig) -> "DependencyNetworkSimulation":
        rng = np.random.default_rng(config.seed)
        dependencies, dependents = cls._generate_dependency_graph(
            node_count=config.node_count,
            mean_dependencies=config.mean_dependencies,
            rng=rng,
        )
        buffers = cls._generate_buffers(
            dependencies=dependencies,
            base_buffer=config.base_buffer,
            buffer_slope=config.buffer_slope,
            rng=rng,
        )
        return cls(
            config=config,
            dependencies=dependencies,
            dependents=dependents,
            buffers=buffers,
            rng=rng,
        )

    @staticmethod
    def _generate_dependency_graph(
        node_count: int,
        mean_dependencies: float,
        rng: np.random.Generator,
    ) -> tuple[list[set[int]], list[set[int]]]:
        """Create a DAG using age ordering plus preferential attachment.

        Newer packages select older packages as dependencies. Popular packages are
        more likely to attract additional dependents, which produces realistic hubs.
        """

        dependencies = [set() for _ in range(node_count)]
        dependents = [set() for _ in range(node_count)]
        popularity = np.ones(node_count, dtype=float)

        for node in range(1, node_count):
            sampled = max(1, int(rng.poisson(max(mean_dependencies, 0.1))))
            dependency_count = min(node, sampled)
            weights = popularity[:node] / popularity[:node].sum()
            parents = rng.choice(
                node,
                size=dependency_count,
                replace=False,
                p=weights,
            )
            for parent in parents:
                parent_id = int(parent)
                dependencies[node].add(parent_id)
                dependents[parent_id].add(node)
                popularity[parent_id] += 1.0

        return dependencies, dependents

    @staticmethod
    def _generate_buffers(
        dependencies: list[set[int]],
        base_buffer: float,
        buffer_slope: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        in_degrees = np.array([len(parents) for parents in dependencies], dtype=int)
        buffer_noise = rng.integers(0, 2, size=len(dependencies))
        raw = base_buffer + buffer_slope * in_degrees + buffer_noise
        return np.maximum(1, np.round(raw)).astype(int)

    def edges(self) -> list[tuple[int, int]]:
        edge_list: list[tuple[int, int]] = []
        for child, parents in enumerate(self.dependencies):
            for parent in parents:
                edge_list.append((parent, child))
        return edge_list

    def node_strain(self, node: int) -> int:
        return sum(
            self.versions[parent] - self.expected_versions[node][parent]
            for parent in self.dependencies[node]
        )

    def state_metrics(self) -> tuple[float, float]:
        stress_ratios: list[float] = []
        vulnerable_nodes = 0
        for node in range(self.node_count):
            if not self.dependencies[node]:
                continue
            strain = self.node_strain(node)
            stress_ratios.append(strain / self.buffers[node])
            if self.buffers[node] - strain <= 1:
                vulnerable_nodes += 1

        if not stress_ratios:
            return 0.0, 0.0

        return float(np.mean(stress_ratios)), vulnerable_nodes / len(stress_ratios)

    def apply_noise(self, source_node: int) -> dict[str, object]:
        """Apply one random upstream update and relax the resulting avalanche."""

        self.versions[source_node] += 1
        queue: deque[tuple[int, int]] = deque(
            (child, 1) for child in self.dependents[source_node]
        )
        queued_nodes = {child for child in self.dependents[source_node]}

        avalanche_nodes: set[int] = set()
        max_depth = 0

        while queue:
            node, depth = queue.popleft()
            queued_nodes.discard(node)

            if not self.dependencies[node]:
                continue

            strain = self.node_strain(node)
            if strain < self.buffers[node]:
                continue

            avalanche_nodes.add(node)
            self.versions[node] += 1
            max_depth = max(max_depth, depth)

            for parent in self.dependencies[node]:
                self.expected_versions[node][parent] = int(self.versions[parent])

            for child in self.dependents[node]:
                if child in queued_nodes:
                    continue
                queue.append((child, depth + 1))
                queued_nodes.add(child)

        mean_stress_ratio, vulnerable_fraction = self.state_metrics()
        return {
            "shock_node": source_node,
            "avalanche_nodes": avalanche_nodes,
            "avalanche_size": len(avalanche_nodes),
            "total_updates": len(avalanche_nodes) + 1,
            "max_depth": max_depth,
            "mean_stress_ratio": mean_stress_ratio,
            "vulnerable_fraction": vulnerable_fraction,
        }

    def run(self) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
        events: list[dict[str, float | int]] = []
        snapshots: list[dict[str, float | int]] = []

        for step in range(self.config.steps):
            shock_node = int(self.rng.integers(0, self.node_count))
            event = self.apply_noise(source_node=shock_node)

            if step < self.config.burn_in:
                continue

            sampled_step = step - self.config.burn_in
            events.append(
                {
                    "step": sampled_step,
                    "shock_node": int(event["shock_node"]),
                    "avalanche_size": int(event["avalanche_size"]),
                    "total_updates": int(event["total_updates"]),
                    "max_depth": int(event["max_depth"]),
                    "mean_stress_ratio": float(event["mean_stress_ratio"]),
                    "vulnerable_fraction": float(event["vulnerable_fraction"]),
                }
            )

            if sampled_step % self.config.sample_interval == 0:
                snapshots.append(
                    {
                        "step": sampled_step,
                        "mean_stress_ratio": float(event["mean_stress_ratio"]),
                        "vulnerable_fraction": float(event["vulnerable_fraction"]),
                    }
                )

        return events, snapshots

    def find_interesting_avalanche(
        self,
        min_size: int = 8,
        max_size: int = 18,
        max_attempts: int = 3000,
    ) -> dict[str, object]:
        """Search for an avalanche with a size range that plots clearly."""

        best_event: dict[str, object] | None = None
        best_distance = float("inf")

        for attempt in range(max_attempts):
            shock_node = int(self.rng.integers(0, self.node_count))
            event = self.apply_noise(source_node=shock_node)
            size = int(event["avalanche_size"])
            distance = 0
            if size < min_size:
                distance = min_size - size
            elif size > max_size:
                distance = size - max_size

            if distance < best_distance:
                best_distance = distance
                best_event = {
                    **event,
                    "attempt": attempt,
                }

            if min_size <= size <= max_size:
                return {
                    **event,
                    "attempt": attempt,
                }

        if best_event is None:
            raise RuntimeError("Unable to locate an example avalanche.")

        return best_event


def empirical_ccdf(values: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(list(values), dtype=int)
    array = array[array > 0]
    if array.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    sorted_values = np.sort(array)
    unique = np.unique(sorted_values)
    ccdf = np.array([(sorted_values >= value).mean() for value in unique], dtype=float)
    return unique, ccdf
