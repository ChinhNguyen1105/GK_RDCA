from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    accuracy_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

from benchmark_data import (
    DATASET_SELECTIONS,
    expand_dataset_selection,
    load_benchmark_data,
)
from mw_kmeans import MWKMeans


METRIC_FIELDS = (
    "dataset",
    "samples",
    "features",
    "clusters",
    "accuracy_percent",
    "precision_macro_percent",
    "recall_macro_percent",
    "f1_macro_percent",
    "ari",
    "nmi",
    "noise_points",
    "runtime_seconds",
    "iterations",
)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def evaluate_dataset(
    dataset_name: str,
    *,
    beta: float = 0.05,
    theta: float = 0.5,
    max_iter: int = 300,
    n_init: int = 10,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit MW-K-means on a benchmark dataset and return clustering metrics."""
    features, true_labels = load_benchmark_data(dataset_name)
    model = MWKMeans(
        n_clusters=len(np.unique(true_labels)),
        beta=beta,
        theta=theta,
        max_iter=max_iter,
        remove_noise=False,
        n_init=n_init,
        random_state=random_state,
    )

    start = time.perf_counter()
    predicted_clusters = model.fit_predict(features)
    runtime_seconds = time.perf_counter() - start

    return score_clustering(
        dataset_name,
        true_labels,
        predicted_clusters,
        n_features=features.shape[1],
        n_clusters=model.n_clusters,
        runtime_seconds=runtime_seconds,
        iterations=model.n_iter_,
    )


def score_clustering(
    dataset_name: str,
    true_labels: np.ndarray,
    predicted_clusters: np.ndarray,
    *,
    n_features: int,
    n_clusters: int,
    runtime_seconds: float,
    iterations: int,
) -> dict[str, Any]:
    """Calculate the shared evaluation metrics for a clustering result."""
    true_classes, encoded_targets = np.unique(true_labels, return_inverse=True)
    cluster_ids, encoded_clusters = np.unique(predicted_clusters, return_inverse=True)
    contingency = np.zeros((len(true_classes), len(cluster_ids)), dtype=np.int64)
    np.add.at(contingency, (encoded_targets, encoded_clusters), 1)
    class_indices, cluster_indices = linear_sum_assignment(-contingency)
    cluster_to_class = {
        cluster_index: class_index
        for class_index, cluster_index in zip(class_indices, cluster_indices)
    }
    unmatched_label = len(true_classes)
    aligned_predictions = np.asarray(
        [
            cluster_to_class.get(cluster_index, unmatched_label + cluster_index)
            for cluster_index in encoded_clusters
        ]
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        encoded_targets,
        aligned_predictions,
        labels=np.arange(len(true_classes)),
        average="macro",
        zero_division=0,
    )

    return {
        "dataset": dataset_name,
        "samples": int(true_labels.shape[0]),
        "features": int(n_features),
        "clusters": int(n_clusters),
        "accuracy_percent": 100.0 * accuracy_score(
            encoded_targets, aligned_predictions
        ),
        "precision_macro_percent": 100.0 * precision,
        "recall_macro_percent": 100.0 * recall,
        "f1_macro_percent": 100.0 * f1,
        "ari": float(adjusted_rand_score(true_labels, predicted_clusters)),
        "nmi": float(
            normalized_mutual_info_score(true_labels, predicted_clusters)
        ),
        "noise_points": int(np.count_nonzero(predicted_clusters == -1)),
        "runtime_seconds": runtime_seconds,
        "iterations": int(iterations),
    }


def print_results(results: list[dict[str, Any]]) -> None:
    """Print a compact, human-readable summary of benchmark results."""
    headers = (
        ("dataset", "Dataset"),
        ("samples", "Samples"),
        ("features", "Features"),
        ("clusters", "Clusters"),
        ("accuracy_percent", "Accuracy %"),
        ("precision_macro_percent", "Precision %"),
        ("recall_macro_percent", "Recall %"),
        ("f1_macro_percent", "F1 %"),
        ("ari", "ARI"),
        ("nmi", "NMI"),
        ("noise_points", "Noise"),
        ("runtime_seconds", "Time (s)"),
        ("iterations", "Iterations"),
    )
    rows = []
    for result in results:
        rows.append(
            [
                result[key]
                if key in {"dataset", "samples", "features", "clusters", "noise_points", "iterations"}
                else f"{result[key]:.4f}"
                if key in {"ari", "nmi", "runtime_seconds"}
                else f"{result[key]:.2f}"
                for key, _ in headers
            ]
        )

    widths = [
        max(len(title), *(len(str(row[index])) for row in rows))
        for index, (_, title) in enumerate(headers)
    ]
    print(" | ".join(title.ljust(widths[i]) for i, (_, title) in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))


def save_csv(
    results: list[dict[str, Any]],
    output_path: str | Path,
    *,
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Write benchmark results to a CSV file."""
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = METRIC_FIELDS

    with output_file_path.open(
        "w",
        newline="",
       encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(results)


def save_plot(
    results: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str = "MW-K-means evaluation metrics",
) -> None:
    """Save evaluation metrics and runtime as a chart image."""
    if not results:
        raise ValueError("At least one dataset result is required to create a plot")

    datasets = [result["dataset"] for result in results]
    metric_names = (
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ARI x100",
        "NMI x100",
    )
    metric_fields = (
        "accuracy_percent",
        "precision_macro_percent",
        "recall_macro_percent",
        "f1_macro_percent",
        "ari",
        "nmi",
    )
    metric_values = np.asarray(
        [[result[field] for field in metric_fields] for result in results],
        dtype=float,
    )
    metric_values[:, 4:] *= 100

    figure, (metrics_axis, runtime_axis) = plt.subplots(
        2, 1, figsize=(max(9, 2.3 * len(results)), 8)
    )
    figure.suptitle(title, fontsize=15)
    x_positions = np.arange(len(results))
    bar_width = 0.12
    for metric_index, metric_name in enumerate(metric_names):
        offset = (metric_index - (len(metric_names) - 1) / 2) * bar_width
        bars = metrics_axis.bar(
            x_positions + offset,
            metric_values[:, metric_index],
            bar_width,
            label=metric_name,
        )
        metrics_axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    metrics_axis.set_ylabel("Score (%)")
    metrics_axis.set_ylim(0, 115)
    metrics_axis.set_xticks(x_positions, datasets)
    metrics_axis.legend(
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
    )
    metrics_axis.grid(axis="y", linestyle=":", alpha=0.5)

    runtimes = [result["runtime_seconds"] for result in results]
    runtime_bars = runtime_axis.bar(x_positions, runtimes, color="slateblue")
    runtime_axis.bar_label(runtime_bars, fmt="%.4f", padding=3)
    runtime_axis.set_title("Execution time")
    runtime_axis.set_ylabel("Seconds")
    runtime_axis.set_ylim(0, max(runtimes, default=0.0) * 1.2 or 1.0)
    runtime_axis.set_xticks(x_positions, datasets)
    runtime_axis.grid(axis="y", linestyle=":", alpha=0.5)

    figure.subplots_adjust(top=0.82, hspace=0.4)
    plot_path = Path(output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MW-K-means on Iris, Wine, Breast Cancer, and OpenML Yeast."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_SELECTIONS,
        default="all",
        help="Dataset to evaluate, or all available datasets (default: all).",
    )
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "results.csv",
        help=f"CSV output path (default: {RESULTS_DIR / 'results.csv'}).",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=RESULTS_DIR / "results.png",
        help=(
            "Chart output path (PNG, SVG, or PDF); "
            f"default: {RESULTS_DIR / 'results.png'}."
        ),
    )
    args = parser.parse_args()

    dataset_names = expand_dataset_selection(args.dataset)
    results = [
        evaluate_dataset(
            name,
            beta=args.beta,
            theta=args.theta,
            max_iter=args.max_iter,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        for name in dataset_names
    ]
    print_results(results)
    save_csv(results, args.output)
    print(f"\nCSV exported to: {args.output}")
    save_plot(results, args.plot)
    print(f"Chart exported to: {args.plot}")


if __name__ == "__main__":
    main()