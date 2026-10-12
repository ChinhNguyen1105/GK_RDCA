from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from benchmark_data import (
    DATASET_SELECTIONS,
    expand_dataset_selection,
    load_benchmark_data,
)
from evaluate_mw_kmeans import (
    RESULTS_DIR,
    print_results,
    save_csv,
    save_plot,
    score_clustering,
)


def evaluate_dataset(
    dataset_name: str,
    *,
    n_init: int = 10,
    random_state: int = 42,
    max_iter: int = 300,
) -> dict[str, Any]:
    """Run classic K-means on one of the shared benchmark datasets."""
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    features, true_labels = load_benchmark_data(dataset_name)
    n_clusters = len(np.unique(true_labels))
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
        algorithm="lloyd",
    )

    start = time.perf_counter()
    with threadpool_limits(limits=1):
        predicted_clusters = model.fit_predict(features)
    runtime_seconds = time.perf_counter() - start

    return score_clustering(
        dataset_name,
        true_labels,
        predicted_clusters,
        n_features=features.shape[1],
        n_clusters=n_clusters,
        runtime_seconds=runtime_seconds,
        iterations=model.n_iter_,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate classic K-means on Iris, Wine, Breast Cancer, "
            "and OpenML Yeast."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_SELECTIONS,
        default="all",
        help="Dataset to evaluate, or all available datasets (default: all).",
    )
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "classic_kmeans_results.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=RESULTS_DIR / "classic_kmeans_results.png",
        help="Chart output path (PNG, SVG, or PDF).",
    )
    args = parser.parse_args()

    dataset_names = expand_dataset_selection(args.dataset)
    results = [
        evaluate_dataset(
            name,
            n_init=args.n_init,
            random_state=args.random_state,
            max_iter=args.max_iter,
        )
        for name in dataset_names
    ]
    print_results(results)
    save_csv(results, args.output)
    print(f"\nCSV exported to: {args.output}")
    save_plot(results, args.plot, title="Classic K-means evaluation metrics")
    print(f"Chart exported to: {args.plot}")


if __name__ == "__main__":
    main()