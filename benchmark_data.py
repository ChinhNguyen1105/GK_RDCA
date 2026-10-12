from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.datasets import (
    fetch_openml,
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.preprocessing import MinMaxScaler


DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "cancer": load_breast_cancer,
}
DATASET_NAMES = (*DATASET_LOADERS, "yeast")
DATASET_SELECTIONS = (*DATASET_NAMES, "all")


def expand_dataset_selection(selection: str) -> tuple[str, ...]:
    """Expand the all-datasets CLI option or validate one dataset name."""
    if selection == "all":
        return DATASET_NAMES
    if selection in DATASET_NAMES:
        return (selection,)
    raise ValueError(
        f"Unknown dataset {selection!r}; choose from {', '.join(DATASET_SELECTIONS)}"
    )


def load_benchmark_data(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load and scale a built-in or OpenML benchmark dataset."""
    if dataset_name == "yeast":
        dataset = fetch_openml(
            data_id=181,
            as_frame=False,
            parser="auto",
        )
        raw_features = np.asarray(dataset.data, dtype=np.float64)
        labels = np.asarray(dataset.target)
        if raw_features.ndim != 2 or raw_features.shape != (1484, 8):
            raise ValueError(
                "OpenML dataset 181 (yeast) must contain 1,484 samples "
                "and 8 numeric features"
            )
        if labels.shape != (raw_features.shape[0],):
            raise ValueError("OpenML dataset 181 (yeast) has invalid target labels")
        if not np.isfinite(raw_features).all():
            raise ValueError("OpenML dataset 181 (yeast) contains non-finite values")
        return MinMaxScaler().fit_transform(raw_features), labels

    try:
        loader = DATASET_LOADERS[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}; choose from {', '.join(DATASET_NAMES)}"
        ) from exc

    dataset: Any = loader()
    features = MinMaxScaler().fit_transform(dataset.data)
    return features, np.asarray(dataset.target)