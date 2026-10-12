from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_PAIRWISE_CHUNK_SIZE = 256


@dataclass
class MWKMeans:
    """MW-K-means based on the paper "K-means Clustering Algorithm and Its Improvement Research".

    The algorithm combines three improvements over classic K-means:
    1. density-based preprocessing to remove low-density/noisy points,
    2. maximum-minimum distance initialization to spread initial centers,
    3. coefficient-of-variation weighting and weighted Euclidean distance.
    """

    n_clusters: int
    beta: float = 0.05
    theta: float = 0.5
    max_iter: int = 300
    tol: float = 1e-5
    remove_noise: bool = True
    n_init: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if not 0 < self.beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if self.theta <= 0:
            raise ValueError("theta must be positive")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        return self._pairwise_distances_between(X, X)

    def _pairwise_distances_between(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        x_squared_norms = np.einsum("ij,ij->i", X, X)
        y_squared_norms = np.einsum("ij,ij->i", Y, Y)
        distances_squared = X @ Y.T
        distances_squared *= -2
        distances_squared += x_squared_norms[:, None]
        distances_squared += y_squared_norms[None, :]
        np.maximum(distances_squared, 0, out=distances_squared)
        np.sqrt(distances_squared, out=distances_squared)
        return distances_squared

    def _mean_pairwise_distance(self, X: np.ndarray) -> float:
        n_samples = X.shape[0]
        pair_count = n_samples * (n_samples - 1) // 2
        if pair_count == 0:
            return 0.0

        total_distance = 0.0
        for row_start in range(0, n_samples, _PAIRWISE_CHUNK_SIZE):
            row_stop = min(row_start + _PAIRWISE_CHUNK_SIZE, n_samples)
            row_block = X[row_start:row_stop]

            diagonal_distances = self._pairwise_distances_between(
                row_block,
                row_block,
            )
            upper_rows, upper_columns = np.triu_indices(
                row_block.shape[0],
                k=1,
            )
            total_distance += float(
                diagonal_distances[upper_rows, upper_columns].sum()
            )

            for column_start in range(row_stop, n_samples, _PAIRWISE_CHUNK_SIZE):
                column_stop = min(
                    column_start + _PAIRWISE_CHUNK_SIZE,
                    n_samples,
                )
                total_distance += float(
                    self._pairwise_distances_between(
                        row_block,
                        X[column_start:column_stop],
                    ).sum()
                )

        return total_distance / pair_count

    def _density_counts(self, X: np.ndarray, epsilon: float) -> np.ndarray:
        density = np.empty(X.shape[0], dtype=int)
        for start in range(0, X.shape[0], _PAIRWISE_CHUNK_SIZE):
            stop = min(start + _PAIRWISE_CHUNK_SIZE, X.shape[0])
            distance_chunk = self._pairwise_distances_between(X[start:stop], X)
            density[start:stop] = np.count_nonzero(
                distance_chunk <= epsilon,
                axis=1,
            ) - 1
        return density

    def _infer_density_params(self, X: np.ndarray) -> tuple[float, int]:
        n_samples = X.shape[0]
        if n_samples <= 1:
            return 0.0, 1

        avg_distance = self._mean_pairwise_distance(X)
        min_pts = max(1, int(np.ceil(self.beta * n_samples)))
        epsilon = self.theta * avg_distance
        return epsilon, min_pts

    def _compute_weights(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.ones(X.shape[1], dtype=float) / max(1, X.shape[1])

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        coefficient_of_variation = np.divide(
            std,
            np.abs(mean),
            out=np.zeros_like(std, dtype=float),
            where=np.abs(mean) > 1e-12,
        )

        total_cv = coefficient_of_variation.sum()
        if total_cv <= 0:
            return np.ones(X.shape[1], dtype=float) / X.shape[1]
        return coefficient_of_variation / total_cv

    def _select_initial_centers(
        self,
        X: np.ndarray,
        density: np.ndarray,
        first_center_index: int | None = None,
    ) -> np.ndarray:
        first_idx = (
            int(np.argmax(density))
            if first_center_index is None
            else first_center_index
        )
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        centers[0] = X[first_idx]
        nearest_to_any_center = np.linalg.norm(X - centers[0], axis=1)

        for center_index in range(1, self.n_clusters):
            next_idx = int(np.argmax(nearest_to_any_center))

            if any(
                np.allclose(X[next_idx], center)
                for center in centers[:center_index]
            ):
                ranked = np.argsort(nearest_to_any_center)[::-1]
                for idx in ranked:
                    if not any(
                        np.allclose(X[idx], center)
                        for center in centers[:center_index]
                    ):
                        next_idx = int(idx)
                        break

            centers[center_index] = X[next_idx]
            distances_to_new_center = np.linalg.norm(
                X - centers[center_index],
                axis=1,
            )
            np.minimum(
                nearest_to_any_center,
                distances_to_new_center,
                out=nearest_to_any_center,
            )

        return centers

    def _weighted_distance(self, X: np.ndarray, center: np.ndarray, weights: np.ndarray) -> np.ndarray:
        diff = X - center
        return np.sqrt(np.sum(weights[None, :] * diff * diff, axis=1))

    def _weighted_distance_matrix(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        scale = np.sqrt(weights)
        weighted_X = X * scale
        weighted_centers = centers * scale
        return self._squared_distance_matrix(weighted_X, weighted_centers)

    @staticmethod
    def _squared_distance_matrix(
        X: np.ndarray,
        centers: np.ndarray,
        squared_norms: np.ndarray | None = None,
    ) -> np.ndarray:
        if squared_norms is None:
            squared_norms = np.einsum("ij,ij->i", X, X)
        center_squared_norms = np.einsum("ij,ij->i", centers, centers)
        distances_squared = X @ centers.T
        distances_squared *= -2
        distances_squared += squared_norms[:, None]
        distances_squared += center_squared_norms[None, :]
        np.maximum(distances_squared, 0, out=distances_squared)
        return distances_squared

    def fit(self, X: Iterable[Iterable[float]]) -> "MWKMeans":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array-like object")
        n_samples, n_features = X.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if n_samples < self.n_clusters:
            raise ValueError("Number of samples must be greater than or equal to n_clusters")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        epsilon, min_pts = self._infer_density_params(X)
        density = self._density_counts(X, epsilon)
        dense_mask = density >= min_pts

        if self.remove_noise:
            working_data = X[dense_mask]
        else:
            working_data = X.copy()

        if working_data.shape[0] == 0 or working_data.shape[0] < self.n_clusters:
            working_data = X.copy()
            dense_mask = np.ones(n_samples, dtype=bool)

        weights = self._compute_weights(working_data)
        weight_scale = np.sqrt(weights)
        weighted_working_data = working_data * weight_scale
        weighted_data_squared_norms = np.einsum(
            "ij,ij->i",
            weighted_working_data,
            weighted_working_data,
        )
        initialization_density = (
            density if not self.remove_noise else density[dense_mask]
        )
        rng = np.random.default_rng(self.random_state)
        first_center_indices = np.empty(self.n_init, dtype=int)
        first_center_indices[0] = int(np.argmax(initialization_density))
        if self.n_init > 1:
            first_center_indices[1:] = rng.integers(
                0,
                working_data.shape[0],
                size=self.n_init - 1,
            )

        best_centers: np.ndarray | None = None
        best_assignments: np.ndarray | None = None
        best_inertia = np.inf
        best_iteration = 0
        for first_center_index in first_center_indices:
            centers = self._select_initial_centers(
                working_data,
                initialization_density,
                int(first_center_index),
            )

            for iteration in range(1, self.max_iter + 1):
                prev_centers = centers.copy()
                distance_matrix = self._squared_distance_matrix(
                    weighted_working_data,
                    centers * weight_scale,
                    weighted_data_squared_norms,
                )
                assignments = np.argmin(distance_matrix, axis=1)

                cluster_counts = np.bincount(
                    assignments,
                    minlength=self.n_clusters,
                )
                new_centers = centers.copy()
                non_empty_clusters = cluster_counts > 0
                for feature_index in range(n_features):
                    feature_sums = np.bincount(
                        assignments,
                        weights=working_data[:, feature_index],
                        minlength=self.n_clusters,
                    )
                    new_centers[non_empty_clusters, feature_index] = (
                        feature_sums[non_empty_clusters]
                        / cluster_counts[non_empty_clusters]
                    )

                centers = new_centers
                if np.max(np.linalg.norm(centers - prev_centers, axis=1)) < self.tol:
                    break

            final_distance_matrix = self._squared_distance_matrix(
                weighted_working_data,
                centers * weight_scale,
                weighted_data_squared_norms,
            )
            final_assignments = np.argmin(final_distance_matrix, axis=1)
            inertia = float(
                final_distance_matrix[
                    np.arange(working_data.shape[0]),
                    final_assignments,
                ].sum()
            )
            if inertia < best_inertia:
                best_centers = centers.copy()
                best_assignments = final_assignments.copy()
                best_inertia = inertia
                best_iteration = iteration

        if best_centers is None or best_assignments is None:
            raise RuntimeError("MW-K-means failed to produce a clustering solution")

        centers = best_centers
        final_assignments = best_assignments
        self.cluster_centers_ = centers
        self.n_iter_ = best_iteration
        self.inertia_ = best_inertia
        self.n_init_ = self.n_init
        self.weights_ = weights
        self.epsilon_ = epsilon
        self.min_pts_ = min_pts
        self.density_ = density

        if self.remove_noise:
            labels = np.full(n_samples, -1, dtype=int)
            labels[dense_mask] = final_assignments
        else:
            labels = final_assignments

        self.labels_ = labels
        return self

    def fit_predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        self.fit(X)
        return self.labels_

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("The model has not been fitted yet")

        distance_matrix = self._weighted_distance_matrix(
            X,
            self.cluster_centers_,
            self.weights_,
        )
        return np.argmin(distance_matrix, axis=1)


__all__ = ["MWKMeans"]