from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class GKRDCA:
    """Regularized DCA for Gaussian-kernel MSSC (GK-RDCA).

    Reference: Le Thi Hoai An, Le Hoai Minh, Pham Dinh Tao, "New and efficient
    DCA based algorithms for minimum sum-of-squares clustering",
    Pattern Recognition 47 (2014), Sections 4.1-4.4.

    The only unknown is the center matrix ``V`` (n_clusters x n_features),
    constrained to the box ``T`` spanned by the data (eq. 36). The objective is

        F_GK(V) = sum_k min_i f_ki(V),  f_ki(V) = 2 - 2 exp(-||x_k - v_i||^2 / (2 sigma^2))

    with the DC decomposition ``F_GK = G2 - H2`` (eq. 35) and the regularization
    ``(tau / 2) ||V||^2`` added to both components (eq. 43). One iteration is

        W^l in dH2(V^l),  V^{l+1}_i = Proj_{T_i}((W^l_i + tau V^l_i) / (n rho + tau)).

    Implementation decisions that are NOT stated verbatim in the paper:
    * the subgradient uses the full-sum form of h_ki (eq. 34/40), i.e. the block
      q != i of grad h_ki equals ``rho * v_q`` (see the note in ``_compute_subgradient_h2``);
    * ties in the active index set I_k are broken by the smallest center index;
    * the starting point is drawn uniformly in the box ``T`` (variant "Rand"),
      VNS initialization (paper Section 5) is not implemented;
    * the best of ``n_init`` runs is the one with the smallest ``F_GK``;
    * convergence is ``||V^{l+1} - V^l||_F <= tol``; reaching ``max_iter`` is
      reported as ``converged_ = False``;
    * labels are the nearest-center assignment (no prediction rule in the paper).

    The algorithm is a local method; no global optimality is claimed.
    """

    n_clusters: int
    sigma: float = 1.0
    rho: float | None = None
    tau: float = 0.1
    max_iter: int = 300
    tol: float = 1e-5
    n_init: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        self._validate_parameters()

    # ------------------------------------------------------------------ #
    # Group A - validation
    # ------------------------------------------------------------------ #
    def _validate_parameters(self) -> None:
        """Validate hyper-parameters against the constraints of the model."""
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        if not self.sigma > 0:
            raise ValueError("sigma must be positive")
        if not self.tau > 0:
            raise ValueError("tau must be positive")
        if self.rho is not None and self.rho < 2.0 / self.sigma**2:
            raise ValueError("rho must be at least 2 / sigma**2")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.n_init <= 0:
            raise ValueError("n_init must be positive")

    def _validate_input_data(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Convert X to a finite float matrix with at least n_clusters rows."""
        data_matrix = np.asarray(X, dtype=float)
        if data_matrix.ndim != 2:
            raise ValueError("X must be a 2D array-like object")
        n_samples, n_features = data_matrix.shape
        if n_samples == 0 or n_features == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if n_samples < self.n_clusters:
            raise ValueError(
                "Number of samples must be greater than or equal to n_clusters"
            )
        if not np.isfinite(data_matrix).all():
            raise ValueError("X must contain only finite values")
        return data_matrix

    def _resolve_rho(self) -> float:
        """Return rho; the paper's experiments use rho = 2 / sigma^2."""
        return 2.0 / self.sigma**2 if self.rho is None else float(self.rho)

    # ------------------------------------------------------------------ #
    # Group B - box T and initialization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_box_bounds(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (zeta, beta): per-feature min and max of the data (Sec. 4.2)."""
        return X.min(axis=0), X.max(axis=0)

    def _initialize_centers(
        self,
        rng: np.random.Generator,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        """Draw n_clusters centers uniformly in the box T (variant 'Rand')."""
        return rng.uniform(
            lower_bounds,
            upper_bounds,
            size=(self.n_clusters, lower_bounds.shape[0]),
        )

    # ------------------------------------------------------------------ #
    # Group C - mathematical operations
    # ------------------------------------------------------------------ #
    @staticmethod
    def _squared_distances(X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Return the (n, c) matrix of squared distances ||x_k - v_i||^2."""
        x_squared_norms = np.einsum("ij,ij->i", X, X)
        v_squared_norms = np.einsum("ij,ij->i", V, V)
        squared_distances = X @ V.T
        squared_distances *= -2.0
        squared_distances += x_squared_norms[:, None]
        squared_distances += v_squared_norms[None, :]
        np.maximum(squared_distances, 0.0, out=squared_distances)
        return squared_distances

    def _compute_kernel_distances(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Return f_ki(V) = 2 - 2 exp(-||x_k - v_i||^2 / (2 sigma^2)), shape (n, c).

        ``-2 * expm1(-t)`` equals ``2 - 2 * exp(-t)`` and is accurate for small t.
        """
        squared_distances = self._squared_distances(X, V)
        return -2.0 * np.expm1(-squared_distances / (2.0 * self.sigma**2))

    def _compute_objective(self, X: np.ndarray, V: np.ndarray) -> float:
        """Return F_GK(V) = sum_k min_i f_ki(V) (eq. 31)."""
        return float(self._compute_kernel_distances(X, V).min(axis=1).sum())

    def _assign_nearest_centers(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Return i*(k) in I_k: nearest center, ties -> smallest index.

        argmax_i h_ki = argmin_i f_ki = argmin_i ||x_k - v_i|| because the first
        term of h_ki does not depend on i (D1).
        """
        return np.argmin(self._squared_distances(X, V), axis=1)

    def _compute_g2(self, V: np.ndarray, n_samples: int) -> float:
        """Return G2(V) + (tau/2)||V||^2 with G2 = (n rho / 2) sum_i ||v_i||^2."""
        squared_norm = float(np.sum(V * V))
        return 0.5 * (n_samples * self._resolve_rho() + self.tau) * squared_norm

    def _compute_h2(self, X: np.ndarray, V: np.ndarray) -> float:
        """Return H2(V) + (tau/2)||V||^2 with H2 = sum_k max_i h_ki (eq. 35, 40).

        h_ki(V) = (rho / 2) sum_l ||v_l||^2 - f_ki(V).
        """
        squared_norm = float(np.sum(V * V))
        h_values = 0.5 * self._resolve_rho() * squared_norm - self._compute_kernel_distances(X, V)
        return float(h_values.max(axis=1).sum()) + 0.5 * self.tau * squared_norm

    def _compute_subgradient_h2(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Return W in dH2(V), shape (c, d), with the one-hot choice of lambda.

        For active index i = i*(k):
            [grad h_ki]_q = rho v_q - [q == i] (2/sigma^2) e_ki (v_i - x_k),
            e_ki = exp(-||x_k - v_i||^2 / (2 sigma^2)).
        The block q != i is ``rho v_q`` because h_ki is defined with the full sum
        over l (eq. 34/40); eq. (42) lists only the nonzero block i. Summing over k:

            W_q = n rho v_q - (2/sigma^2) sum_{k: i*(k)=q} e_kq (v_q - x_k).
        """
        n_samples = X.shape[0]
        squared_distances = self._squared_distances(X, V)
        active_indices = np.argmin(squared_distances, axis=1)
        active_squared = squared_distances[np.arange(n_samples), active_indices]
        kernel_values = np.exp(-active_squared / (2.0 * self.sigma**2))

        weight_sums = np.bincount(
            active_indices, weights=kernel_values, minlength=self.n_clusters
        )
        weighted_points = np.zeros_like(V)
        np.add.at(weighted_points, active_indices, kernel_values[:, None] * X)

        kernel_term = (2.0 / self.sigma**2) * (
            weight_sums[:, None] * V - weighted_points
        )
        return n_samples * self._resolve_rho() * V - kernel_term

    @staticmethod
    def _project_onto_box(
        V: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        """Return Proj_T(V): coordinate-wise clipping onto the box T."""
        return np.clip(V, lower_bounds, upper_bounds)

    def _update_centers(
        self,
        X: np.ndarray,
        V: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        """One GK-RDCA step: V+ = Proj_T((W + tau V) / (n rho + tau)) (eq. 39, Sec. 4.4)."""
        n_samples = X.shape[0]
        subgradient = self._compute_subgradient_h2(X, V)
        scale = n_samples * self._resolve_rho() + self.tau
        return self._project_onto_box(
            (subgradient + self.tau * V) / scale, lower_bounds, upper_bounds
        )

    def _check_convergence(self, V_next: np.ndarray, V: np.ndarray) -> bool:
        """Return True iff ||V^{l+1} - V^l||_F <= tol (stopping rule of GK-DCA)."""
        return bool(np.linalg.norm(V_next - V) <= self.tol)

    # ------------------------------------------------------------------ #
    # Group D - main loop
    # ------------------------------------------------------------------ #
    def _run_single(
        self,
        X: np.ndarray,
        V_start: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[np.ndarray, int, bool, list[float]]:
        """Run GK-RDCA from V_start; return (V, n_iter, converged, history)."""
        V = V_start
        history = [self._compute_objective(X, V)]
        converged = False
        iteration = 0
        for iteration in range(1, self.max_iter + 1):
            V_next = self._update_centers(X, V, lower_bounds, upper_bounds)
            if not np.isfinite(V_next).all():
                raise FloatingPointError("GK-RDCA produced non-finite centers")
            converged = self._check_convergence(V_next, V)
            V = V_next
            history.append(self._compute_objective(X, V))
            if converged:
                break
        return V, iteration, converged, history

    def fit(self, X: Iterable[Iterable[float]]) -> "GKRDCA":
        """Run n_init GK-RDCA starts and keep the one with the smallest F_GK."""
        data_matrix = self._validate_input_data(X)
        lower_bounds, upper_bounds = self._compute_box_bounds(data_matrix)
        rng = np.random.default_rng(self.random_state)

        best_run: tuple[np.ndarray, int, bool, list[float]] | None = None
        best_objective = np.inf
        for _ in range(self.n_init):
            V_start = self._initialize_centers(rng, lower_bounds, upper_bounds)
            run = self._run_single(data_matrix, V_start, lower_bounds, upper_bounds)
            objective_value = run[3][-1]
            if not np.isfinite(objective_value):
                raise FloatingPointError("GK-RDCA produced a non-finite objective")
            if objective_value < best_objective:
                best_run, best_objective = run, objective_value

        if best_run is None:
            raise RuntimeError("GK-RDCA failed to produce a solution")

        centers, n_iter, converged, history = best_run
        labels = self._assign_nearest_centers(data_matrix, centers)
        empty_clusters = int(
            self.n_clusters - np.count_nonzero(np.bincount(labels, minlength=self.n_clusters))
        )
        if empty_clusters > 0:
            warnings.warn(
                f"GK-RDCA finished with {empty_clusters} empty cluster(s)",
                RuntimeWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = centers
        self.labels_ = labels
        self.n_iter_ = int(n_iter)
        self.objective_value_ = float(best_objective)
        self.objective_history_ = history
        self.converged_ = bool(converged)
        self.empty_clusters_ = empty_clusters
        self.rho_ = self._resolve_rho()
        return self

    def fit_predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Fit the model and return nearest-center labels of the training data."""
        self.fit(X)
        return self.labels_

    def predict(self, X: Iterable[Iterable[float]]) -> np.ndarray:
        """Assign each row of X to its nearest fitted center (implementation rule)."""
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("The model has not been fitted yet")
        data_matrix = np.asarray(X, dtype=float)
        if data_matrix.ndim != 2:
            raise ValueError("X must be a 2D array-like object")
        if data_matrix.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError("X must have the same number of features as the training data")
        if not np.isfinite(data_matrix).all():
            raise ValueError("X must contain only finite values")
        return self._assign_nearest_centers(data_matrix, self.cluster_centers_)


__all__ = ["GKRDCA"]