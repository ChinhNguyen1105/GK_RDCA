import math
from types import SimpleNamespace

import numpy as np
import pytest

import benchmark_data
from evaluate_classic_kmeans import evaluate_dataset as evaluate_classic_dataset
from evaluate_gk_rdca import (
    GK_RDCA_EXTRA_FIELDS,
    GK_RDCA_FIELDS,
    evaluate_dataset,
    print_solver_details,
)
from evaluate_mw_kmeans import (
    METRIC_FIELDS,
    evaluate_dataset as evaluate_mw_dataset,
    save_csv,
    save_plot,
)
from gk_rdca import GKRDCA


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_blobs_unit_box(seed: int = 7, per_cluster: int = 40) -> np.ndarray:
    """Two well separated blobs inside [0, 1]^2."""
    rng = np.random.default_rng(seed)
    blob_a = rng.normal(loc=[0.15, 0.15], scale=0.03, size=(per_cluster, 2))
    blob_b = rng.normal(loc=[0.85, 0.85], scale=0.03, size=(per_cluster, 2))
    return np.vstack([blob_a, blob_b])


def make_three_blobs(seed: int = 21, per_cluster: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal(loc=[0.15, 0.15], scale=0.05, size=(per_cluster, 2)),
            rng.normal(loc=[0.85, 0.20], scale=0.05, size=(per_cluster, 2)),
            rng.normal(loc=[0.50, 0.85], scale=0.05, size=(per_cluster, 2)),
        ]
    )


def reference_step(X, V, sigma, rho, tau, lower, upper):
    """Loop-based transcription of eq. (39) with W + tau V and n rho + tau."""
    n_samples, n_features = X.shape
    n_clusters = V.shape[0]
    W = np.zeros_like(V)
    for q in range(n_clusters):
        W[q] = n_samples * rho * V[q]
    for k in range(n_samples):
        distances = [np.sum((X[k] - V[i]) ** 2) for i in range(n_clusters)]
        i_star = int(np.argmin(distances))
        e_ki = math.exp(-distances[i_star] / (2.0 * sigma**2))
        W[i_star] -= (2.0 / sigma**2) * e_ki * (V[i_star] - X[k])
    V_next = (W + tau * V) / (n_samples * rho + tau)
    return np.clip(V_next, lower, upper)


# --------------------------------------------------------------------------- #
# IN - input and parameter validation
# --------------------------------------------------------------------------- #
class TestInputValidation:
    def test_in1_rejects_empty_matrix(self):
        with pytest.raises(ValueError, match="at least one sample"):
            GKRDCA(n_clusters=2).fit(np.empty((0, 2)))

    def test_in1_rejects_zero_features(self):
        with pytest.raises(ValueError, match="at least one sample and one feature"):
            GKRDCA(n_clusters=2).fit(np.empty((3, 0)))

    def test_in2_rejects_non_two_dimensional_input(self):
        with pytest.raises(ValueError, match="2D"):
            GKRDCA(n_clusters=2).fit(np.array([0.0, 1.0, 2.0]))

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_in3_rejects_non_finite_values(self, bad_value):
        X = np.array([[0.0, 0.0], [1.0, bad_value], [2.0, 2.0]])
        with pytest.raises(ValueError, match="finite"):
            GKRDCA(n_clusters=2).fit(X)

    def test_in4_rejects_fewer_samples_than_clusters(self):
        with pytest.raises(ValueError, match="greater than or equal to n_clusters"):
            GKRDCA(n_clusters=3).fit(np.array([[0.0], [1.0]]))

    @pytest.mark.parametrize("n_clusters", [-1, 0, 1])
    def test_in5_rejects_invalid_n_clusters(self, n_clusters):
        with pytest.raises(ValueError, match="n_clusters"):
            GKRDCA(n_clusters=n_clusters)

    @pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan")])
    def test_in7_rejects_invalid_sigma(self, sigma):
        with pytest.raises(ValueError, match="sigma"):
            GKRDCA(n_clusters=2, sigma=sigma)

    def test_in8_rejects_rho_below_proposition_bound(self):
        with pytest.raises(ValueError, match="rho"):
            GKRDCA(n_clusters=2, sigma=2.0, rho=0.4)  # 2 / sigma^2 = 0.5

    def test_in8_accepts_rho_at_and_above_bound(self):
        GKRDCA(n_clusters=2, sigma=2.0, rho=0.5)
        GKRDCA(n_clusters=2, sigma=2.0, rho=3.0)

    def test_in8_default_rho_is_two_over_sigma_squared(self):
        assert GKRDCA(n_clusters=2, sigma=2.0)._resolve_rho() == pytest.approx(0.5)
        assert GKRDCA(n_clusters=2, sigma=0.5)._resolve_rho() == pytest.approx(8.0)

    @pytest.mark.parametrize("tau", [0.0, -0.1])
    def test_in9_rejects_non_positive_tau(self, tau):
        with pytest.raises(ValueError, match="tau"):
            GKRDCA(n_clusters=2, tau=tau)

    def test_in10_rejects_invalid_stopping_parameters(self):
        with pytest.raises(ValueError, match="max_iter"):
            GKRDCA(n_clusters=2, max_iter=0)
        with pytest.raises(ValueError, match="tol"):
            GKRDCA(n_clusters=2, tol=-1e-9)
        with pytest.raises(ValueError, match="n_init must be positive"):
            GKRDCA(n_clusters=2, n_init=0)


# --------------------------------------------------------------------------- #
# OP - mathematical operations
# --------------------------------------------------------------------------- #
class TestOperations:
    def test_op1_kernel_distance_matches_hand_value(self):
        model = GKRDCA(n_clusters=2, sigma=1.0)
        X = np.array([[0.0, 0.0]])
        V = np.array([[1.0, 0.0], [0.0, 0.0]])

        f = model._compute_kernel_distances(X, V)

        assert f.shape == (1, 2)
        assert f[0, 0] == pytest.approx(2.0 - 2.0 * math.exp(-0.5))
        assert f[0, 0] == pytest.approx(0.786938680574733)
        assert f[0, 1] == pytest.approx(0.0, abs=1e-15)

    def test_op1_kernel_distance_uses_squared_distance_and_sigma(self):
        model = GKRDCA(n_clusters=2, sigma=2.0)
        X = np.array([[0.0, 0.0]])
        V = np.array([[3.0, 4.0], [0.0, 1.0]])  # distances 5 and 1

        f = model._compute_kernel_distances(X, V)

        assert f[0, 0] == pytest.approx(2.0 - 2.0 * math.exp(-25.0 / 8.0))
        assert f[0, 1] == pytest.approx(2.0 - 2.0 * math.exp(-1.0 / 8.0))

    def test_op1_kernel_distance_is_in_zero_two_and_finite(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 4))
        V = rng.normal(size=(5, 4)) * 10
        f = GKRDCA(n_clusters=5, sigma=0.7)._compute_kernel_distances(X, V)

        assert f.shape == (30, 5)
        assert np.isfinite(f).all()
        assert (f >= 0).all() and (f <= 2.0).all()

    def test_op2_objective_matches_hand_value(self):
        model = GKRDCA(n_clusters=2, sigma=1.0)
        X = np.array([[0.0, 0.0], [2.0, 0.0]])
        V = np.array([[1.0, 0.0], [5.0, 0.0]])
        # both points have squared distance 1 to center 0 (the nearest one)
        expected = 2.0 * (2.0 - 2.0 * math.exp(-0.5))

        assert model._compute_objective(X, V) == pytest.approx(expected)

    def test_op2_objective_is_zero_when_centers_coincide_with_points(self):
        model = GKRDCA(n_clusters=2, sigma=1.0)
        X = np.array([[0.0, 0.0], [2.0, 0.0]])

        assert model._compute_objective(X, X.copy()) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("tau", [0.1, 0.5])
    @pytest.mark.parametrize("rho_factor", [1.0, 2.5])
    def test_op3_dc_decomposition_reproduces_objective(self, tau, rho_factor):
        """G2 - H2 = F_GK (eq. 35), also after adding (tau/2)||V||^2 to both parts."""
        rng = np.random.default_rng(3)
        sigma = 0.8
        model = GKRDCA(
            n_clusters=3, sigma=sigma, tau=tau, rho=rho_factor * 2.0 / sigma**2
        )
        X = rng.random((25, 3))
        lower, upper = model._compute_box_bounds(X)
        for _ in range(5):
            V = model._initialize_centers(rng, lower, upper)
            difference = model._compute_g2(V, X.shape[0]) - model._compute_h2(X, V)
            assert difference == pytest.approx(
                model._compute_objective(X, V), rel=1e-9, abs=1e-9
            )

    def test_op4_active_index_is_nearest_center(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(40, 3))
        V = rng.normal(size=(4, 3))
        expected = np.array(
            [np.argmin(np.linalg.norm(V - x, axis=1)) for x in X]
        )

        np.testing.assert_array_equal(
            GKRDCA(n_clusters=4)._assign_nearest_centers(X, V), expected
        )

    def test_op4_ties_choose_smallest_center_index(self):
        X = np.array([[1.0, 0.0]])
        V = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.0]])

        index = GKRDCA(n_clusters=3)._assign_nearest_centers(X, V)

        np.testing.assert_array_equal(index, [0])

    def test_op4_active_index_maximises_h_ki(self):
        """Direct check of D1: argmax_i h_ki = argmin_i f_ki."""
        rng = np.random.default_rng(8)
        model = GKRDCA(n_clusters=3, sigma=0.9)
        X = rng.random((20, 2))
        V = rng.random((3, 2))
        h_values = 0.5 * model._resolve_rho() * np.sum(V * V) - model._compute_kernel_distances(X, V)

        np.testing.assert_array_equal(
            np.argmax(h_values, axis=1), model._assign_nearest_centers(X, V)
        )

    @pytest.mark.parametrize("tau", [0.1, 0.2])
    def test_op5_subgradient_inequality_for_regularized_h2(self, tau):
        """H2r(V') >= H2r(V) + <W + tau V, V' - V>, with H2r = H2 + (tau/2)||V||^2."""
        rng = np.random.default_rng(11)
        sigma = 0.6
        model = GKRDCA(n_clusters=3, sigma=sigma, tau=tau)
        X = rng.random((30, 2))
        lower, upper = model._compute_box_bounds(X)
        for _ in range(50):
            V = model._initialize_centers(rng, lower, upper)
            V_other = rng.uniform(-1.0, 2.0, size=V.shape)  # also outside the box
            W = model._compute_subgradient_h2(X, V)
            lhs = model._compute_h2(X, V_other)
            rhs = model._compute_h2(X, V) + np.sum((W + tau * V) * (V_other - V))
            assert lhs >= rhs - 1e-8 * max(1.0, abs(lhs))

    def test_op5_h2_midpoint_convexity(self):
        """H2 convex (needs rho >= 2/sigma^2): H2(mid) <= mean of endpoint values."""
        rng = np.random.default_rng(12)
        model = GKRDCA(n_clusters=3, sigma=0.5)
        X = rng.random((30, 2))
        for _ in range(50):
            V_a = rng.uniform(-1.0, 2.0, size=(3, 2))
            V_b = rng.uniform(-1.0, 2.0, size=(3, 2))
            mid = model._compute_h2(X, 0.5 * (V_a + V_b))
            mean = 0.5 * (model._compute_h2(X, V_a) + model._compute_h2(X, V_b))
            assert mid <= mean + 1e-8 * max(1.0, abs(mean))

    def test_op5_subgradient_shape_and_finiteness(self):
        rng = np.random.default_rng(1)
        X = rng.random((17, 4))
        V = rng.random((3, 4))
        W = GKRDCA(n_clusters=3)._compute_subgradient_h2(X, V)

        assert W.shape == (3, 4)
        assert np.isfinite(W).all()

    def test_op6_projection_stays_in_box_is_idempotent_and_nearest(self):
        rng = np.random.default_rng(2)
        lower = np.array([0.0, -1.0, 2.0])
        upper = np.array([1.0, 1.0, 3.0])
        V = rng.uniform(-5.0, 5.0, size=(4, 3))

        projected = GKRDCA._project_onto_box(V, lower, upper)

        assert projected.shape == V.shape
        assert (projected >= lower).all() and (projected <= upper).all()
        np.testing.assert_array_equal(
            GKRDCA._project_onto_box(projected, lower, upper), projected
        )
        for _ in range(100):
            other = rng.uniform(lower, upper, size=V.shape)
            assert np.linalg.norm(V - projected) <= np.linalg.norm(V - other) + 1e-12

    def test_op6_box_bounds_are_feature_min_and_max(self):
        X = np.array([[0.0, 5.0], [2.0, -1.0], [1.0, 3.0]])
        lower, upper = GKRDCA._compute_box_bounds(X)

        np.testing.assert_array_equal(lower, [0.0, -1.0])
        np.testing.assert_array_equal(upper, [2.0, 5.0])

    def test_op7_update_matches_hand_computation(self):
        # n=2, d=1, c=2, sigma=1 -> rho=2, n*rho=4, tau=0.1, scale=4.1, T=[0,1]
        model = GKRDCA(n_clusters=2, sigma=1.0, tau=0.1)
        X = np.array([[0.0], [1.0]])
        V = np.array([[0.2], [0.9]])
        e_0 = math.exp(-0.04 / 2.0)  # point 0 -> center 0, d^2 = 0.04
        e_1 = math.exp(-0.01 / 2.0)  # point 1 -> center 1, d^2 = 0.01
        W_0 = 4.0 * 0.2 - 2.0 * e_0 * (0.2 - 0.0)
        W_1 = 4.0 * 0.9 - 2.0 * e_1 * (0.9 - 1.0)
        expected = np.array([[(W_0 + 0.1 * 0.2) / 4.1], [(W_1 + 0.1 * 0.9) / 4.1]])

        updated = model._update_centers(X, V, np.array([0.0]), np.array([1.0]))

        np.testing.assert_allclose(updated, expected, rtol=1e-12)
        assert expected[0, 0] == pytest.approx(0.104371, abs=1e-6)
        assert expected[1, 0] == pytest.approx(0.948537, abs=1e-6)

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 3.0])
    @pytest.mark.parametrize("tau", [0.1, 0.2])
    def test_op7_update_matches_loop_reference(self, sigma, tau):
        rng = np.random.default_rng(4)
        model = GKRDCA(n_clusters=3, sigma=sigma, tau=tau)
        X = rng.random((40, 3))
        lower, upper = model._compute_box_bounds(X)
        V = model._initialize_centers(rng, lower, upper)

        updated = model._update_centers(X, V, lower, upper)
        expected = reference_step(
            X, V, sigma, model._resolve_rho(), tau, lower, upper
        )

        np.testing.assert_allclose(updated, expected, rtol=1e-10, atol=1e-12)

    def test_op7_update_result_is_projected_onto_box(self):
        model = GKRDCA(n_clusters=2, sigma=1.0, tau=0.1)
        X = np.array([[0.0], [1.0]])
        V = np.array([[-50.0], [50.0]])  # start far outside T

        updated = model._update_centers(X, V, np.array([0.0]), np.array([1.0]))

        assert (updated >= 0.0).all() and (updated <= 1.0).all()

    def test_op7_empty_cluster_row_is_a_fixed_point(self):
        """No active point -> W_q = n rho v_q -> v_q unchanged (before projection)."""
        model = GKRDCA(n_clusters=2, sigma=1.0, tau=0.1)
        X = np.array([[0.0], [0.2], [0.4]])
        V = np.array([[0.2], [0.2]])  # tie: every point -> center 0, center 1 empty

        updated = model._update_centers(X, V, np.array([0.0]), np.array([0.4]))

        assert updated[1, 0] == pytest.approx(0.2, abs=1e-15)

    @pytest.mark.parametrize("sigma", [0.7, 2.0])
    def test_op8_subgradient_matches_finite_difference_gradient(self, sigma):
        """n rho V - W = grad F_GK when there is no tie (distinguishes the M1 choice)."""
        rng = np.random.default_rng(9)
        model = GKRDCA(n_clusters=3, sigma=sigma)
        X = rng.random((25, 2))
        V = rng.random((3, 2))
        n_samples = X.shape[0]
        analytic = n_samples * model._resolve_rho() * V - model._compute_subgradient_h2(X, V)

        numeric = np.zeros_like(V)
        step = 1e-6
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_plus, V_minus = V.copy(), V.copy()
                V_plus[i, j] += step
                V_minus[i, j] -= step
                numeric[i, j] = (
                    model._compute_objective(X, V_plus)
                    - model._compute_objective(X, V_minus)
                ) / (2.0 * step)

        np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)

    def test_check_convergence_uses_frobenius_norm_and_inclusive_tolerance(self):
        model = GKRDCA(n_clusters=2, tol=0.5)
        V = np.zeros((2, 2))
        step = np.array([[0.3, 0.0], [0.0, 0.4]])  # Frobenius norm = 0.5

        assert model._check_convergence(V + step, V) is True
        assert model._check_convergence(V + 1.01 * step, V) is False

    def test_initial_centers_are_inside_box_and_reproducible(self):
        model = GKRDCA(n_clusters=4)
        lower = np.array([0.0, -2.0])
        upper = np.array([1.0, 3.0])

        first = model._initialize_centers(np.random.default_rng(5), lower, upper)
        second = model._initialize_centers(np.random.default_rng(5), lower, upper)

        assert first.shape == (4, 2)
        assert (first >= lower).all() and (first <= upper).all()
        np.testing.assert_array_equal(first, second)


# --------------------------------------------------------------------------- #
# AL - whole algorithm
# --------------------------------------------------------------------------- #
class TestAlgorithm:
    def test_al1_separates_two_blobs(self):
        X = make_blobs_unit_box()
        model = GKRDCA(n_clusters=2, sigma=1.0, tau=0.1, n_init=10, random_state=0)

        labels = model.fit_predict(X)

        assert labels.shape == (X.shape[0],)
        assert set(labels.tolist()) == {0, 1}
        assert len(set(labels[:40].tolist())) == 1
        assert len(set(labels[40:].tolist())) == 1
        assert labels[0] != labels[-1]
        assert model.cluster_centers_.shape == (2, 2)
        assert model.empty_clusters_ == 0

    def test_al1_centers_approach_blob_means(self):
        X = make_blobs_unit_box()
        model = GKRDCA(n_clusters=2, n_init=10, random_state=0).fit(X)
        centers = model.cluster_centers_[np.argsort(model.cluster_centers_[:, 0])]

        np.testing.assert_allclose(centers[0], X[:40].mean(axis=0), atol=0.02)
        np.testing.assert_allclose(centers[1], X[40:].mean(axis=0), atol=0.02)

    @pytest.mark.parametrize("seed", range(6))
    def test_al2_objective_never_increases_in_any_run(self, seed):
        """DCA is a descent method: F_GK(V^{l+1}) <= F_GK(V^l) for every start."""
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, sigma=1.0, tau=0.1)
        lower, upper = model._compute_box_bounds(X)
        rng = np.random.default_rng(seed)
        V_start = model._initialize_centers(rng, lower, upper)

        _, _, _, history = model._run_single(X, V_start, lower, upper)

        assert len(history) >= 2
        differences = np.diff(np.asarray(history))
        assert (differences <= 1e-10).all()
        assert history[-1] < history[0]

    @pytest.mark.parametrize("sigma", [0.5, 2.0, 10.0])
    def test_al2_descent_holds_for_other_sigma_and_larger_rho(self, sigma):
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, sigma=sigma, tau=0.2, rho=3.0 / sigma**2)
        lower, upper = model._compute_box_bounds(X)
        V_start = model._initialize_centers(np.random.default_rng(1), lower, upper)

        _, _, _, history = model._run_single(X, V_start, lower, upper)

        assert (np.diff(np.asarray(history)) <= 1e-10).all()

    def test_al3_final_centers_stay_in_box(self):
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, n_init=4, random_state=3).fit(X)
        lower, upper = X.min(axis=0), X.max(axis=0)

        assert (model.cluster_centers_ >= lower).all()
        assert (model.cluster_centers_ <= upper).all()

    def test_al4_same_seed_reproduces_result(self):
        X = make_three_blobs()
        first = GKRDCA(n_clusters=3, n_init=5, random_state=12).fit(X)
        second = GKRDCA(n_clusters=3, n_init=5, random_state=12).fit(X)

        np.testing.assert_array_equal(first.cluster_centers_, second.cluster_centers_)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        assert first.objective_value_ == second.objective_value_
        assert first.n_iter_ == second.n_iter_

    def test_al4_multi_start_is_not_worse_than_its_first_start(self):
        X = make_three_blobs()
        single = GKRDCA(n_clusters=3, n_init=1, random_state=12).fit(X)
        multi = GKRDCA(n_clusters=3, n_init=8, random_state=12).fit(X)

        assert multi.objective_value_ <= single.objective_value_ + 1e-12

    def test_al5_iteration_count_is_within_limits(self):
        X = make_blobs_unit_box()
        model = GKRDCA(n_clusters=2, max_iter=50, random_state=0).fit(X)

        assert 1 <= model.n_iter_ <= 50

    def test_al6_hitting_max_iter_is_not_reported_as_converged(self):
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, max_iter=1, tol=0.0, n_init=3, random_state=0).fit(X)

        assert model.n_iter_ == 1
        assert model.converged_ is False

    def test_al6_reports_convergence_when_tolerance_is_met(self):
        X = make_blobs_unit_box()
        model = GKRDCA(n_clusters=2, tol=1e-5, max_iter=2000, n_init=5, random_state=0).fit(X)

        assert model.converged_ is True
        assert model.n_iter_ < 2000

    def test_al7_solution_is_finite(self):
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, random_state=2).fit(X)

        assert np.isfinite(model.cluster_centers_).all()
        assert np.isfinite(model.objective_value_)
        assert np.isfinite(np.asarray(model.objective_history_)).all()

    def test_al7_reported_objective_matches_final_centers(self):
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, n_init=3, random_state=2).fit(X)

        assert model.objective_value_ == pytest.approx(
            model._compute_objective(X, model.cluster_centers_)
        )
        assert model.objective_history_[-1] == pytest.approx(model.objective_value_)

    def test_al8_predict_matches_training_labels(self):
        X = make_three_blobs()
        model = GKRDCA(n_clusters=3, random_state=2).fit(X)

        np.testing.assert_array_equal(model.predict(X), model.labels_)
        assert set(model.labels_.tolist()).issubset({0, 1, 2})

    def test_al8_predict_requires_fit_and_matching_features(self):
        X = make_three_blobs()
        with pytest.raises(ValueError, match="not been fitted"):
            GKRDCA(n_clusters=3).predict(X)

        model = GKRDCA(n_clusters=3, random_state=2).fit(X)
        with pytest.raises(ValueError, match="number of features"):
            model.predict(np.zeros((2, 5)))
        with pytest.raises(ValueError, match="finite"):
            model.predict(np.array([[np.nan, 0.0]]))

    def test_resolved_rho_is_exposed_after_fit(self):
        X = make_blobs_unit_box()
        model = GKRDCA(n_clusters=2, sigma=2.0, random_state=0).fit(X)

        assert model.rho_ == pytest.approx(0.5)

    def test_m9_empty_cluster_is_reported_not_repaired(self, monkeypatch):
        """Both centers start equal: center 1 stays empty and the solution is not altered."""
        monkeypatch.setattr(
            GKRDCA,
            "_initialize_centers",
            lambda self, rng, lower, upper: np.array([[0.2], [0.2]]),
        )
        X = np.array([[0.0], [0.2], [0.4]])
        model = GKRDCA(n_clusters=2, n_init=1, random_state=0)

        with pytest.warns(RuntimeWarning, match="empty cluster"):
            model.fit(X)

        assert model.empty_clusters_ == 1
        np.testing.assert_allclose(model.cluster_centers_, [[0.2], [0.2]], atol=1e-12)


# --------------------------------------------------------------------------- #
# EV - integration with the shared evaluation modules
# --------------------------------------------------------------------------- #
class TestEvaluation:
    def test_ev1_iris_result_has_all_shared_and_gk_rdca_fields(self):
        result = evaluate_dataset("iris", n_init=3)

        assert set(result) == set(GK_RDCA_FIELDS)
        assert result["samples"] == 150
        assert result["features"] == 4
        assert result["clusters"] == 3
        assert result["noise_points"] == 0
        for metric in (
            "accuracy_percent",
            "precision_macro_percent",
            "recall_macro_percent",
            "f1_macro_percent",
        ):
            assert 0 <= result[metric] <= 100
        assert -1 <= result["ari"] <= 1
        assert 0 <= result["nmi"] <= 1
        assert result["runtime_seconds"] >= 0
        assert result["iterations"] > 0
        assert isinstance(result["converged"], bool)
        assert np.isfinite(result["objective_value"])
        assert result["rho"] == pytest.approx(2.0 / result["sigma"] ** 2)

    def test_ev1_wine_and_cancer_report_expected_shapes(self):
        expected = {"wine": (178, 3, 13), "cancer": (569, 2, 30)}
        for name, (samples, clusters, features) in expected.items():
            result = evaluate_dataset(name, n_init=2, max_iter=50)
            assert (result["samples"], result["clusters"], result["features"]) == (
                samples,
                clusters,
                features,
            )

    def test_ev2_all_evaluators_use_all_iris_features(self):
        gk_result = evaluate_dataset("iris", n_init=2)

        assert (
            gk_result["features"]
            == evaluate_mw_dataset("iris")["features"]
            == evaluate_classic_dataset("iris")["features"]
            == 4
        )

    def test_ev2_shared_field_names_are_a_prefix_of_gk_rdca_fields(self):
        assert GK_RDCA_FIELDS[: len(METRIC_FIELDS)] == METRIC_FIELDS
        assert GK_RDCA_FIELDS[len(METRIC_FIELDS) :] == GK_RDCA_EXTRA_FIELDS
        assert len(set(GK_RDCA_FIELDS)) == len(GK_RDCA_FIELDS)

    def test_ev2_evaluation_is_reproducible_apart_from_runtime(self):
        first = evaluate_dataset("iris", n_init=3, random_state=5)
        second = evaluate_dataset("iris", n_init=3, random_state=5)
        first.pop("runtime_seconds")
        second.pop("runtime_seconds")

        assert first == second

    def test_ev3_save_csv_exports_gk_rdca_columns_after_shared_prefix(self, tmp_path):
        result = evaluate_dataset("iris", n_init=2)
        output_path = tmp_path / "results" / "gk_rdca.csv"

        save_csv([result], output_path, fieldnames=GK_RDCA_FIELDS)

        lines = output_path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(GK_RDCA_FIELDS)
        assert lines[0].startswith("dataset,samples,features,clusters,")
        assert lines[1].startswith("iris,150,4,3,")

    def test_ev3_save_csv_default_columns_are_unchanged(self, tmp_path):
        result = evaluate_mw_dataset("iris")
        output_path = tmp_path / "mw.csv"

        save_csv([result], output_path)

        assert output_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(
            METRIC_FIELDS
        )

    def test_ev3_save_plot_accepts_gk_rdca_results(self, tmp_path):
        results = [evaluate_dataset("iris", n_init=2), evaluate_dataset("wine", n_init=2)]
        output_path = tmp_path / "gk_rdca.png"

        save_plot(results, output_path, title="GK-RDCA evaluation metrics")

        assert output_path.is_file() and output_path.stat().st_size > 0

    def test_ev3_print_solver_details_lists_every_dataset(self, capsys):
        results = [evaluate_dataset("iris", n_init=2)]

        print_solver_details(results)

        output = capsys.readouterr().out
        assert "iris" in output and "Converged" in output

    def test_ev4_rejects_unknown_dataset(self):
        with pytest.raises(ValueError, match="iris, wine, cancer"):
            evaluate_dataset("unknown")

    def test_ev5_yeast_runs_without_network(self, monkeypatch):
        rng = np.random.default_rng(17)
        features = rng.random((1484, 8))
        labels = np.asarray([f"class-{index % 10}" for index in range(1484)])

        def fake_fetch_openml(data_id, as_frame, parser):
            assert (data_id, as_frame, parser) == (181, False, "auto")
            return SimpleNamespace(data=features, target=labels)

        monkeypatch.setattr(benchmark_data, "fetch_openml", fake_fetch_openml)

        result = evaluate_dataset("yeast", n_init=2, max_iter=30)

        assert result["samples"] == 1484
        assert result["features"] == 8
        assert result["clusters"] == 10
        assert result["noise_points"] == 0