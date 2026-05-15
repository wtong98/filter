import numpy as np

from task.filter import KalmanFilterTask, pred_kalman


def test_kalman_filter_task_default_batch_shape():
    task = KalmanFilterTask(length=5, n_dims=3, n_obs_dims=2, batch_size=7, seed=0)

    xs = next(task)

    assert xs.shape == (7, 5, 2)
    assert np.isfinite(xs).all()


def test_kalman_filter_task_ac_batch_contains_context_and_observations():
    task = KalmanFilterTask(
        length=6,
        n_dims=3,
        n_obs_dims=2,
        mode="ac",
        n_tasks=None,
        batch_size=4,
        seed=0,
    )

    xs = next(task)

    assert xs.shape == (4, 11, 5)

    A = xs[:, :3, 2:]
    C = xs[:, 3:5, 2:]
    observations = xs[:, 5:, :2]

    assert A.shape == (4, 3, 3)
    assert C.shape == (4, 2, 3)
    assert observations.shape == (4, 6, 2)
    assert np.any(np.abs(A) > 0)
    assert np.any(np.abs(C) > 0)
    assert np.any(np.abs(observations) > 0)

    assert np.allclose(xs[:, :5, :2], 0)
    assert np.allclose(xs[:, 5:, 2:], 0)


def test_kalman_filter_task_ac_snapshot_estimate_keeps_shape():
    task = KalmanFilterTask(
        length=4,
        n_dims=3,
        n_obs_dims=2,
        mode="ac",
        n_tasks=None,
        n_snaps=8,
        batch_size=5,
        seed=1,
    )

    xs = next(task)

    assert xs.shape == (5, 9, 5)
    assert np.isfinite(xs).all()


def test_pred_kalman_returns_predictions_and_variances_for_batched_matrices():
    task = KalmanFilterTask(
        length=6,
        n_dims=3,
        n_obs_dims=2,
        mode="ac",
        n_tasks=None,
        batch_size=4,
        t_noise=0.1,
        o_noise=0.1,
        seed=0,
    )
    xs_full = next(task)
    A = xs_full[:, :3, 2:]
    C = xs_full[:, 3:5, 2:]
    observations = xs_full[:, 5:, :2]

    preds, true_mse = pred_kalman(observations, task, A=A, C=C)

    assert preds.shape == observations.shape
    assert len(true_mse) == observations.shape[1]
    assert np.isfinite(preds).all()
    assert np.isfinite(true_mse).all()


def test_pred_kalman_return_mat_shape_for_fixed_system():
    task = KalmanFilterTask(
        length=5,
        n_dims=3,
        n_obs_dims=2,
        n_tasks=1,
        batch_size=4,
        t_noise=0.1,
        o_noise=0.1,
        seed=0,
    )
    observations = next(task)

    preds, kalman_mat = pred_kalman(observations, task, return_mat=True)

    assert preds.shape == observations.shape
    assert kalman_mat.shape == (
        (observations.shape[1] - 1) * task.n_obs_dims,
        (observations.shape[1] - 1) * task.n_obs_dims,
    )
    assert np.isfinite(kalman_mat).all()
