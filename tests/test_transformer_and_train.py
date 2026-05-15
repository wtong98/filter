import jax
import numpy as np

from model.transformer import TransformerConfig
from task.filter import KalmanFilterTask
from train import train


def test_transformer_sequence_output_shape():
    config = TransformerConfig(
        n_layers=1,
        n_hidden=16,
        n_heads=1,
        n_mlp_layers=1,
        pos_emb=False,
        return_final_logits_only=False,
        n_out=2,
    )
    model = config.to_model()
    xs = np.zeros((3, 5, 4), dtype=np.float32)

    params = model.init(jax.random.key(0), xs)["params"]
    preds = model.apply({"params": params}, xs)

    assert preds.shape == (3, 5, 2)


def test_transformer_final_scalar_output_shape():
    config = TransformerConfig(
        n_layers=1,
        n_hidden=16,
        n_heads=1,
        n_mlp_layers=0,
        pos_emb=False,
        return_final_logits_only=True,
        n_out=1,
    )
    model = config.to_model()
    xs = np.zeros((3, 5, 4), dtype=np.float32)

    params = model.init(jax.random.key(0), xs)["params"]
    preds = model.apply({"params": params}, xs)

    assert preds.shape == (3,)


def test_train_smoke_updates_and_records_history():
    task = KalmanFilterTask(
        length=4,
        n_dims=2,
        n_obs_dims=2,
        mode="ac",
        n_tasks=None,
        batch_size=4,
        seed=0,
    )
    config = TransformerConfig(
        n_layers=1,
        n_hidden=8,
        n_heads=1,
        n_mlp_layers=0,
        pos_emb=False,
        return_final_logits_only=False,
        n_out=2,
    )

    state, hist = train(
        config,
        data_iter=iter(task),
        train_iters=2,
        test_every=1,
        test_iters=1,
        seed=0,
    )

    assert len(hist["train"]) == 2
    assert len(hist["test"]) == 2
    assert np.isfinite(float(hist["train"][-1].loss))
    assert np.isfinite(float(hist["test"][-1].loss))
    assert state.params

