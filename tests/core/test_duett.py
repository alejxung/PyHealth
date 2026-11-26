"""Tests for the DuETT model, on small synthetic data only.

Authors:
    Hritish Mahajan (NetID: hritish2), Alex Jung (NetID: jhjung2)

Paper:
    Labach et al., "DuETT: Dual Event Time Transformer for Electronic Health
    Records," MLHC 2023. https://proceedings.mlr.press/v219/labach23a.html
"""

import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DuETT
from pyhealth.models.duett import (
    DuETTEncoderLayer,
    PerVariableEmbedding,
    ScaleNorm,
    bin_time_series,
)
from pyhealth.trainer import Trainer

N_VARS, LENGTH, N_STATIC = 3, 8, 2
TINY = {
    "d_embedding": 8,
    "d_feedforward": 16,
    "d_hidden_embedding": 8,
    "d_hidden_static": 8,
    "d_hidden_head": 8,
}


def _irregular_samples(n: int, seed: int = 0) -> list[dict]:
    """Irregular samples like PhysioNet2012Mortality's: NaN-padded x/x_time."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        x = rng.normal(size=(LENGTH, N_VARS)).astype("float32")
        x[rng.random((LENGTH, N_VARS)) < 0.5] = np.nan
        t = np.sort(rng.uniform(0, 48, LENGTH)).astype("float32")
        t[LENGTH - 2 :] = np.nan  # padding
        samples.append({"patient_id": f"p{i}", "x": x, "x_time": t, "label": i % 2})
    return samples


def _binned_samples(n: int, n_bins: int, seed: int = 0) -> list[dict]:
    """Pre-binned samples with counts and static features."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n):
        counts = rng.integers(0, 3, size=(n_bins, N_VARS)).astype("float32")
        values = rng.normal(size=(n_bins, N_VARS)).astype("float32") * (counts > 0)
        samples.append(
            {
                "patient_id": f"p{i}",
                "values": values,
                "counts": counts,
                "times": (np.arange(1, n_bins + 1) / n_bins * 2).astype("float32"),
                "static": rng.normal(size=N_STATIC).astype("float32"),
                "label": i % 2,
            }
        )
    return samples


def _reference_binning(x, t, n_bins):
    """The authors' PhysioNet loop (physionet.py), for one sample."""
    keep = ~np.isnan(t)
    x, t = x[keep], t[keep]
    values = np.zeros((n_bins, x.shape[1]), dtype=np.float32)
    counts = np.zeros_like(values)
    for i_t, time in enumerate(t):
        b = n_bins - 1 if time == t[-1] else int(time / t[-1] * n_bins)
        for v in range(x.shape[1]):
            if not np.isnan(x[i_t, v]):
                values[b, v] = x[i_t, v]
                counts[b, v] += 1
    ends = np.arange(1, n_bins + 1) / n_bins * t[-1] / 24
    return values, counts, ends


class TestBinTimeSeries(unittest.TestCase):
    def test_matches_authors_binning(self):
        samples = _irregular_samples(4, seed=1)
        x = torch.tensor(np.stack([s["x"] for s in samples]))
        t = torch.tensor(np.stack([s["x_time"] for s in samples]))
        values, counts, ends = bin_time_series(x, t, n_bins=4)
        for i, s in enumerate(samples):
            ref_v, ref_c, ref_e = _reference_binning(s["x"], s["x_time"], 4)
            np.testing.assert_allclose(values[i].numpy(), ref_v, rtol=1e-6)
            np.testing.assert_array_equal(counts[i].numpy(), ref_c)
            np.testing.assert_allclose(ends[i].numpy(), ref_e, rtol=1e-5)

    def test_fixed_window(self):
        x = torch.tensor([[[1.0], [2.0], [3.0], [5.0]]])
        t = torch.tensor([[1.0, 13.0, 47.0, 60.0]])
        values, counts, ends = bin_time_series(x, t, n_bins=4, window=48.0)
        # 60 h is past the window, so it goes to the last bin, after 47 h.
        self.assertEqual(values[0, :, 0].tolist(), [1.0, 2.0, 0.0, 5.0])
        self.assertEqual(counts[0, :, 0].tolist(), [1.0, 1.0, 0.0, 2.0])
        self.assertEqual(ends[0].tolist(), [0.5, 1.0, 1.5, 2.0])

    def test_empty_sample_gives_zeros(self):
        x = torch.full((1, 3, 2), float("nan"))
        t = torch.full((1, 3), float("nan"))
        values, counts, ends = bin_time_series(x, t, n_bins=2)
        self.assertTrue(torch.all(values == 0) and torch.all(counts == 0))
        self.assertTrue(torch.isfinite(ends).all())


class TestLayers(unittest.TestCase):
    def test_scale_norm(self):
        out = ScaleNorm(9)(torch.randn(4, 9) * 10)
        torch.testing.assert_close(out.norm(dim=-1), torch.full((4,), 3.0))

    def test_encoder_layer_shape_and_dropout(self):
        layer = DuETTEncoderLayer(dim=12, n_heads=2, d_head=3, d_feedforward=8)
        x = torch.randn(2, 5, 12)
        self.assertEqual(layer(x).shape, (2, 5, 12))
        layer.eval()  # dropout is off in evaluation mode
        torch.testing.assert_close(layer(x, dropout=0.9), layer(x))

    def test_per_variable_embedding_is_separate_per_variable(self):
        emb = PerVariableEmbedding(n_vars=3, d_hidden=4, d_embedding=5).eval()
        cells = torch.randn(2, 4, 3, 2)
        changed = cells.clone()
        changed[:, :, 0] += 1.0
        a, b = emb(cells), emb(changed)
        self.assertFalse(torch.allclose(a[:, :, 0], b[:, :, 0]))
        torch.testing.assert_close(a[:, :, 1:], b[:, :, 1:])


class TestDuETT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = create_sample_dataset(
            _irregular_samples(6),
            {"x": "tensor", "x_time": "tensor"},
            {"label": "binary"},
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=6)))

    def _model(self, **kwargs) -> DuETT:
        return DuETT(self.dataset, n_time_bins=4, **{**TINY, **kwargs})

    def test_init(self):
        model = self._model()
        self.assertEqual(model.n_vars, N_VARS)
        self.assertEqual(model.d_static, 0)
        self.assertEqual(model.mode, "binary")
        self.assertEqual(len(model.event_transformers), 2)
        self.assertEqual(model.event_embedding.shape, (N_VARS + 1, 8 * 5))

    def test_forward_shapes(self):
        out = self._model()(**self.batch, embed=True)
        self.assertEqual(out["logit"].shape, (6, 1))
        self.assertEqual(out["y_prob"].shape, (6, 1))
        self.assertEqual(out["y_true"].shape, (6, 1))
        self.assertEqual(out["embed"].shape, (6, 8 * (N_VARS + 1)))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(((out["y_prob"] > 0) & (out["y_prob"] < 1)).all())

    def test_forward_without_labels(self):
        batch = {k: v for k, v in self.batch.items() if k != "label"}
        out = self._model()(**batch)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (6, 1))

    def test_backward(self):
        model = self._model()
        model(**self.batch)["loss"].backward()
        for name in (
            "head.0.weight",
            "cell_embedding.w1",
            "count_embedding.weight",
            "event_embedding",
            "rep_embedding",
            "time_embedding.0.weight",
            "event_transformers.0.to_qkv.weight",
            "time_transformers.1.ff_out.weight",
        ):
            grad = model.get_parameter(name).grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)

    def test_pretrain_forward(self):
        model = self._model()
        out = model.pretrain_forward(**self.batch)
        parts = (
            "time_value_loss",
            "time_presence_loss",
            "event_value_loss",
            "event_presence_loss",
        )
        self.assertEqual(set(out), {"loss", *parts})
        expected = (
            out["time_value_loss"]
            + out["event_value_loss"]
            + 0.2 * (out["time_presence_loss"] + out["event_presence_loss"])
        )
        torch.testing.assert_close(out["loss"], expected)
        out["loss"].backward()
        self.assertIsNotNone(model.value_proj.weight.grad)
        self.assertIsNotNone(model.event_presence_proj.weight.grad)
        self.assertIsNone(model.head[0].weight.grad)  # head unused in SSL

    def test_presence_loss_weight(self):
        torch.manual_seed(1)
        out = self._model(presence_loss_weight=1.0).pretrain_forward(**self.batch)
        expected = sum(out[k] for k in out if k != "loss")
        torch.testing.assert_close(out["loss"], expected)

    def test_value_loss_observed_only(self):
        pred, target = torch.ones(2, 4), torch.zeros(2, 4)
        presence = torch.tensor([[1.0, 0, 0, 0], [0, 0, 0, 0]])
        model = self._model()
        self.assertAlmostEqual(model._value_loss(pred, target, presence).item(), 1 / 8)
        model.pretrain_loss_observed_only = True
        self.assertAlmostEqual(model._value_loss(pred, target, presence).item(), 1.0)

    def test_eval_is_deterministic(self):
        model = self._model().eval()
        torch.testing.assert_close(
            model(**self.batch)["logit"], model(**self.batch)["logit"]
        )

    def test_aug_mask_hides_all_values(self):
        # With every bin masked, the prediction cannot depend on the values.
        model = self._model(aug_mask=1.0, dropout=0.0).train()
        shifted = dict(self.batch, x=self.batch["x"] + 5.0)
        torch.testing.assert_close(
            model(**self.batch)["logit"], model(**shifted)["logit"]
        )

    def test_event_mask_hides_that_event(self):
        model = self._model().eval()
        values, counts, times, _ = model.grid(**self.batch)
        mask = torch.zeros(6, N_VARS, dtype=torch.bool)
        mask[:, 0] = True
        changed = values.clone()
        changed[:, :, 0] += 3.0
        a = model.encode(values, counts, times, event_mask=mask)
        b = model.encode(changed, counts, times, event_mask=mask)
        torch.testing.assert_close(a, b)
        c = model.encode(changed, counts, times)
        self.assertFalse(torch.allclose(a, c))

    def test_pretrained_weights_load_for_fine_tuning(self):
        pretrained = self._model(dropout=0.0, aug_mask=0.0)
        pretrained.pretrain_forward(**self.batch)["loss"].backward()
        fine_tune = self._model(dropout=0.5, aug_mask=0.5)
        fine_tune.load_state_dict(pretrained.state_dict())
        fine_tune.eval()
        pretrained.eval()
        torch.testing.assert_close(
            fine_tune(**self.batch)["logit"], pretrained(**self.batch)["logit"]
        )

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self._model(n_heads=3)
        with self.assertRaises(ValueError):
            self._model(aug_mask=1.5)

    def test_trainer_one_epoch(self):
        model = self._model()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(model=model, output_path=tmp, enable_logging=False)
            loader = get_dataloader(self.dataset, batch_size=3, shuffle=True)
            trainer.train(loader, loader, epochs=1, monitor="roc_auc")
            self.assertIn("roc_auc", trainer.evaluate(loader))


class TestDuETTPreBinned(unittest.TestCase):
    def _dataset(self, n_bins: int):
        return create_sample_dataset(
            _binned_samples(4, n_bins),
            {
                "values": "tensor",
                "counts": "tensor",
                "times": "tensor",
                "static": "tensor",
            },
            {"label": "binary"},
        )

    def test_time_bins_and_static(self):
        for n_bins in (2, 6):
            with self.subTest(n_bins=n_bins):
                dataset = self._dataset(n_bins)
                model = DuETT(
                    dataset,
                    n_time_bins=n_bins,
                    value_key="values",
                    time_key="times",
                    count_key="counts",
                    static_key="static",
                    **TINY,
                )
                self.assertEqual(model.d_static, N_STATIC)
                batch = next(iter(get_dataloader(dataset, batch_size=4)))
                out = model(**batch)
                self.assertEqual(out["y_prob"].shape, (4, 1))
                out["loss"].backward()
                self.assertIsNotNone(model.static_encoder[0].weight.grad)
                self.assertTrue(torch.isfinite(model.pretrain_forward(**batch)["loss"]))

    def test_bin_count_mismatch_raises(self):
        with self.assertRaises(ValueError):
            DuETT(
                self._dataset(4),
                n_time_bins=8,
                value_key="values",
                time_key="times",
                count_key="counts",
                **TINY,
            )


if __name__ == "__main__":
    unittest.main()
