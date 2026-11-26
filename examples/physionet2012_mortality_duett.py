"""DuETT on PhysioNet 2012 in-hospital mortality: reproduction and ablations.

Authors:
    Hritish Mahajan (NetID: hritish2), Alex Jung (NetID: jhjung2)

Paper:
    Labach et al., "DuETT: Dual Event Time Transformer for Electronic Health
    Records," MLHC 2023. https://proceedings.mlr.press/v219/labach23a.html

Runs the authors' recipe (self-supervised pre-training, then fine-tuning,
model selection on validation PR-AUC) with :class:`~pyhealth.models.DuETT`
on :class:`~pyhealth.tasks.PhysioNet2012Mortality`, plus ablations the paper
did not run.

Usage::

    # About a minute on a laptop CPU, no data needed: synthetic stays.
    python examples/physionet2012_mortality_duett.py

    # The base recipe on PhysioNet 2012 (downloaded into --root if missing).
    python examples/physionet2012_mortality_duett.py --root data/physionet2012

    # All ablations (pre-trained encoders are cached and shared).
    python examples/physionet2012_mortality_duett.py --root data/physionet2012 \\
        --ablation all

Setup (following the authors' ``physionet.py`` / ``train.py``):

- 70 / 15 / 15 random split of the stays (seed 0, fixed across runs); the
  time series z-scored with training-set statistics, |z| >= 5 dropped (the
  raw data has unit errors such as a pH of 735).
- 8 static features as in the authors' code: weight, age, gender, height
  and the ICU type one-hot, z-scored on the training set, missing as 0.
- Pre-training: AdamW (lr 3e-4, weight decay 0.1), batch 512, linear
  warm-up then inverse square-root decay, gradient clipping at 1.0, the
  epoch with the lowest validation loss kept.
- Fine-tuning from it: AdamW (lr 1e-4, weight decay 1e-5), transformer
  dropout 0.5, time-bin masking 0.5, class-balanced BCE, the weights of
  the 5 epochs with the best validation PR-AUC averaged (as in the code).

Differences from the paper, for time: 30 pre-training epochs instead of
300 and 20 fine-tuning epochs instead of 50, with the warm-up shortened to
10% of the steps (the code's 2,000 / 1,000 steps would never finish). The
split is not the authors' (theirs comes from the ``torchtime`` package).

Ablations (``--ablation``), none of them in the paper's Table 2:

- ``n_bins``: the number of time bins, 8 / 16 / 32 / 64 (the code fixes 32).
- ``alpha``: the presence-loss weight, 0.05 / 0.2 / 1.0 (the paper never
  reports it; the code uses 0.2).
- ``regularization``: the two tricks that are in the code but not the
  paper, pre-training input dropout and fine-tuning time-bin masking,
  removed one at a time and together.
- ``hparams``: fine-tuning learning rate (3e-5 / 3e-4) and the embedding
  width (16 / 32).
- ``paper``: no pre-training vs. pre-training, a reproduction check of the
  paper's largest ablation effect (not a new ablation).

Results are appended to ``<output>/results.jsonl`` and printed as a table.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import torch

from pyhealth.datasets import (
    PhysioNet2012Dataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.metrics.binary import binary_metrics_fn
from pyhealth.models import DuETT
from pyhealth.tasks import (
    PhysioNet2012Mortality,
    physionet2012_mortality_statistics,
    split_by_record_ids,
)

STATIC_NUMERIC = ("weight", "age", "gender", "height")
ICU_TYPES = (1, 2, 3, 4)


@dataclasses.dataclass(frozen=True)
class Config:
    """One experiment; the defaults are the authors' recipe."""

    name: str = "base"
    pretrain: bool = True
    n_time_bins: int = 32
    d_embedding: int = 24
    presence_loss_weight: float = 0.2
    pretrain_input_dropout: float = 0.5
    aug_mask: float = 0.5
    dropout: float = 0.5
    lr: float = 1e-4

    def pretrain_key(self) -> str:
        """Identifies the pre-trained encoder, shared by fine-tuning runs."""
        return (
            f"nt{self.n_time_bins}_d{self.d_embedding}_"
            f"a{self.presence_loss_weight}_in{self.pretrain_input_dropout}"
        )


BASE = Config()
ABLATIONS: dict[str, list[Config]] = {
    "base": [BASE],
    "paper": [BASE, dataclasses.replace(BASE, name="no_ssl", pretrain=False)],
    "n_bins": [BASE]
    + [
        dataclasses.replace(BASE, name=f"n_bins={n}", n_time_bins=n)
        for n in (8, 16, 64)
    ],
    "alpha": [BASE]
    + [
        dataclasses.replace(BASE, name=f"alpha={a}", presence_loss_weight=a)
        for a in (0.05, 1.0)
    ],
    "regularization": [
        BASE,
        dataclasses.replace(BASE, name="no_input_dropout", pretrain_input_dropout=0.0),
        dataclasses.replace(BASE, name="no_aug_mask", aug_mask=0.0),
        dataclasses.replace(
            BASE, name="neither", pretrain_input_dropout=0.0, aug_mask=0.0
        ),
    ],
    "hparams": [BASE]
    + [dataclasses.replace(BASE, name=f"lr={lr}", lr=lr) for lr in (3e-5, 3e-4)]
    + [
        dataclasses.replace(BASE, name=f"d_embedding={d}", d_embedding=d)
        for d in (16, 32)
    ],
}


class PhysioNet2012MortalityWithStatics(PhysioNet2012Mortality):
    """:class:`PhysioNet2012Mortality` plus the authors' 8 static features.

    Args:
        static_mean: Training-set mean of each numeric static feature.
        static_std: Training-set standard deviation of each.
        pos_weight: Weight of positive stays in the BCE loss.
        **kwargs: Passed to :class:`PhysioNet2012Mortality`.
    """

    task_name: str = "PhysioNet2012MortalityWithStatics"
    input_schema: ClassVar[dict[str, str]] = {
        "x": "tensor",
        "x_time": "tensor",
        "static": "tensor",
    }

    def __init__(
        self,
        static_mean: dict[str, float],
        static_std: dict[str, float],
        pos_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.static_mean, self.static_std = static_mean, static_std
        self.output_schema = {"mortality": ("binary", {"pos_weight": pos_weight})}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = super().__call__(patient)
        if not samples:
            return samples
        info = patient.get_events(event_type="patients")[0].attr_dict
        static = []
        for name in STATIC_NUMERIC:
            value = _number(info.get(name))
            z = (value - self.static_mean[name]) / self.static_std[name]
            static.append(0.0 if math.isnan(z) else z)
        icu = _number(info.get("icutype"))
        static += [float(icu == t) for t in ICU_TYPES]
        samples[0]["static"] = np.array(static, dtype=np.float32)
        return samples


def _number(value: Any) -> float:
    """A static value as a float; PhysioNet's -1 and blanks are missing."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return math.nan if number < 0 else number


def load_physionet(root: str, cache_dir: str | None) -> tuple[dict, dict, dict, Any]:
    """Builds the split samples as in-memory tensors."""
    dataset = PhysioNet2012Dataset(root=root, download=True, cache_dir=cache_dir)
    patients = (
        dataset.global_event_df.filter(pl.col("event_type") == "patients")
        .select(
            "patient_id",
            "patients/in_hospital_death",
            "patients/icutype",
            *[f"patients/{n}" for n in STATIC_NUMERIC],
        )
        .collect()
    )
    ids = sorted(patients["patient_id"].to_list())
    order = np.random.default_rng(0).permutation(len(ids))
    n_train, n_val = int(0.7 * len(ids)), int(0.15 * len(ids))
    train_ids = [ids[i] for i in order[:n_train]]
    val_ids = [ids[i] for i in order[n_train : n_train + n_val]]
    test_ids = [ids[i] for i in order[n_train + n_val :]]

    train_rows = patients.filter(pl.col("patient_id").is_in(train_ids))
    static_mean, static_std = {}, {}
    for name in STATIC_NUMERIC:
        values = np.array([_number(v) for v in train_rows[f"patients/{name}"]])
        values = values[~np.isnan(values)]
        static_mean[name] = float(values.mean())
        static_std[name] = float(values.std()) or 1.0
    labels = np.array([_number(v) for v in train_rows["patients/in_hospital_death"]])
    p = float(np.nanmean(labels))

    mean, std = physionet2012_mortality_statistics(dataset, train_ids)
    task = PhysioNet2012MortalityWithStatics(
        static_mean,
        static_std,
        pos_weight=(1 - p) / p,
        mean=mean,
        std=std,
        outlier_sd=5.0,
    )
    samples = dataset.set_task(task)
    parts = split_by_record_ids(samples, train_ids, val_ids, test_ids)
    return (*(to_tensors(part) for part in parts), samples)


def synthetic_samples(n: int = 600, seed: int = 0) -> Any:
    """Fake PhysioNet-like stays whose label depends on two variables."""
    rng = np.random.default_rng(seed)
    n_vars, length = 6, 40
    samples = []
    for i in range(n):
        label = int(rng.random() < 0.25)
        x = rng.normal(size=(length, n_vars)).astype(np.float32)
        x[:, 0] += 1.5 * label  # a higher level of variable 0
        missing = rng.random((length, n_vars)) < 0.7
        missing[:, 1] = rng.random(length) < (0.3 if label else 0.9)  # sampled more
        x[missing] = np.nan
        n_obs = int(rng.integers(10, length))
        times = np.sort(rng.uniform(0, 48, length)).astype(np.float32)
        x[n_obs:], times[n_obs:] = np.nan, np.nan
        static = rng.normal(size=8).astype(np.float32)
        static[0] += 0.5 * label
        samples.append(
            {
                "patient_id": f"s{i}",
                "x": x,
                "x_time": times,
                "static": static,
                "mortality": label,
            }
        )
    return create_sample_dataset(
        samples,
        {"x": "tensor", "x_time": "tensor", "static": "tensor"},
        {"mortality": ("binary", {"pos_weight": 3.0})},
    )


def to_tensors(samples: Any) -> dict[str, torch.Tensor]:
    """All samples as stacked tensors (PhysioNet 2012 fits in memory)."""
    batch = next(iter(get_dataloader(samples, batch_size=len(samples))))
    return {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}


def batches(
    data: dict[str, torch.Tensor], size: int, generator: torch.Generator | None = None
) -> Iterator[dict[str, torch.Tensor]]:
    """Mini-batches, shuffled with ``generator`` if given."""
    n = len(data["mortality"])
    order = torch.randperm(n, generator=generator) if generator else torch.arange(n)
    for start in range(0, n, size):
        index = order[start : start + size]
        if len(index) > 1:  # BatchNorm needs two samples
            yield {k: v[index] for k, v in data.items()}


def warmup_inverse_sqrt(
    optimizer: torch.optim.Optimizer, warmup: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """The code's schedule: linear warm-up, then ``sqrt(warmup / step)``."""
    warmup = max(warmup, 1)

    def factor(step: int) -> float:
        if step < warmup:
            return step / warmup
        return math.sqrt(warmup / (step - warmup + warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# Input field names of DuETT; the MIMIC-IV example sets the pre-binned ones.
MODEL_FIELDS: dict[str, Any] = {"static_key": "static"}


def make_model(samples: Any, config: Config, **overrides: Any) -> DuETT:
    kwargs = {
        **MODEL_FIELDS,
        "n_time_bins": config.n_time_bins,
        "d_embedding": config.d_embedding,
        "presence_loss_weight": config.presence_loss_weight,
        "pretrain_input_dropout": config.pretrain_input_dropout,
        "aug_mask": config.aug_mask,
        "dropout": config.dropout,
    }
    kwargs.update(overrides)
    return DuETT(samples, **kwargs)


@torch.no_grad()
def pretrain_loss(model: DuETT, data: dict, size: int) -> float:
    """Validation pre-training loss with fixed masks (seeded)."""
    model.eval()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        losses = [
            model.pretrain_forward(**b)["loss"].item() for b in batches(data, size)
        ]
    return float(np.mean(losses))


def pretrain(
    samples: Any, config: Config, train: dict, val: dict, args: argparse.Namespace
) -> dict[str, torch.Tensor]:
    """Pre-trains an encoder (or loads the cached one) and returns its weights."""
    path = Path(args.output) / f"pretrain_{config.pretrain_key()}_seed{args.seed}.pt"
    if path.exists():
        return torch.load(path)
    torch.manual_seed(args.seed)
    model = make_model(samples, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    steps = args.pretrain_epochs * math.ceil(len(train["mortality"]) / args.batch_size)
    scheduler = warmup_inverse_sqrt(optimizer, steps // 10)
    generator = torch.Generator().manual_seed(args.seed)
    best, best_state = math.inf, None
    for epoch in range(args.pretrain_epochs):
        start = time.time()
        model.train()
        for batch in batches(train, args.batch_size, generator):
            optimizer.zero_grad()
            model.pretrain_forward(**batch)["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_loss = pretrain_loss(model, val, args.batch_size)
        print(
            f"  pretrain {config.pretrain_key()} epoch {epoch}: val loss "
            f"{val_loss:.4f} ({time.time() - start:.0f}s)",
            flush=True,
        )
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, path)
    return best_state


@torch.no_grad()
def evaluate(model: DuETT, data: dict, size: int) -> dict[str, float]:
    model.eval()
    probs = [model(**b)["y_prob"].view(-1) for b in batches(data, size)]
    return binary_metrics_fn(
        data["mortality"].view(-1).numpy(),
        torch.cat(probs).numpy(),
        metrics=["roc_auc", "pr_auc"],
    )


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """The mean of several state dicts (integer buffers from the first)."""
    out = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            out[key] = sum(s[key] for s in states) / len(states)
        else:
            out[key] = value
    return out


def fine_tune(
    samples: Any,
    config: Config,
    data: tuple[dict, dict, dict],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Fine-tunes (from the pre-trained encoder if ``config.pretrain``)."""
    train, val, test = data
    start = time.time()
    state = pretrain(samples, config, train, val, args) if config.pretrain else None
    torch.manual_seed(args.seed)
    model = make_model(samples, config)
    if state is not None:
        model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
    steps = args.finetune_epochs * math.ceil(len(train["mortality"]) / args.batch_size)
    scheduler = warmup_inverse_sqrt(optimizer, steps // 10)
    generator = torch.Generator().manual_seed(args.seed)
    top: list[tuple[float, int, dict]] = []  # (val PR-AUC, epoch, weights)
    for epoch in range(args.finetune_epochs):
        model.train()
        for batch in batches(train, args.batch_size, generator):
            optimizer.zero_grad()
            model(**batch)["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        scores = evaluate(model, val, args.batch_size)
        print(
            f"  finetune {config.name} epoch {epoch}: val PR-AUC "
            f"{scores['pr_auc']:.4f} ROC-AUC {scores['roc_auc']:.4f}",
            flush=True,
        )
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        top = sorted(top + [(scores["pr_auc"], epoch, weights)], key=lambda t: -t[0])
        top = top[: args.top_k]
    model.load_state_dict(average_states([w for _, _, w in top]))
    return {
        "name": config.name,
        "seed": args.seed,
        "config": dataclasses.asdict(config),
        "val": evaluate(model, val, args.batch_size),
        "test": evaluate(model, test, args.batch_size),
        "top_epochs": sorted(e for _, e, _ in top),
        "minutes": round((time.time() - start) / 60, 1),
        "data": "synthetic" if args.root is None else "physionet2012",
        "epochs": [args.pretrain_epochs, args.finetune_epochs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", default=None, help="PhysioNet 2012 folder; synthetic data if omitted."
    )
    parser.add_argument("--cache-dir", default=None, help="PyHealth cache folder.")
    parser.add_argument("--output", default="duett_physionet2012")
    parser.add_argument("--ablation", default="base", choices=[*ABLATIONS, "all"])
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Average the weights of the k best fine-tuning epochs.",
    )
    args = parser.parse_args()
    synthetic = args.root is None
    if args.pretrain_epochs is None:
        args.pretrain_epochs = 5 if synthetic else 30
    if args.finetune_epochs is None:
        args.finetune_epochs = 8 if synthetic else 20
    if synthetic:
        args.batch_size = min(args.batch_size, 64)
        args.top_k = min(args.top_k, 2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if synthetic:
        samples = synthetic_samples()
        n = len(samples)
        parts = [
            samples.subset(list(range(a, b)))
            for a, b in (
                (0, int(0.7 * n)),
                (int(0.7 * n), int(0.85 * n)),
                (int(0.85 * n), n),
            )
        ]
        train, val, test = (to_tensors(p) for p in parts)
    else:
        train, val, test, samples = load_physionet(args.root, args.cache_dir)
    print(
        f"train / val / test: {len(train['mortality'])} / "
        f"{len(val['mortality'])} / {len(test['mortality'])} stays, "
        f"{train['mortality'].float().mean():.3f} positive in train",
        flush=True,
    )

    names = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
    configs = list({c: None for n in names for c in ABLATIONS[n]})  # dedupe
    results = []
    for config in configs:
        print(f"== {config.name}", flush=True)
        result = fine_tune(samples, config, (train, val, test), args)
        results.append(result)
        with open(output / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

    print(
        f"\n{'config':<20} {'val PR-AUC':>10} {'test PR-AUC':>11} "
        f"{'test ROC-AUC':>12} {'min':>5}"
    )
    for r in results:
        print(
            f"{r['name']:<20} {r['val']['pr_auc']:>10.4f} "
            f"{r['test']['pr_auc']:>11.4f} {r['test']['roc_auc']:>12.4f} "
            f"{r['minutes']:>5}"
        )


if __name__ == "__main__":
    main()
