"""DuETT on MIMIC-IV ICU mortality: the full pipeline and ablations.

Authors:
    Hritish Mahajan (NetID: hritish2), Alex Jung (NetID: jhjung2)

Paper:
    Labach et al., "DuETT: Dual Event Time Transformer for Electronic Health
    Records," MLHC 2023. https://proceedings.mlr.press/v219/labach23a.html

Data loading (:class:`~pyhealth.datasets.MIMIC4EHRDataset`) -> task
(:class:`~pyhealth.tasks.DuETTMortalityMIMIC4`: one sample per ICU stay, the
first 48 hours of 114 chart and lab variables binned into last values and
counts, plus static features) -> :class:`~pyhealth.models.DuETT`
pre-training and fine-tuning -> test PR-AUC / ROC-AUC. The training recipe
and the ablations are those of ``physionet2012_mortality_duett.py`` (see
its docstring), imported from it.

Usage::

    # About a minute, no data needed: synthetic MIMIC-IV tables written to a
    # temporary folder and loaded through MIMIC4EHRDataset.
    python examples/mimic4_icu_mortality_duett.py

    # MIMIC-IV (credentialed access). hosp/ and icu/ with patients,
    # admissions, d_labitems, labevents, icustays and chartevents.
    python examples/mimic4_icu_mortality_duett.py --root /path/to/mimiciv \\
        --split-file mimic-iv-patient-split.json --ablation all

    # With about 13 GB of RAM (free Colab): first write a copy with only the
    # task's events (about 8% of chartevents, 4% of labevents), then load it.
    python examples/mimic4_icu_mortality_duett.py --root /path/to/mimiciv \\
        --prefilter /content/mimiciv_duett --split-file mimic-iv-patient-split.json

``--split-file`` is the authors' patient split (``train`` / ``test`` subject
IDs; validation is the first 15% of ``train``) from
https://github.com/layer6ai-labs/DuETT; without it, patients are split
70 / 15 / 15 at random. Values are clipped at 3 median absolute deviations
and z-scored with training statistics (:func:`fit_value_normalizer`), as in
the paper; age and the window length are z-scored too.

Only aggregate metrics are printed and saved, never patient-level data.

Results on MIMIC-IV v3.1 (one Colab T4 run, seed 2020, the recipe of
``physionet2012_mortality_duett.py``: 30 pre-training and 20 fine-tuning
epochs). 46,337 stays pass the cohort criteria; 27,659 / 4,969 / 5,748 of
them (about 14% deaths) fall in the authors' train / val / test patients::

    config        val PR-AUC  test PR-AUC  test ROC-AUC
    base              0.597        0.575         0.885
    no_ssl            0.571        0.554         0.880
    no_aug_mask       0.593        0.581         0.886
    lr=3e-4           0.601        0.571         0.882

Unlike on PhysioNet 2012, pre-training helps on MIMIC-IV (+0.021 test
PR-AUC, one seed), which has about three times as many training stays. The two
fine-tuning changes that helped on PhysioNet 2012 are within noise here.
The paper reports 0.627 PR-AUC and 0.912 ROC-AUC for DuETT on its MIMIC-IV
cohort after 300 pre-training epochs; ours are lower, as on PhysioNet
2012. The other ablations ran on PhysioNet 2012 only, as each bin count
needs its own MIMIC-IV preprocessing (about 40 minutes on Colab).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physionet2012_mortality_duett as recipe

from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit, Split
from pyhealth.tasks import (
    DuETTMortalityMIMIC4,
    fit_value_normalizer,
    normalize_values,
)

recipe.MODEL_FIELDS.update(
    value_key="values", time_key="times", count_key="counts", static_key="static"
)
TABLES = ["chartevents", "labevents"]


class AuthorsSplit(Split):
    """The authors' MIMIC-IV patient split: train / val (first 15%) / test.

    Args:
        path: ``mimic-iv-patient-split.json`` with ``train`` and ``test``
            subject IDs.
    """

    def __init__(self, path: str) -> None:
        raw = json.loads(Path(path).read_text())
        train = [str(i) for i in raw["train"]]
        n_val = int(0.15 * len(train))
        self.parts = (train[n_val:], train[:n_val], [str(i) for i in raw["test"]])
        self.digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        return [
            np.array(
                sorted(
                    i
                    for p in part
                    if p in patient_to_index
                    for i in patient_to_index[p]
                ),
                dtype=np.int64,
            )
            for part in self.parts
        ]

    def to_dict(self) -> dict:
        return {"kind": "duett_authors", "sha256": self.digest}


def write_synthetic_mimic(root: Path, n_patients: int = 300, seed: int = 0) -> None:
    """Writes tiny fake MIMIC-IV tables; deaths have higher heart rates and
    more lactate measurements."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2150, 1, 1)
    fmt = "%Y-%m-%d %H:%M:%S"
    rows: dict[str, list[list[Any]]] = {
        k: []
        for k in ("patients", "admissions", "icustays", "chartevents", "labevents")
    }
    for subject in range(1, n_patients + 1):
        died = int(rng.random() < 0.2)
        hadm, stay = 1000 + subject, 5000 + subject
        intime = t0 + timedelta(days=int(rng.integers(0, 300)))
        hours = float(rng.uniform(30, 120))  # some stays are under 48 h
        rows["patients"].append(
            [
                subject,
                rng.choice(["M", "F"]),
                int(rng.integers(16, 90)),
                2150,
                "2017 - 2019",
                "",
            ]
        )
        rows["admissions"].append(
            [
                subject,
                hadm,
                (intime - timedelta(hours=3)).strftime(fmt),
                (intime + timedelta(hours=hours + 24)).strftime(fmt),
                "",
                "EW EMER.",
                "EMERGENCY ROOM",
                "Medicare",
                "English",
                "MARRIED",
                "WHITE",
                "HOME",
                died,
            ]
        )
        rows["icustays"].append(
            [
                subject,
                hadm,
                stay,
                "Medical Intensive Care Unit (MICU)",
                "Medical Intensive Care Unit (MICU)",
                intime.strftime(fmt),
                (intime + timedelta(hours=hours)).strftime(fmt),
                round(hours / 24, 2),
            ]
        )
        for h in np.sort(rng.uniform(0, hours, int(rng.integers(20, 60)))):
            time = (intime + timedelta(hours=float(h))).strftime(fmt)
            hr = rng.normal(80 + 6 * died, 12)
            rows["chartevents"].append(
                [subject, hadm, stay, time, 220045, f"{hr:.0f}", round(hr), "bpm"]
            )
            if rng.random() < 0.3:
                temp = rng.normal(98.6, 1.0)
                rows["chartevents"].append(
                    [
                        subject,
                        hadm,
                        stay,
                        time,
                        223761,
                        f"{temp:.1f}",
                        round(temp, 1),
                        "°F",
                    ]
                )
        for h in np.sort(rng.uniform(0, hours, int(rng.integers(2, 6 + 8 * died)))):
            time = (intime + timedelta(hours=float(h))).strftime(fmt)
            lactate = rng.normal(1.5 + 0.5 * died, 0.8)
            rows["labevents"].append(
                [
                    subject,
                    hadm,
                    time,
                    50813,
                    f"{lactate:.1f}",
                    round(lactate, 1),
                    "mmol/L",
                    "",
                    "",
                    "",
                    time,
                ]
            )
    headers = {
        "patients": "subject_id,gender,anchor_age,anchor_year,anchor_year_group,dod",
        "admissions": "subject_id,hadm_id,admittime,dischtime,deathtime,"
        "admission_type,admission_location,insurance,language,marital_status,"
        "race,discharge_location,hospital_expire_flag",
        "icustays": "subject_id,hadm_id,stay_id,first_careunit,last_careunit,"
        "intime,outtime,los",
        "chartevents": "subject_id,hadm_id,stay_id,charttime,itemid,value,"
        "valuenum,valueuom",
        "labevents": "subject_id,hadm_id,charttime,itemid,value,valuenum,"
        "valueuom,ref_range_lower,ref_range_upper,flag,storetime",
    }
    folders = {
        "patients": "hosp",
        "admissions": "hosp",
        "labevents": "hosp",
        "icustays": "icu",
        "chartevents": "icu",
    }
    for table, header in headers.items():
        path = root / folders[table] / f"{table}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [header] + [",".join(str(v) for v in row) for row in rows[table]]
        with gzip.open(path, "wt") as f:
            f.write("\n".join(lines) + "\n")
    with gzip.open(root / "hosp" / "d_labitems.csv.gz", "wt") as f:
        f.write("itemid,label,fluid,category\n50813,Lactate,Blood,Blood Gas\n")


def prefilter_mimic(root: Path, out: Path, window_hours: float = 48.0) -> None:
    """Writes a smaller copy of MIMIC-IV with only the events the task uses.

    ``MIMIC4EHRDataset`` sorts every event of every table before the task's
    ``pre_filter`` runs, which needs more memory than a free Colab runtime
    has. This keeps only the task's items, and only events in the first
    ``window_hours`` of stays at least that long (labs: of any such stay of
    the admission), streaming the files in 64 MB blocks. The small tables
    are linked, not copied.

    Args:
        root: The MIMIC-IV folder (``hosp/`` and ``icu/``).
        out: Where to write the copy.
        window_hours: The task's observation window.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pv

    task = DuETTMortalityMIMIC4(window_hours=window_hours)
    items = {
        t: {str(i) for v in task.variables if v.table == t for i in v.itemids}
        for t in TABLES
    }
    fmt = "%Y-%m-%d %H:%M:%S"
    stays: dict[str, tuple[str, str]] = {}
    admissions: dict[str, tuple[str, str]] = {}
    with gzip.open(root / "icu" / "icustays.csv.gz", "rt", newline="") as f:
        for row in csv.DictReader(f):
            if not row["intime"] or not row["outtime"]:
                continue
            intime = datetime.strptime(row["intime"], fmt)
            outtime = datetime.strptime(row["outtime"], fmt)
            if (outtime - intime).total_seconds() < window_hours * 3600:
                continue
            start = intime.strftime(fmt)
            end = (intime + timedelta(hours=window_hours)).strftime(fmt)
            stays[row["stay_id"]] = (start, end)
            lo, hi = admissions.get(row["hadm_id"], (start, end))
            admissions[row["hadm_id"]] = (min(lo, start), max(hi, end))

    for table, folder, key, bounds in (
        ("chartevents", "icu", "stay_id", stays),
        ("labevents", "hosp", "hadm_id", admissions),
    ):
        source = root / folder / f"{table}.csv.gz"
        target = out / folder / f"{table}.csv.gz"
        target.parent.mkdir(parents=True, exist_ok=True)
        names = pv.open_csv(source).schema.names
        reader = pv.open_csv(
            source,
            read_options=pv.ReadOptions(block_size=1 << 26),
            parse_options=pv.ParseOptions(newlines_in_values=True),
            convert_options=pv.ConvertOptions(
                column_types={name: pa.string() for name in names}
            ),
        )
        kept = total = 0
        with pa.CompressedOutputStream(str(target), "gzip") as stream:
            writer = pv.CSVWriter(stream, reader.schema)
            for batch in reader:
                total += batch.num_rows
                batch = batch.filter(
                    pc.is_in(batch["itemid"], pa.array(sorted(items[table])))
                )
                ids = batch[key].to_pylist()
                lower = pa.array([bounds.get(i, ("~", ""))[0] for i in ids])
                upper = pa.array([bounds.get(i, ("~", ""))[1] for i in ids])
                time = batch["charttime"]
                batch = batch.filter(
                    pc.and_(pc.greater_equal(time, lower), pc.less(time, upper))
                )
                kept += batch.num_rows
                writer.write_batch(batch)
            writer.close()
        print(f"{table}: kept {kept:,} of {total:,} rows", flush=True)
    for name in ("hosp/patients", "hosp/admissions", "hosp/d_labitems", "icu/icustays"):
        link = out / f"{name}.csv.gz"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to((root / f"{name}.csv.gz").resolve())


def load(
    dataset: MIMIC4EHRDataset, n_time_bins: int, args: argparse.Namespace
) -> tuple[tuple[dict, dict, dict], Any]:
    """The normalized train / val / test tensors for one number of bins."""
    task = DuETTMortalityMIMIC4(n_time_bins=n_time_bins, pos_weight="balanced")
    split = (
        AuthorsSplit(args.split_file)
        if args.split_file
        else PatientSplit(ratios=(0.7, 0.15, 0.15), seed=0)
    )
    parts = dataset.set_task(task, split=split, num_workers=args.num_workers)
    data = [recipe.to_tensors(part) for part in parts]
    train = data[0]
    stats = fit_value_normalizer(train["values"].numpy(), train["counts"].numpy())
    continuous = [0, len(task.static_names) - 1]  # age, window length
    mean = train["static"][:, continuous].mean(0)
    std = train["static"][:, continuous].std(0).clamp(min=1e-6)
    for part in data:
        part["values"] = torch.from_numpy(
            normalize_values(part["values"].numpy(), part["counts"].numpy(), stats)
        )
        part["static"][:, continuous] = (part["static"][:, continuous] - mean) / std
    for name, part in zip(("train", "val", "test"), data):
        print(
            f"{name}: {len(part['mortality'])} stays, "
            f"{part['mortality'].float().mean():.4f} positive",
            flush=True,
        )
    return tuple(data), parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--root", default=None, help="MIMIC-IV folder; synthetic tables if omitted."
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--output", default="duett_mimic4")
    parser.add_argument(
        "--ablation", default="base", choices=[*recipe.ABLATIONS, "all"]
    )
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Only the first 1,000 patients (MIMIC4EHRDataset dev).",
    )
    parser.add_argument(
        "--prefilter",
        default=None,
        help="Write a copy of --root with only the task's events here (once) "
        "and load that; needed with ~13 GB of RAM.",
    )
    args = parser.parse_args()
    recipe.DEVICE = torch.device(args.device)
    synthetic = args.root is None
    if args.pretrain_epochs is None:
        args.pretrain_epochs = 5 if synthetic else 30
    if args.finetune_epochs is None:
        args.finetune_epochs = 8 if synthetic else 20
    if synthetic:
        args.batch_size, args.top_k = min(args.batch_size, 64), min(args.top_k, 2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        root = args.root
        if synthetic:
            root = str(Path(tmp) / "mimiciv")
            write_synthetic_mimic(Path(root))
        if args.prefilter and not synthetic:
            if not (Path(args.prefilter) / "icu" / "icustays.csv.gz").exists():
                prefilter_mimic(Path(root), Path(args.prefilter))
            root = args.prefilter
        dataset = MIMIC4EHRDataset(
            root=root,
            tables=TABLES,
            dev=args.dev,
            cache_dir=args.cache_dir
            or (str(Path(tmp) / "cache") if synthetic else None),
        )
        names = list(recipe.ABLATIONS) if args.ablation == "all" else [args.ablation]
        configs = list({c: None for n in names for c in recipe.ABLATIONS[n]})
        loaded: dict[int, tuple] = {}
        results = []
        for config in configs:
            if config.n_time_bins not in loaded:
                loaded[config.n_time_bins] = load(dataset, config.n_time_bins, args)
            data, samples = loaded[config.n_time_bins]
            print(f"== {config.name}", flush=True)
            result = recipe.fine_tune(samples, config, data, args)
            result["data"] = "synthetic_mimic4" if synthetic else "mimic4"
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
