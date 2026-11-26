"""Tests for DuETTMortalityMIMIC4, on synthetic patients only.

Authors:
    Hritish Mahajan (NetID: hritish2), Alex Jung (NetID: jhjung2)

Paper:
    Labach et al., "DuETT: Dual Event Time Transformer for Electronic Health
    Records," MLHC 2023. https://proceedings.mlr.press/v219/labach23a.html
"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DuETT
from pyhealth.tasks import (
    DuETTMortalityMIMIC4,
    fit_value_normalizer,
    normalize_values,
)

T0 = datetime(2150, 1, 1, 8, 0, 0)
MICU = "Medical Intensive Care Unit (MICU)"
TASK = DuETTMortalityMIMIC4()
NAMES = [v.name for v in TASK.variables]
HR, TEMP, ALARMS = (
    NAMES.index("Heart rate"),
    NAMES.index("Temperature"),
    NAMES.index("Alarms On"),
)
POTASSIUM = NAMES.index("Potassium")
STATIC = TASK.static_names


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _chart(hours, itemid, valuenum=None, value=None, stay_id=10, hadm_id=1):
    value = value if value is not None else valuenum
    return _event(
        "chartevents",
        T0 + timedelta(hours=hours),
        stay_id=stay_id,
        hadm_id=hadm_id,
        itemid=itemid,
        valuenum=valuenum,
        value=value,
    )


def _lab(hours, itemid, valuenum, hadm_id=1):
    return _event(
        "labevents",
        T0 + timedelta(hours=hours),
        hadm_id=hadm_id,
        itemid=itemid,
        valuenum=valuenum,
        value=valuenum,
    )


def _patient(
    events,
    *,
    anchor_age=60,
    anchor_year=2150,
    stays=((10, 1, 72),),
    expire=0,
    language="ENGLISH",
    race="WHITE - RUSSIAN",
):
    """A patient with admission 1 at T0 - 2 h and the given ICU stays."""
    rows = [
        _event(
            "patients", None, gender="M", anchor_age=anchor_age, anchor_year=anchor_year
        ),
        _event(
            "admissions",
            T0 - timedelta(hours=2),
            hadm_id=1,
            hospital_expire_flag=expire,
            language=language,
            marital_status="MARRIED",
            insurance="Medicare",
            admission_location="EMERGENCY ROOM",
            admission_type="EW EMER.",
            race=race,
        ),
    ]
    for stay_id, hadm_id, hours in stays:
        start = T0 if stay_id == 10 else T0 + timedelta(hours=100)
        rows.append(
            _event(
                "icustays",
                start,
                stay_id=stay_id,
                hadm_id=hadm_id,
                first_careunit=MICU,
                outtime=(start + timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
    rows.extend(events)
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


class TestDuETTMortalityMIMIC4(unittest.TestCase):
    def test_variables(self):
        self.assertEqual(len(TASK.variables), 114)
        itemids = [(v.table, i) for v in TASK.variables for i in v.itemids]
        self.assertEqual(len(itemids), len(set(itemids)))  # no item counted twice
        self.assertEqual(len(STATIC), 49)

    def test_sample_values_counts_and_times(self):
        patient = _patient(
            [
                _chart(0.5, 220045, 80),
                _chart(1.0, 220045, 90),  # bin 0: last 90
                _chart(47.9, 220045, 70),  # last bin
                _chart(50.0, 220045, 200),  # after the window: dropped
                _chart(3.2, 223761, 98.6),  # Fahrenheit -> Celsius
                _chart(4.0, 224641, value="On"),  # text item: presence only
                _lab(2.0, 50971, 4.1),  # lab matched on the admission
            ]
        )
        (sample,) = TASK(patient)
        self.assertEqual(sample["values"].shape, (32, 114))
        self.assertEqual(sample["counts"].shape, (32, 114))
        self.assertEqual(sample["values"][0, HR], 90)
        self.assertEqual(sample["counts"][0, HR], 2)
        self.assertEqual(sample["values"][31, HR], 70)
        self.assertEqual(sample["counts"][:, HR].sum(), 3)
        self.assertAlmostEqual(float(sample["values"][2, TEMP]), 37.0, places=4)
        self.assertEqual(sample["values"][2, ALARMS], 1.0)
        self.assertEqual(sample["values"][1, POTASSIUM], np.float32(4.1))
        self.assertEqual(sample["counts"].sum(), 6)
        np.testing.assert_allclose(sample["times"][[0, -1]], [1.5 / 24, 2.0])
        self.assertEqual(sample["mortality"], 0)
        self.assertEqual(sample["visit_id"], "10")

    def test_static(self):
        (sample,) = TASK(
            _patient([_chart(24.0, 220045, 80)], anchor_age=60, anchor_year=2148)
        )
        static = dict(zip(STATIC, sample["static"]))
        self.assertEqual(static["age"], 62)  # 60 + (2150 - 2148)
        self.assertEqual(static["gender_male"], 1)
        self.assertEqual(static["language_english"], 1)
        self.assertEqual(static["race=WHITE"], 1)
        self.assertEqual(static["insurance=MEDICARE"], 1)
        self.assertEqual(static[f"first_careunit={MICU}"], 1)
        self.assertAlmostEqual(static["observation_window_length"], 0.5)
        (other,) = TASK(
            _patient([_chart(24.0, 220045, 80)], race="UNKNOWN", language="?")
        )
        other = dict(zip(STATIC, other["static"]))
        self.assertEqual(other["race=OTHER"], 1)
        self.assertEqual(other["language_english"], 0)

    def test_label(self):
        (sample,) = TASK(_patient([_chart(1.0, 220045, 80)], expire=1))
        self.assertEqual(sample["mortality"], 1)

    def test_exclusions(self):
        events = [_chart(1.0, 220045, 80)]
        self.assertEqual(TASK(_patient(events, anchor_age=17)), [])  # minor
        self.assertEqual(TASK(_patient(events, stays=((10, 1, 47),))), [])  # < 48 h
        self.assertEqual(TASK(_patient([])), [])  # no events
        self.assertEqual(TASK(_patient([_chart(49.0, 220045, 80)])), [])  # none < 48 h
        self.assertEqual(TASK(_patient([_chart(1.0, 999999, 1)])), [])  # unused item

    def test_multiple_stays(self):
        events = [
            _chart(1.0, 220045, 80),
            _chart(101.0, 220045, 95, stay_id=11),
            _lab(102.0, 50971, 3.9),  # only in stay 11's window
        ]
        samples = TASK(_patient(events, stays=((10, 1, 72), (11, 1, 60))))
        self.assertEqual([s["visit_id"] for s in samples], ["10", "11"])
        self.assertEqual(samples[0]["counts"][:, POTASSIUM].sum(), 0)
        self.assertEqual(samples[1]["counts"][:, POTASSIUM].sum(), 1)
        self.assertEqual(samples[1]["counts"][:, HR].sum(), 1)

    def test_n_time_bins_and_window(self):
        task = DuETTMortalityMIMIC4(n_time_bins=4, window_hours=24)
        (sample,) = task(_patient([_chart(7.0, 220045, 80), _chart(30, 220045, 1)]))
        self.assertEqual(sample["values"].shape, (4, 114))
        self.assertEqual(sample["counts"][1, HR], 1)  # 7 h of 24 h -> bin 1
        self.assertEqual(sample["counts"].sum(), 1)
        np.testing.assert_allclose(sample["times"], [0.25, 0.5, 0.75, 1.0])
        with self.assertRaises(ValueError):
            DuETTMortalityMIMIC4(n_time_bins=0)

    def test_pre_filter(self):
        df = pl.DataFrame(
            {
                "event_type": ["chartevents", "chartevents", "labevents", "icustays"],
                "chartevents/itemid": ["220045", "1", None, None],
                "labevents/itemid": [None, None, "2", None],
            }
        ).lazy()
        kept = TASK.pre_filter(df).collect()
        self.assertEqual(kept["event_type"].to_list(), ["chartevents", "icustays"])

    def test_pos_weight(self):
        task = DuETTMortalityMIMIC4(pos_weight="balanced")
        self.assertEqual(task.output_schema["mortality"][1], {"pos_weight": "balanced"})
        self.assertEqual(TASK.output_schema, {"mortality": "binary"})

    def test_normalizer(self):
        values = np.array([[[1.0, 1.0], [2.0, 1.0], [3.0, 0.0], [100.0, 1.0]]])
        counts = np.array([[[1, 1], [1, 1], [1, 0], [1, 1]]], dtype=float)
        stats = fit_value_normalizer(values, counts)
        self.assertEqual(stats["upper"][0], 5.5)  # median 2.5 + 3 * MAD 1.0
        self.assertEqual(stats["upper"][1], np.inf)  # MAD 0: not clipped
        out = normalize_values(values, counts, stats)
        self.assertEqual(out[0, 2, 1], 0.0)  # unobserved stays 0
        self.assertLess(out[0, 3, 0], 2.0)  # the outlier is clipped
        self.assertAlmostEqual(float(out[0, [0, 1, 2, 3], 0].mean()), 0.0, places=5)

    def test_samples_run_through_duett(self):
        patients = [
            _patient([_chart(h, 220045, 80 + h) for h in (1.0, 20.0)], expire=e)
            for e in (0, 1, 0)
        ]
        samples = [s for p in patients for s in TASK(p)]
        for i, s in enumerate(samples):
            s["patient_id"] = f"p{i}"
        dataset = create_sample_dataset(samples, TASK.input_schema, TASK.output_schema)
        model = DuETT(
            dataset,
            n_time_bins=32,
            d_embedding=4,
            d_feedforward=8,
            value_key="values",
            time_key="times",
            count_key="counts",
            static_key="static",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (3, 1))
        out["loss"].backward()


if __name__ == "__main__":
    unittest.main()
