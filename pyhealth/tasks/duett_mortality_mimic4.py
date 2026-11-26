"""MIMIC-IV ICU mortality from the first 48 hours, binned for DuETT.

Authors:
    Hritish Mahajan (NetID: hritish2, hritish2@illinois.edu)
    Alex Jung (NetID: jhjung2, jhjung2@illinois.edu)

Paper:
    Labach et al., "DuETT: Dual Event Time Transformer for Electronic Health
    Records," MLHC 2023, PMLR 219:403-422.
    https://proceedings.mlr.press/v219/labach23a.html

Description:
    The paper's MIMIC-IV in-hospital mortality cohort (Section 4 and
    Appendix C): one sample per ICU stay, the chart and lab events of the
    first 48 hours binned into a time x variable grid of last values and
    observation counts, the paper's static variables, and whether the
    patient died later in the hospital stay. The authors released only their
    patient split for MIMIC-IV, not the cohort code, so the choices the
    paper leaves open are documented on :class:`DuETTMortalityMIMIC4`.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, ClassVar, NamedTuple

import numpy as np
import polars as pl

from .base_task import BaseTask

_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
_F_TO_C = (5 / 9, -160 / 9)  # (scale, offset): C = F * 5/9 - 160/9
_LBS_TO_KG = (0.453592, 0.0)
_INCH_TO_CM = (2.54, 0.0)


class DuETTVariable(NamedTuple):
    """One time-series variable: the item IDs whose events it collects.

    ``conversions`` maps an item ID to ``(scale, offset)``, applied as
    ``value * scale + offset`` to put all items of a variable in one unit.
    """

    name: str
    table: str
    itemids: tuple[int, ...]
    conversions: tuple[tuple[int, tuple[float, float]], ...] = ()


def _chart(name: str, *itemids: int, **conversions: tuple[float, float]):
    converted = tuple((int(k[1:]), v) for k, v in conversions.items())
    return DuETTVariable(name, "chartevents", itemids, converted)


# Harutyunyan et al. (2019) variables, with item IDs grouped (Table 4). GCS
# total has no MIMIC-IV item ID and is left out, as in the paper.
HARUTYUNYAN_VARIABLES: tuple[DuETTVariable, ...] = (
    _chart("Capillary refill rate", 223951, 224308),
    _chart("Diastolic blood pressure", 220051, 220180, 224643, 225310, 227242),
    _chart("Fraction inspired oxygen", 223835),
    _chart("Glasgow coma scale eye opening", 220739),
    _chart("Glasgow coma scale verbal response", 223900),
    _chart("Glasgow coma scale motor response", 223901),
    _chart("Glucose", 220621, 225664, 226537, 228388),
    _chart("Heart rate", 220045),
    _chart("Height", 226707, 226730, i226707=_INCH_TO_CM),
    _chart("Mean blood pressure", 220052, 220181),
    _chart("Oxygen saturation", 220227, 220277),
    _chart("Respiratory rate", 220210, 223851, 224689, 224690),
    _chart("Systolic blood pressure", 220050, 220179, 224167, 225309, 227243),
    _chart("Temperature", 223761, 223762, 224027, i223761=_F_TO_C),
    _chart("Weight", 224639, 226512, 226531, i226531=_LBS_TO_KG),
    _chart("pH", 220274, 220734, 223830, 228243),
)

# The paper's "additional time series variables" from chartevents (85 item
# IDs, Table 4). Items already in a Harutyunyan group above are not repeated.
ADDITIONAL_CHART_ITEMS: dict[int, str] = {
    220045: "Heart Rate",
    220277: "O2 saturation pulseoxymetry",
    220210: "Respiratory Rate",
    220739: "GCS - Eye Opening",
    223900: "GCS - Verbal Response",
    223901: "GCS - Motor Response",
    224641: "Alarms On",
    224168: "Parameters Checked",
    220047: "Heart Rate Alarm - Low",
    220046: "Heart rate Alarm - High",
    220181: "Non Invasive Blood Pressure mean",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    223770: "O2 Saturation Pulseoxymetry Alarm - Low",
    223769: "O2 Saturation Pulseoxymetry Alarm - High",
    224161: "Resp Alarm - High",
    224162: "Resp Alarm - Low",
    224054: "Braden Sensory Perception",
    224057: "Braden Mobility",
    224055: "Braden Moisture",
    224056: "Braden Activity",
    224058: "Braden Nutrition",
    224059: "Braden Friction/Shear",
    226253: "SpO2 Desat Limit",
    223761: "Temperature Fahrenheit",
    227344: "IV/Saline lock",
    227345: "Gait/Transferring",
    227343: "Ambulatory aid",
    227346: "Mental status",
    227342: "Secondary diagnosis",
    227341: "History of falling (within 3 mnths)",
    227442: "Potassium (serum)",
    220645: "Sodium (serum)",
    220602: "Chloride (serum)",
    220615: "Creatinine (serum)",
    225624: "BUN",
    227443: "HCO3 (serum)",
    227073: "Anion gap",
    220545: "Hematocrit (serum)",
    220621: "Glucose (serum)",
    220228: "Hemoglobin",
    227457: "Platelet Count",
    220546: "WBC",
    220635: "Magnesium",
    223752: "Non-Invasive Blood Pressure Alarm - Low",
    223751: "Non-Invasive Blood Pressure Alarm - High",
    225677: "Phosphorous",
    225625: "Calcium non-ionized",
    223791: "Pain Level",
    228096: "Richmond-RAS Scale",
    227465: "Prothrombin time",
    227467: "INR",
    227466: "PTT",
    223951: "Capillary Refill R",
    224308: "Capillary Refill L",
    226531: "Admission Weight (lbs.)",
    228299: "Goal Richmond-RAS Scale",
    228305: "ST Segment Monitoring On",
    223834: "O2 Flow",
    225664: "Glucose finger stick (range 70-100)",
    224409: "Pain Level Response",
    225103: "Intravenous / IV access prior to admission",
    227368: "20 Gauge Dressing Occlusive",
    228412: "Strength R Arm",
    228409: "Strength L Arm",
    228411: "Strength R Leg",
    228410: "Strength L Leg",
    226138: "20 Gauge placed in outside facility",
    228236: "Insulin pump",
    225092: "Self ADL",
    228100: "20 Gauge placed in the field",
    225094: "History of slips / falls",
    227349: "High risk (>51) interventions",
    225668: "Lactic Acid",
    228648: "Home TF",
    225106: "ETOH",
    228649: "Pressure Ulcer Present",
    225118: "Difficulty swallowing",
    227367: "18 Gauge Dressing Occlusive",
    226137: "18 Gauge placed in outside facility",
    225184: "Eye Care",
    225087: "Visual / hearing deficit",
    225113: "Currently experiencing pain",
    225126: "Dialysis patient",
    224639: "Daily Weight",
}

# The paper's 29 labevents variables (Table 4).
LAB_ITEMS: dict[int, str] = {
    50971: "Potassium",
    50902: "Chloride",
    50983: "Sodium",
    50912: "Creatinine",
    51006: "Urea Nitrogen",
    50882: "Bicarbonate",
    50868: "Anion Gap",
    50931: "Glucose (lab)",
    51221: "Hematocrit",
    51265: "Platelet Count (lab)",
    51301: "White Blood Cells",
    51222: "Hemoglobin (lab)",
    51279: "Red Blood Cells",
    51250: "MCV",
    51248: "MCH",
    51249: "MCHC",
    51277: "RDW",
    50960: "Magnesium (lab)",
    50970: "Phosphate",
    50893: "Calcium, Total",
    51274: "PT",
    51237: "INR(PT)",
    51275: "PTT (lab)",
    50820: "pH (lab)",
    50813: "Lactate",
    50802: "Base Excess",
    50821: "pO2",
    50818: "pCO2",
    50804: "Calculated Total CO2",
}


def _variables() -> tuple[DuETTVariable, ...]:
    grouped = {i for v in HARUTYUNYAN_VARIABLES for i in v.itemids}
    extra = tuple(
        _chart(name, itemid)
        for itemid, name in ADDITIONAL_CHART_ITEMS.items()
        if itemid not in grouped
    )
    labs = tuple(
        DuETTVariable(name, "labevents", (itemid,))
        for itemid, name in LAB_ITEMS.items()
    )
    return HARUTYUNYAN_VARIABLES + extra + labs


# Category vocabularies of the static one-hots (MIMIC-IV v2.2 and v3.1
# values); anything else, including a missing value, is "OTHER".
STATIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "marital_status": ("MARRIED", "SINGLE", "WIDOWED", "DIVORCED"),
    "insurance": ("MEDICARE", "MEDICAID", "PRIVATE", "OTHER"),
    "admission_location": (
        "EMERGENCY ROOM",
        "PHYSICIAN REFERRAL",
        "TRANSFER FROM HOSPITAL",
        "WALK-IN/SELF REFERRAL",
        "CLINIC REFERRAL",
        "PROCEDURE SITE",
        "PACU",
        "TRANSFER FROM SKILLED NURSING FACILITY",
        "INFORMATION NOT AVAILABLE",
    ),
    "admission_type": (
        "EW EMER.",
        "URGENT",
        "SURGICAL SAME DAY ADMISSION",
        "DIRECT EMER.",
        "ELECTIVE",
        "OBSERVATION ADMIT",
        "EU OBSERVATION",
        "DIRECT OBSERVATION",
        "AMBULATORY OBSERVATION",
    ),
    "race": ("WHITE", "BLACK", "HISPANIC", "ASIAN"),
    "first_careunit": (
        "Medical Intensive Care Unit (MICU)",
        "Surgical Intensive Care Unit (SICU)",
        "Medical/Surgical Intensive Care Unit (MICU/SICU)",
        "Cardiac Vascular Intensive Care Unit (CVICU)",
        "Coronary Care Unit (CCU)",
        "Trauma SICU (TSICU)",
        "Neuro Surgical Intensive Care Unit (Neuro SICU)",
        "Neuro Intermediate",
        "Neuro Stepdown",
    ),
}


def _race_group(race: str | None) -> str | None:
    """Groups MIMIC-IV's ~30 race values by their prefix."""
    if race is None:
        return None
    race = race.upper()
    if race.startswith(("HISPANIC", "SOUTH AMERICAN")):
        return "HISPANIC"
    for group in ("WHITE", "BLACK", "ASIAN"):
        if race.startswith(group):
            return group
    return None


def _static_names() -> list[str]:
    names = ["age", "gender_male", "language_english"]
    for field, categories in STATIC_CATEGORIES.items():
        names += [f"{field}={c}" for c in categories] + [f"{field}=OTHER"]
    return names + ["observation_window_length"]


class DuETTMortalityMIMIC4(BaseTask):
    """In-hospital mortality from the first 48 hours of an ICU stay (DuETT).

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` with the tables
    ``patients``, ``admissions``, ``icustays``, ``chartevents`` and
    ``labevents``.

    **Cohort** (paper Section 4 and Appendix C), one sample per ICU stay:
    stays of patients 18 or older (``anchor_age`` plus the years from
    ``anchor_year`` to the admission), at least ``window_hours`` long, with
    at least one chart or lab event of the task's variables in the first
    ``window_hours``. Unlike Harutyunyan et al. (2019), patients with several
    ICU stays or transfers are kept. The label is the admission's
    ``hospital_expire_flag``; since stays are at least ``window_hours`` long,
    a death in the hospital comes after the window.

    **Variables.** The 16 Harutyunyan variables (their item IDs grouped,
    Fahrenheit, inches and pounds converted to Celsius, centimetres and
    kilograms), then the paper's other chart items, then its 29 lab items,
    each as one variable (:attr:`variables`), 114 in all. Of the paper's 85
    chart items, 16 are also in a Harutyunyan group; those are counted once, in
    the group, so no observation enters two variables.

    **Grid.** The window is split into ``n_time_bins`` equal bins
    (``floor(hours / window_hours * n_time_bins)``). ``values`` holds the
    last value of each (bin, variable), 0 when unobserved; ``counts`` the
    number of events; ``times`` each bin's end time in days. Lab events have
    no ICU stay ID and are matched on the hospital admission and the window.

    **Static features** (:attr:`static_names`): age, male, English language,
    one-hots of marital status, insurance, admission location, admission
    type, race (grouped by prefix) and first care unit, each with an
    ``OTHER`` bucket, and the observation window length (hours from ICU
    admission to the last event in the window, divided by
    ``window_hours``).

    Choices the paper leaves open:

    - Text-valued chart items (alarms, flags, "Ambulatory aid", ...) have no
      ``valuenum``; such events have value 1, so they carry presence and
      counts.
    - Events at the same time each count; the last value is the last event
      in time order (ties in dataset order).
    - The paper's 10 listed static variables (the text says 9) include the
      observation window length, which it does not define; see above.
    - Values are raw: normalize them with training-set statistics
      (:func:`fit_value_normalizer`, :func:`normalize_values`), as the
      paper does (clip at 3 median absolute deviations, then z-score).

    Args:
        n_time_bins: Number of time bins (32 in the paper).
        window_hours: Length of the observation window and the minimum
            stay length, in hours (48 in the paper).
        pos_weight: ``pos_weight`` of the binary label processor, e.g.
            ``"balanced"`` for the paper's class weighting; ``None`` for none.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimiciv",
        ...     tables=["icustays", "chartevents", "labevents"],
        ... )  # doctest: +SKIP
        >>> samples = dataset.set_task(DuETTMortalityMIMIC4())  # doctest: +SKIP
        >>> samples[0]["values"].shape  # doctest: +SKIP
        torch.Size([32, 114])
    """

    task_name: str = "DuETTMortalityMIMIC4"
    input_schema: ClassVar[dict[str, str]] = {
        "values": "tensor",
        "counts": "tensor",
        "times": "tensor",
        "static": "tensor",
    }
    output_schema: ClassVar[dict[str, Any]] = {"mortality": "binary"}
    variables: ClassVar[tuple[DuETTVariable, ...]] = _variables()
    static_names: ClassVar[list[str]] = _static_names()

    def __init__(
        self,
        n_time_bins: int = 32,
        window_hours: float = 48.0,
        pos_weight: float | str | None = None,
    ) -> None:
        super().__init__()
        if n_time_bins < 1 or window_hours <= 0:
            raise ValueError("n_time_bins and window_hours must be positive.")
        self.n_time_bins = int(n_time_bins)
        self.window_hours = float(window_hours)
        if pos_weight is not None:
            self.output_schema = {"mortality": ("binary", {"pos_weight": pos_weight})}
        rows = []
        for index, variable in enumerate(self.variables):
            conversions = dict(variable.conversions)
            for itemid in variable.itemids:
                scale, offset = conversions.get(itemid, (1.0, 0.0))
                rows.append((variable.table, itemid, index, scale, offset))
        self._items = pl.DataFrame(
            rows,
            schema={
                "table": pl.String,
                "itemid": pl.Int64,
                "variable": pl.Int64,
                "scale": pl.Float64,
                "offset": pl.Float64,
            },
            orient="row",
        )

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps only the chart and lab events of the task's item IDs.

        ``chartevents`` has hundreds of millions of rows; filtering before
        the patients are built keeps memory in check.
        """
        columns = df.collect_schema().names()
        keep = pl.lit(True)
        for table in ("chartevents", "labevents"):
            if f"{table}/itemid" not in columns:
                continue
            ids = self._items.filter(pl.col("table") == table)["itemid"]
            keep = keep & (
                (pl.col("event_type") != table)
                | pl.col(f"{table}/itemid").is_in([str(i) for i in ids])
            )
        return df.filter(keep)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        info = demographics[0].attr_dict
        anchor_age = _float(info.get("anchor_age"))
        anchor_year = _float(info.get("anchor_year"))
        male = 1.0 if info.get("gender") == "M" else 0.0

        admissions = {
            int(row["admissions/hadm_id"]): row
            for row in patient.get_events(
                event_type="admissions", return_df=True
            ).iter_rows(named=True)
        }
        events = self._events(patient)
        samples = []
        stays = patient.get_events(event_type="icustays", return_df=True)
        for stay in stays.iter_rows(named=True):
            stay_id = int(stay["icustays/stay_id"])
            hadm_id = int(stay["icustays/hadm_id"])
            intime = stay["timestamp"]
            outtime = _parse_time(stay["icustays/outtime"])
            admission = admissions.get(hadm_id)
            if intime is None or outtime is None or admission is None:
                continue
            if (outtime - intime).total_seconds() / 3600 < self.window_hours:
                continue
            age = anchor_age + admission["timestamp"].year - anchor_year
            if math.isnan(age) or age < 18:
                continue
            label = _float(admission.get("admissions/hospital_expire_flag"))
            if math.isnan(label):
                continue
            window = self._window(events, stay_id, hadm_id, intime)
            if window.height == 0:
                continue
            values, counts, last_hour = self._grid(window)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(stay_id),
                    "values": values,
                    "counts": counts,
                    "times": self._bin_ends(),
                    "static": self._static(
                        age, male, admission, stay, last_hour / self.window_hours
                    ),
                    "mortality": int(label),
                }
            )
        return samples

    def _events(self, patient: Any) -> pl.DataFrame:
        """The patient's task events: (timestamp, stay_id, hadm_id, variable, value)."""
        frames = []
        for table in ("chartevents", "labevents"):
            df = patient.get_events(event_type=table, return_df=True)
            if df.height == 0:
                continue
            stay = (
                pl.col(f"{table}/stay_id").cast(pl.Int64, strict=False)
                if f"{table}/stay_id" in df.columns
                else pl.lit(None, dtype=pl.Int64)
            )
            numeric = pl.col(f"{table}/valuenum").cast(pl.Float64, strict=False)
            text = pl.col(f"{table}/value")
            frames.append(
                df.select(
                    pl.col("timestamp"),
                    stay.alias("stay_id"),
                    pl.col(f"{table}/hadm_id")
                    .cast(pl.Int64, strict=False)
                    .alias("hadm_id"),
                    pl.lit(table).alias("table"),
                    pl.col(f"{table}/itemid")
                    .cast(pl.Int64, strict=False)
                    .alias("itemid"),
                    pl.when(numeric.is_not_null() & numeric.is_not_nan())
                    .then(numeric)
                    .when(text.is_not_null())
                    .then(pl.lit(1.0))  # text-valued: presence only
                    .otherwise(None)
                    .alias("value"),
                )
            )
        schema = {
            "timestamp": pl.Datetime("ms"),
            "stay_id": pl.Int64,
            "hadm_id": pl.Int64,
            "variable": pl.Int64,
            "value": pl.Float64,
        }
        if not frames:
            return pl.DataFrame(schema=schema)
        return (
            pl.concat(frames, how="vertical_relaxed")
            .join(self._items, on=["table", "itemid"], how="inner")
            .filter(pl.col("value").is_not_null())
            .with_columns(pl.col("value") * pl.col("scale") + pl.col("offset"))
            .select(list(schema))
        )

    def _window(
        self, events: pl.DataFrame, stay_id: int, hadm_id: int, intime: datetime
    ) -> pl.DataFrame:
        """The stay's events in the window, with hours since ICU admission."""
        start = pl.lit(intime).cast(events.schema["timestamp"])
        hours = (pl.col("timestamp") - start).dt.total_seconds() / 3600
        own = (pl.col("stay_id") == stay_id) | (
            pl.col("stay_id").is_null() & (pl.col("hadm_id") == hadm_id)
        )
        return (
            events.filter(own)
            .with_columns(hours.alias("hours"))
            .filter((pl.col("hours") >= 0) & (pl.col("hours") < self.window_hours))
            .sort("hours", maintain_order=True)
        )

    def _grid(self, window: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
        """Last value and count per (bin, variable), and the last event hour."""
        n_vars = len(self.variables)
        values = np.zeros((self.n_time_bins, n_vars), dtype=np.float32)
        counts = np.zeros((self.n_time_bins, n_vars), dtype=np.float32)
        hours = window["hours"].to_numpy()
        bins = np.floor(hours / self.window_hours * self.n_time_bins).astype(int)
        bins = np.clip(bins, 0, self.n_time_bins - 1)
        variables = window["variable"].to_numpy()
        np.add.at(counts, (bins, variables), 1.0)
        values[bins, variables] = window["value"].to_numpy()  # last write wins
        return values, counts, float(hours.max())

    def _bin_ends(self) -> np.ndarray:
        steps = np.arange(1, self.n_time_bins + 1, dtype=np.float32)
        return steps / self.n_time_bins * np.float32(self.window_hours / 24)

    @staticmethod
    def _static(
        age: float,
        male: float,
        admission: dict[str, Any],
        stay: dict[str, Any],
        window_length: float,
    ) -> np.ndarray:
        language = admission.get("admissions/language") or ""
        fields = {
            "marital_status": admission.get("admissions/marital_status"),
            "insurance": admission.get("admissions/insurance"),
            "admission_location": admission.get("admissions/admission_location"),
            "admission_type": admission.get("admissions/admission_type"),
            "race": _race_group(admission.get("admissions/race")),
            "first_careunit": stay.get("icustays/first_careunit"),
        }
        static = [age, male, float(language.strip().upper() == "ENGLISH")]
        for field, categories in STATIC_CATEGORIES.items():
            value = fields[field]
            if value is not None and field != "first_careunit":
                value = value.strip().upper()
            one_hot = [float(value == c) for c in categories]
            static += one_hot + [float(not any(one_hot))]
        static.append(window_length)
        return np.array(static, dtype=np.float32)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_time(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), _TIME_FORMAT)
    except ValueError:
        return None


def fit_value_normalizer(
    values: np.ndarray, counts: np.ndarray, n_mad: float = 3.0
) -> dict[str, np.ndarray]:
    """Per-variable clipping bounds and z-score statistics, from training data.

    Follows the paper: values are clipped at ``n_mad`` median absolute
    deviations from the median, then z-scored. Only observed cells are used.
    A variable with a MAD of 0 (e.g. text items, all 1) is not clipped.

    Args:
        values: ``[N, T, V]`` raw last values of the training samples.
        counts: ``[N, T, V]`` their counts.
        n_mad: Clipping threshold in median absolute deviations.

    Returns:
        ``lower``, ``upper``, ``mean`` and ``std``, each ``[V]``.

    Examples:
        >>> v = np.array([[[1.0], [2.0], [3.0], [100.0]]])
        >>> stats = fit_value_normalizer(v, np.ones_like(v))
        >>> float(stats["upper"][0])
        5.5
    """
    n_vars = values.shape[-1]
    flat_v = values.reshape(-1, n_vars)
    observed = counts.reshape(-1, n_vars) > 0
    stats = {k: np.zeros(n_vars) for k in ("lower", "upper", "mean")}
    stats["std"] = np.ones(n_vars)
    for v in range(n_vars):
        x = flat_v[observed[:, v], v].astype(np.float64)
        if len(x) == 0:
            stats["lower"][v], stats["upper"][v] = -np.inf, np.inf
            continue
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        bound = n_mad * mad if mad > 0 else np.inf
        stats["lower"][v], stats["upper"][v] = median - bound, median + bound
        x = np.clip(x, stats["lower"][v], stats["upper"][v])
        stats["mean"][v] = x.mean()
        stats["std"][v] = x.std() or 1.0
    return stats


def normalize_values(
    values: np.ndarray, counts: np.ndarray, stats: dict[str, np.ndarray]
) -> np.ndarray:
    """Clips and z-scores observed cells; unobserved cells stay 0.

    Args:
        values: ``[..., V]`` raw last values.
        counts: ``[..., V]`` their counts.
        stats: From :func:`fit_value_normalizer`.

    Returns:
        The normalized values, as float32.

    Examples:
        >>> v = np.array([[[1.0], [2.0], [3.0], [100.0]]])
        >>> c = np.ones_like(v)
        >>> bool(normalize_values(v, c, fit_value_normalizer(v, c)).max() < 2)
        True
    """
    clipped = np.clip(values, stats["lower"], stats["upper"])
    z = (clipped - stats["mean"]) / stats["std"]
    return np.where(counts > 0, z, 0.0).astype(np.float32)
