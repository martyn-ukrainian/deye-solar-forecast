"""Чотири моделі сонця «сьогодні» для порівняння на живих днях. Навчання — train.py (на Mac), прогноз — live.py (на Pi).

Усі ознаки рахуються тими самими функціями, що й у ноутбуках (dataset, evaluate), — щоб на Pi було як на перевірці.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import data_io as io
import evaluate as ev

MODELS = {
    "physics": "фізика панелей + динамічна поправка",
    "lgb_v1": "LightGBM v1 (20 ознак) + динамічна поправка",
    "lgb_v2": "LightGBM v2 (+ ефективність положення, сніг та іній) + динамічна поправка",
    "physics_corr": "фізика панелей + модель поправки (2 роки)",
}
TILT, AZ = 10, 220  # площина панелей з ясних днів 2025 (07_model_v2)
SNOW = ["snow_on_panels_cm", "days_since_snow", "forecast__snow_depth", "night_min_temp", "night_max_humidity"]
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)


def features_v1():
    return ev.variants()["повний (20 ознак)"]


def features_v2():
    return features_v1() + ["orient_factor"] + SNOW


def add_features(df):
    """Ознаки поверх dataset.build(): сніг та іній, ефективність положення панелей."""
    df = ev.extra_features(df)
    df["orient_factor"] = ev.orient_factor(df.index, TILT, AZ).to_numpy()
    return df


def train(df):
    """Вчить усі 4 на всіх чистих годинах. Повертає набір для save()."""
    clean = df[df.enough & ~df.clipped]
    y = clean.pv_w
    orient = ev.fit_orientation(clean)
    boosters = {}
    for name, feats in (("lgb_v1", features_v1()), ("lgb_v2", features_v2())):
        boosters[name] = lgb.LGBMRegressor(**PARAMS).fit(clean[feats], y).booster_
    # Модель поправки вчиться на чесних прогнозах фізики (кожен місяць — моделлю без нього), як у 04_4
    oof = ev.oof_predict(df, ev.physics_fit_predict, ev.PHYSICS_COLS)
    F = ev.corr_features(df, oof, "sequence")
    rows = F.base > 50
    corr_feats = [c for c in F.columns if c != "base"]
    k = (y[rows] / F.base[rows]).clip(0, 2)
    boosters["physics_corr"] = lgb.LGBMRegressor(**ev.CORR_PARAMS).fit(
        F.loc[rows, corr_feats], k, sample_weight=F.base[rows]).booster_
    return {"orient": {k: float(v) for k, v in orient.items()}, "boosters": boosters, "corr_features": corr_feats,
            "trained_until": str(clean.index.max()), "hours": int(len(clean))}


def save(bundle, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, b in bundle["boosters"].items():
        b.save_model(str(folder / f"{name}.txt"))
    meta = {k: v for k, v in bundle.items() if k != "boosters"}
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def load(folder):
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    meta["boosters"] = {p.stem: lgb.Booster(model_file=str(p)) for p in folder.glob("*.txt")}
    return meta


def predict(bundle, df, today):
    """Прогноз на години дня `today` (дата за Києвом). df — таблиця за останні дні + сьогодні (з add_features).

    Поправка — з чистих годин попередніх днів (факт / прогноз), як у перевірці.
    Повертає таблицю: година × модель → без поправки, з поправкою, множник.
    """
    day = df.index.tz_convert(io.TZ).normalize()
    is_today = np.asarray(day.date == today)
    past_clean = np.asarray(df.enough & ~df.clipped & df.pv_w.notna()) & np.asarray(day.date < today)
    use = df[past_clean | is_today]
    y = use.pv_w.where(past_clean[past_clean | is_today])  # сьогоднішній факт у поправку не йде

    o = bundle["orient"]
    raw = {"physics": pd.Series(o["k"] * ev.poa(use, "forecast", o["tilt"], o["azimuth"]), index=use.index)}
    for name, feats in (("lgb_v1", features_v1()), ("lgb_v2", features_v2())):
        raw[name] = pd.Series(bundle["boosters"][name].predict(use[feats]), index=use.index)
    raw = {k: v.clip(lower=0) for k, v in raw.items()}

    out = []
    for name in ("physics", "lgb_v1", "lgb_v2"):
        fac = ev.weighted_factor(raw[name], y, 2)
        out.append(pd.DataFrame({"model": name, "raw_w": raw[name], "factor": fac, "pred_w": raw[name] * fac}))
    F = ev.corr_features(df.assign(pv_w=df.pv_w.where(pd.Series(past_clean, index=df.index))), raw["physics"], "sequence")
    k = np.clip(bundle["boosters"]["physics_corr"].predict(F[bundle["corr_features"]]), 0.3, 1.7)
    out.append(pd.DataFrame({"model": "physics_corr", "raw_w": raw["physics"], "factor": k, "pred_w": raw["physics"] * k}))
    res = pd.concat(out)
    return res[np.asarray(res.index.tz_convert(io.TZ).date == today)]
