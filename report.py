"""Звірка моделей на живих днях з журналу live/solar.db.

python report.py [з YYYY-MM-DD] [по YYYY-MM-DD]   (за замовчуванням — усі дні в журналі)
На Pi: ssh pi docker exec solar-forecast python report.py

Два погляди:
- ранковий прогноз — перший запуск дня з 05:00: «скільки сонця дасть сьогодні», як планувати день;
- останній перед годиною — найсвіжіший прогноз на кожну годину.
Лише чисті години (батарея не притискала сонце, записів досить) — як у перевірці.
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Kyiv"
DB = Path(__file__).parent / "live" / "solar.db"
NAMES = {
    "physics": "1. фізика + динамічна поправка",
    "lgb_v1": "2. LightGBM v1 + динамічна",
    "lgb_v2": "3. LightGBM v2 + динамічна",
    "physics_corr": "4. фізика + модель поправки",
}


def main():
    con = sqlite3.connect(DB)
    f = pd.read_sql("SELECT * FROM forecasts", con, parse_dates=["hour"])
    a = pd.read_sql("SELECT * FROM actuals", con, parse_dates=["hour"]).set_index("hour")
    f["hour"] = pd.to_datetime(f.hour, utc=True)
    a.index = pd.to_datetime(a.index, utc=True)
    f["run"] = pd.to_datetime(f.run_at, utc=True)
    f["day"] = f.hour.dt.tz_convert(TZ).dt.date
    if len(sys.argv) > 1:
        f = f[f.day >= pd.Timestamp(sys.argv[1]).date()]
    if len(sys.argv) > 2:
        f = f[f.day <= pd.Timestamp(sys.argv[2]).date()]
    clean = a[(a.enough == 1) & (a.clipped == 0) & a.pv_w.notna()]

    runs_local = f.run.dt.tz_convert(TZ)
    morning = f[runs_local.dt.hour >= 5]
    first = morning.groupby("day").run.transform("min")
    morning = morning[morning.run == first]
    latest = f[f.run <= f.hour].sort_values("run").groupby(["model", "hour"]).tail(1)

    for title, p in (("Ранковий прогноз на день", morning), ("Останній прогноз перед годиною", latest)):
        p = p.join(clean.pv_w, on="hour", how="inner")
        if p.empty:
            print(f"\n{title}: ще нема годин з фактом")
            continue
        rows = {}
        for m, g in p.groupby("model"):
            err = g.pred_w - g.pv_w
            rows[NAMES.get(m, m)] = {"годин": len(g), "днів": g.day.nunique(), "промах, Вт": round(err.abs().mean()),
                                     "промах, %": round(err.abs().mean() / g.pv_w.mean() * 100, 1), "зсув, Вт": round(err.mean())}
        print(f"\n{title} (чисті години):")
        print(pd.DataFrame(rows).T.sort_values("промах, Вт").to_string())

    p = morning.join(clean.pv_w, on="hour", how="inner")
    if not p.empty:
        d = p.pivot_table(index="day", columns="model", values="pred_w", aggfunc="sum") / 1000
        d.insert(0, "факт", p.drop_duplicates("hour").groupby("day").pv_w.sum() / 1000)
        d.columns = ["факт"] + [NAMES.get(c, c) for c in d.columns[1:]]
        print("\nПо днях, кВт·год за чисті години (ранковий прогноз):")
        print(d.round(1).to_string())


if __name__ == "__main__":
    main()
