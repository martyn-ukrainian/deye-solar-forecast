"""На Pi: щогодини прогноз сонця на сьогодні від усіх 4 моделей і факт — у журнал live/solar.db.

Нічого не шле. Порівняння — report.py.
Погода: Open-Meteo (свіжий прогноз + прогнози за добу і за дві, як у навчанні). Факт: хмара Deye, кадри 5 хв.
"""
import json
import shutil
import sqlite3
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

import data_io as io

HERE = Path(__file__).parent
LIVE = HERE / "live"
io.DATA = LIVE  # dataset і evaluate читають погоду, Deye і станцію звідси

import dataset  # noqa: E402
import fetch_deye  # noqa: E402
import fetch_weather  # noqa: E402
import models  # noqa: E402

WEATHER_DAYS = 40  # «сніг на панелях» накопичується — потрібна історія погоди
DEYE_DAYS = 15     # поправка дивиться на 7–14 днів назад
EVERY_S = 3600


def log(*a):
    print(f"{datetime.now():%Y-%m-%d %H:%M}", *a, flush=True)


def get_weather(today, station):
    fetch_weather.URLS["forecast"] = "https://api.open-meteo.com/v1/forecast"  # свіжий прогноз; історичний — його ж архів
    (LIVE / "weather").mkdir(parents=True, exist_ok=True)
    for kind in ("forecast", "day1", "day2"):
        df = fetch_weather.fetch(kind, today - timedelta(days=WEATHER_DAYS), today + timedelta(days=1), station)
        df.to_parquet(LIVE / "weather" / f"{kind}.parquet")


def get_deye(today):
    out = LIVE / "deye"
    out.mkdir(parents=True, exist_ok=True)
    tok = fetch_deye.token()
    for i in range(DEYE_DAYS, -1, -1):
        d = today - timedelta(days=i)
        path = out / f"{d}.json"
        if path.exists() and d < today - timedelta(days=1):
            continue  # минулі дні вже не міняються (вчора — перекачуємо: хмара дописує із запізненням)
        items = fetch_deye.api(tok, "/v1.0/station/history",
                               {"stationId": fetch_deye.STATION, "startAt": str(d), "endAt": str(d), "granularity": 1})
        path.write_text(json.dumps(items.get("stationDataItems", [])))
        time.sleep(fetch_deye.PAUSE_S)
    for p in out.glob("*.json"):
        if date.fromisoformat(p.stem) < today - timedelta(days=DEYE_DAYS + 5):
            p.unlink()


def db():
    con = sqlite3.connect(LIVE / "solar.db")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS forecasts (run_at TEXT, model TEXT, hour TEXT, raw_w REAL, factor REAL, pred_w REAL,
                                              PRIMARY KEY (run_at, model, hour));
        CREATE TABLE IF NOT EXISTS actuals (hour TEXT PRIMARY KEY, pv_w REAL, clipped INTEGER, enough INTEGER, updated_at TEXT);
    """)
    return con


def once(bundle):
    now = datetime.now().astimezone()
    today = now.date()
    station = io.station()
    get_weather(today, station)
    get_deye(today)
    start = pd.Timestamp(today - timedelta(days=DEYE_DAYS), tz=io.TZ).tz_convert("UTC")
    end = pd.Timestamp(today + timedelta(days=1), tz=io.TZ).tz_convert("UTC")
    idx = pd.date_range(start, end, freq="1h", inclusive="left")
    df = models.add_features(dataset.build(index=idx, archive=False))
    run_at = now.isoformat(timespec="minutes")
    con = db()
    # Факт: усі завершені години (минулі дні + сьогодні до цієї години)
    done = df[df.index + pd.Timedelta(hours=1) <= pd.Timestamp(now)]
    con.executemany("INSERT OR REPLACE INTO actuals VALUES (?,?,?,?,?)",
                    [(h.isoformat(), None if pd.isna(r.pv_w) else float(r.pv_w), int(bool(r.clipped)), int(bool(r.enough)), run_at)
                     for h, r in done.iterrows()])
    # Прогноз — на години сьогодні, що ще попереду (і поточну)
    p = models.predict(bundle, df, today)
    p = p[p.index + pd.Timedelta(hours=1) > pd.Timestamp(now)]
    con.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?,?)",
                    [(run_at, r.model, h.isoformat(), float(r.raw_w), float(r.factor), float(r.pred_w)) for h, r in p.iterrows()])
    con.commit()
    con.close()
    left = p.groupby("model").pred_w.sum() / 1000
    log(f"прогноз на решту дня, кВт·год: " + ", ".join(f"{m} {v:.1f}" for m, v in left.items()) if len(p) else "сонце вже сіло")


def main():
    LIVE.mkdir(exist_ok=True)
    if not (LIVE / "station.json").exists():
        shutil.copy(HERE / "models" / "station.json", LIVE / "station.json")
    bundle = models.load(HERE / "models")
    log(f"моделі навчені до {bundle['trained_until']}")
    while True:
        try:
            once(bundle)
        except Exception:
            log("помилка:\n" + traceback.format_exc())
        # наступний запуск — на 10-й хвилині години (свіжий прогноз Open-Meteo оновлюється щогодини)
        now = datetime.now()
        nxt = (now + timedelta(hours=1)).replace(minute=10, second=0, microsecond=0)
        time.sleep(max(60, (nxt - now).total_seconds()))


if __name__ == "__main__":
    main()
