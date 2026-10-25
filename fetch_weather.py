"""Погода Open-Meteo для станції, погодинно, UTC: data/weather/{archive,forecast,day1,day2}.parquet.

archive  — реаналіз ERA5: якою погода була насправді.
forecast — архів прогнозів, склеєний зі свіжих запусків (на кілька годин уперед): майже факт.
day1     — що прогноз обіцяв за добу наперед (Previous Runs API). Чесний варіант для «вистачить до ранку».
day2     — те саме за дві доби: для довгих зимових відключень.
Колонки в усіх трьох однакові (суфікс _previous_dayN знято), щоб моделі їх міняти місцями.

python fetch_weather.py [з YYYY-MM-DD] [по YYYY-MM-DD]  (за замовчуванням: від запуску станції до вчора)
"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).parent / "data"
OUT = DATA / "weather"
URLS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "day1": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "day2": "https://previous-runs-api.open-meteo.com/v1/forecast",
}
SUFFIX = {"day1": "_previous_day1", "day2": "_previous_day2"}
HOURLY = [
    # Сонце: усе, з чого рахується радіація на панелі
    "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance",
    "sunshine_duration", "is_day",
    # Хмари і те, що закриває панелі
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "precipitation", "snowfall", "snow_depth",
    # Нагрів панелей знижує ККД, вітер охолоджує
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
]

# Цих полів у Previous Runs немає або вони не залежать від прогнозу
DAY1_SKIP = {"is_day", "snow_depth", "sunshine_duration", "direct_normal_irradiance"}


def fetch(kind, start, end, station):
    r = requests.get(URLS[kind], params={
        "latitude": station["locationLat"],
        "longitude": station["locationLng"],
        "start_date": str(start),
        "end_date": str(end),
        "hourly": ",".join(v + SUFFIX.get(kind, "") for v in HOURLY if not (kind in SUFFIX and v in DAY1_SKIP)),
        "timezone": "GMT",
    }, timeout=120)
    r.raise_for_status()
    body = r.json()
    df = pd.DataFrame(body["hourly"]).rename(columns=lambda c: c.removesuffix(SUFFIX.get(kind, "")))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    print(f"{kind}: {len(df)} годин, {df['time'].min():%Y-%m-%d}..{df['time'].max():%Y-%m-%d}, "
          f"висота моделі {body.get('elevation')} м")
    return df.set_index("time")


def main():
    station = json.loads((DATA / "station.json").read_text())
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.fromtimestamp(station["startOperatingTime"])
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today() - timedelta(days=1)
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in URLS:
        df = fetch(kind, start, end, station)
        df.to_parquet(OUT / f"{kind}.parquet")
        empty = df.isna().mean()
        if (empty > 0).any():
            print("  порожні частки:", empty[empty > 0].round(3).to_dict())


if __name__ == "__main__":
    main()
