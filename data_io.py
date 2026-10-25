"""Спільне завантаження даних для ноутбуків і скриптів."""
import json
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
TZ = "Europe/Kyiv"
WEATHER_SETS = ("archive", "forecast", "day1", "day2")
PANELS_SINCE = "2024-12-23"  # до цього інвертор із батареєю працював без панелей


def station():
    return json.loads((DATA / "station.json").read_text())


def deye_frames():
    """Усі 5-хвилинні кадри хмари Deye, індекс — UTC. battery_w: + віддає в дім, − заряджається."""
    rows = []
    for f in sorted((DATA / "deye").glob("*.json")):
        rows += json.loads(f.read_text())
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timeStamp"], unit="s", utc=True)
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated()]
    return df.rename(columns={
        "generationPower": "pv_w",
        "consumptionPower": "load_w",
        "purchasePower": "grid_w",
        "batteryPower": "battery_w",
        "batterySOC": "soc",
    })[["pv_w", "grid_w", "battery_w", "load_w", "soc"]].assign(  # джерела → споживач → заряд
        # Хмара не пише мережу, коли з неї нічого не беремо: баланс дім − сонце − батарея тоді ≈ 0
        grid_w=lambda x: x["grid_w"].fillna(0),
    )


def weather(kind):
    """Погода Open-Meteo. Мітка в Open-Meteo — кінець години; зсуваємо на початок, як у Deye."""
    df = pd.read_parquet(DATA / "weather" / f"{kind}.parquet")
    df.index = df.index - pd.Timedelta(hours=1)
    return df
