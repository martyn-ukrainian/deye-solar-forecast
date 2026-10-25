"""Навчає 4 моделі на всій історії і кладе їх у models/ (звідти — в образ для Pi).

python train.py   (спершу fetch_deye.py і fetch_weather.py — свіжі дані)
Перенавчати раз на кілька місяців: щомісячне перенавчання майже нічого не дає (06_1).
"""
import shutil
from pathlib import Path

import data_io as io
import dataset
import models

OUT = Path(__file__).parent / "models"

if __name__ == "__main__":
    df = models.add_features(dataset.build())
    bundle = models.train(df)
    models.save(bundle, OUT)
    shutil.copy(io.DATA / "station.json", OUT / "station.json")
    print(f"навчено на {bundle['hours']:,} чистих годинах до {bundle['trained_until']}; "
          f"фізика: нахил {bundle['orient']['tilt']:.0f}°, напрямок {bundle['orient']['azimuth']:.0f}°")
