"""Історія станції з хмари Deye, крок 5 хв: data/deye/ДАТА.json. Вже скачані дні пропускає.

python fetch_deye.py [з YYYY-MM-DD] [по YYYY-MM-DD]  (за замовчуванням: від запуску станції до вчора)
"""
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
BASE = os.environ["DEYE_CLOUD_API_BASE_URL"]
STATION = int(os.environ["DEYE_CLOUD_STATION_ID"])
OUT = Path(__file__).parent / "data" / "deye"
PAUSE_S = 1.5  # ліміти API не документовані — не поспішаємо


def token():
    r = requests.post(
        f"{BASE}/v1.0/account/token",
        params={"appId": os.environ["DEYE_CLOUD_APP_ID"]},
        json={
            "appSecret": os.environ["DEYE_CLOUD_APP_SECRET"],
            "email": os.environ["DEYE_CLOUD_EMAIL"],
            "password": hashlib.sha256(os.environ["DEYE_CLOUD_PASSWORD"].encode()).hexdigest(),
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["accessToken"]


def api(tok, path, body):
    r = requests.post(f"{BASE}{path}", json=body, headers={"Authorization": f"Bearer {tok}"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("code") not in ("0", "1000000", 0) and not data.get("success"):
        raise RuntimeError(data.get("msg"))
    return data


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    station = next(s for s in api(tok, "/v1.0/station/list", {"page": 1, "size": 20})["stationList"] if s["id"] == STATION)
    (OUT.parent / "station.json").write_text(json.dumps(station, ensure_ascii=False, indent=1))
    first = date.fromtimestamp(station["startOperatingTime"])
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else first
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else date.today() - timedelta(days=1)
    day, done, fails = start, 0, 0
    while day <= end:
        path = OUT / f"{day}.json"
        if not path.exists():
            try:
                items = api(tok, "/v1.0/station/history",
                            {"stationId": STATION, "startAt": str(day), "endAt": str(day), "granularity": 1})
                path.write_text(json.dumps(items.get("stationDataItems", [])))
                done += 1
                if done % 30 == 0:
                    print(f"{datetime.now():%H:%M} {day}: скачано {done}", flush=True)
                time.sleep(PAUSE_S)
            except Exception as e:
                fails += 1
                print(f"{day}: {e}", flush=True)
                if fails > 20:
                    sys.exit("забагато помилок поспіль")
                time.sleep(10)
                tok = token()
                continue
        fails = 0
        day += timedelta(days=1)
    print(f"готово: {start}..{end}, нових днів {done}")


if __name__ == "__main__":
    main()
