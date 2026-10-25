"""Погодинна таблиця для моделі сонця: ціль з Deye + ознаки з погоди і положення сонця.

Ті самі ознаки рахує й прогноз на Pi (features), тому вони тут, а не в ноутбуці.
"""
import numpy as np
import pandas as pd
import pvlib

import data_io as io

# Ознаки погоди, які беремо з кожного набору прогнозу
WEATHER = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
           "precipitation", "snowfall", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
# Для них дивимось, як прогноз мінявся day2 → day1 → свіжий
DYNAMICS = ["shortwave_radiation", "cloud_cover"]
MIN_FRAMES = 6  # справжніх кадрів за годину з 12: менше половини — година ненадійна
FULL_SOC = 93   # з 93 % інвертор уже притискає заряд: сонця бере менше, ніж могло б дати
FULL_SHARE = 0.5  # година обрізана, якщо батарея майже повна і мережа не потрібна хоча б пів години
GAP_FILL = 6    # кроків по 5 хв: дірку до 30 хв заповнюємо лінійно


def target():
    """Ціль по годинах: середня генерація, Вт. Плюс прапорці, чи годині можна вірити."""
    d = io.deye_frames()
    d = d[d.index >= io.PANELS_SINCE]
    real = d.pv_w.resample("1h").count()  # скільки справжніх записів у годині
    full = ((d.soc >= FULL_SOC) & (d.grid_w < 30)).resample("1h").mean()  # частка часу: сонце могло бути притиснуте
    # Сітка 5 хв: дірку до 30 хв заповнюємо середнім між сусідами (лінійно), довшу лишаємо порожньою
    g = d.resample("5min").mean().interpolate(method="time", limit=GAP_FILL, limit_area="inside")
    h = g.resample("1h").agg({"pv_w": "mean", "soc": "max", "grid_w": "mean", "load_w": "mean"})
    h.columns = ["pv_w", "soc_max", "grid_w", "load_w"]
    h["frames"] = real.reindex(h.index).fillna(0).astype(int)
    h["full_share"] = full.reindex(h.index)
    # Обрізання: батарея майже повна і з мережі не беремо — інвертор бере сонця рівно на дім, а могло бути більше.
    # Перевірено в 07: з ≥ 50 % таких записів факт 0.58–0.91 від очікуваного, з меншою часткою — як звичайно.
    h["clipped"] = h.full_share >= FULL_SHARE
    h["enough"] = h.frames >= MIN_FRAMES
    return h


def sun(index):
    """Положення сонця в середині години і радіація ясного неба: те, що робить пору року фізично."""
    st = io.station()
    loc = pvlib.location.Location(st["locationLat"], st["locationLng"], altitude=357)
    mid = index + pd.Timedelta(minutes=30)
    pos = loc.get_solarposition(mid)
    clear = loc.get_clearsky(mid, model="ineichen")
    out = pd.DataFrame({
        "sun_elevation": pos["apparent_elevation"].to_numpy(),
        "sun_azimuth": pos["azimuth"].to_numpy(),
        "clearsky_ghi": clear["ghi"].to_numpy(),
    }, index=index)
    doy = index.tz_convert(io.TZ).day_of_year
    # День року по колу: 31 грудня поруч з 1 січня
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def weather_features(sets=("forecast", "day1", "day2"), archive=True):
    """Ознаки з усіх наборів, з префіксом набору. archive — факт, лише для порівняння (стеля)."""
    parts = {k: io.weather(k)[[c for c in WEATHER if c in io.weather(k).columns]] for k in sets}
    if archive:
        parts["archive"] = io.weather("archive")[WEATHER + ["snow_depth"]]
    out = pd.concat({k: v for k, v in parts.items()}, axis=1)
    out.columns = [f"{k}__{c}" for k, c in out.columns]
    # Динаміка прогнозу: як змінювалась обіцянка і наскільки версії розходяться
    for c in DYNAMICS:
        if all(f"{k}__{c}" in out for k in ("forecast", "day1", "day2")):
            out[f"dyn__{c}__d1_minus_d2"] = out[f"day1__{c}"] - out[f"day2__{c}"]
            out[f"dyn__{c}__fc_minus_d1"] = out[f"forecast__{c}"] - out[f"day1__{c}"]
            out[f"dyn__{c}__spread"] = out[[f"{k}__{c}" for k in ("forecast", "day1", "day2")]].std(axis=1)
    # Сніг на панелях: скільки нападало за останні 3 доби. З прогнозу, не з ERA5: ERA5 приходить із затримкою
    # ~5 днів, і на Pi його ще нема — навчання і Pi мають бачити однакове
    out["snow_3d_cm"] = io.weather("forecast")["snowfall"].rolling(72, min_periods=1).sum()
    return out


def clear_sky_index(df, sets=("archive", "forecast", "day1", "day2")):
    """Індекс ясності (clear-sky index): прогноз радіації / радіація ясного неба. 1 — ясно, 0.1 — суцільні хмари.

    Стандарт у сонячних прогнозах: pvlib.irradiance.clearsky_index; Lauret et al., Solar 2022
    (https://hal.univ-reunion.fr/hal-03964785) — нормування на ясне небо дає кращі прогнози.

    Розплутує хмари і висоту сонця: радіація ясного неба на 0.99 корелює з висотою сонця,
    тож окремою ознакою вона не потрібна — лише як знаменник тут.
    """
    low = df.clearsky_ghi < 20  # на світанку ділення на майже нуль дає шум
    for k in sets:
        col = f"{k}__shortwave_radiation"
        if col in df:
            # Стандартна функція pvlib; стеля 1.5, а не 2.0 за замовчуванням: дані погодинні,
            # короткі сплески «хмара відбила сонце» в середньому за годину згладжуються
            csi = pvlib.irradiance.clearsky_index(df[col], df.clearsky_ghi, max_clearsky_index=1.5)
            df[f"csi__{k}"] = csi.mask(low)
    return df


def diffuse_fraction(df, sets=("archive", "forecast", "day1", "day2")):
    """Частка розсіяної радіації: розсіяна / загальна. ~1 — димка чи хмари (світло з усього неба), ~0.2 — пряме сонце.

    Розплутує загальну і пряму радіацію (між собою 0.94): загальна = пряма + розсіяна,
    тож лишаємо загальну + частку. Теж стандарт (diffuse fraction у моделях розкладу радіації, pvlib.irradiance.erbs).
    """
    for k in sets:
        total, diffuse = f"{k}__shortwave_radiation", f"{k}__diffuse_radiation"
        if total in df and diffuse in df:
            df[f"dfrac__{k}"] = (df[diffuse] / df[total]).clip(0, 1).mask(df[total] < 20)
    return df


def build(index=None, archive=True):
    """Повна таблиця: лише денні години з панелями. Ніч не потрібна: там сонце = 0 за визначенням.

    index — години, на які потрібна таблиця (на Pi: останні дні + сьогодні, де факту ще нема);
    archive=False — без ERA5 (на Pi його ще нема).
    """
    t = target()
    if index is not None:  # одиниці часу мають збігатись, інакше join дає не-часовий індекс
        t = t.reindex(pd.DatetimeIndex(index).as_unit(t.index.unit))
    s = sun(t.index)
    w = weather_features(archive=archive)
    df = t.join(s).join(w, how="inner")
    df = df[df.sun_elevation > 0]
    df = diffuse_fraction(clear_sky_index(df))
    # Збій: за фактом погоди ясно, снігу нема, а сонця < 25 % від звичайного — година ненадійна
    if "csi__archive" not in df:
        df["odd"] = False
        return df
    odd = (df.csi__archive >= 0.7) & (df.archive__snow_depth.fillna(0) < 0.001) & \
          (df.archive__shortwave_radiation > 400) & (df.pv_w < 0.25 * 5 * df.archive__shortwave_radiation) & ~df.clipped
    df["odd"] = odd
    df.loc[odd, "enough"] = False
    return df


# Не ознаки моделі: ціль, службові прапорці, знаменник для індексу ясності
NOT_FEATURES = {"pv_w", "frames", "soc_max", "grid_w", "load_w", "clipped", "enough", "clearsky_ghi", "full_share", "odd"}

# Набори ознак. Таблиця містить усе; модель бере один з наборів.
SUN = ["sun_elevation", "sun_azimuth", "snow_3d_cm"]
# Під перевірку «з ними і без них»: корелюють з висотою сонця / сезоном, але можуть нести своє
# (день року — сніг, листя, пил; температура — гарячі панелі дають менше)
TRY = ["doy_sin", "doy_cos", "{base}__temperature_2m"]
WEATHER_ONCE = ["relative_humidity_2m", "wind_speed_10m", "precipitation", "snowfall"]


def features(horizon="today", try_extra=True):
    """Ознаки моделі.

    today    — свіжий прогноз + як він мінявся (day2 → day1 → свіжий) + розкид версій.
               Свіжий прогноз є лише на найближчі години: для «скільки сонця до вечора».
    tomorrow — прогноз за добу + зміна day2 → day1. Свіжого на завтра ще нема:
               для «чи зарядить сонце батарею завтра», коли рахуємо ввечері.
    Сирі day1/day2 не беремо: вони точно відновлюються зі свіжого + змін (дублікати 0.9+).
    """
    base = {"today": "forecast", "tomorrow": "day1"}[horizon]
    f = SUN + [f"{base}__shortwave_radiation", f"csi__{base}", f"dfrac__{base}", f"{base}__cloud_cover"]
    f += [f"{base}__{v}" for v in WEATHER_ONCE]
    if horizon == "today":
        f += [f"dyn__{v}__{k}" for v in DYNAMICS for k in ("d1_minus_d2", "fc_minus_d1", "spread")]
    else:
        f += [f"dyn__{v}__d1_minus_d2" for v in DYNAMICS]
    if try_extra:
        f += [x.format(base=base) for x in TRY]
    return f


if __name__ == "__main__":
    df = build()
    df.to_parquet(io.DATA / "dataset.parquet")
    use = df.enough & ~df.clipped
    print(f"{len(df):,} денних годин, для навчання {use.sum():,} "
          f"(відкинуто: мало кадрів {(~df.enough).sum():,}, обрізано {(df.enough & df.clipped).sum():,})")
