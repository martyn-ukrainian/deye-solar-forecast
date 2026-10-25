"""Людські назви ознак для ноутбуків: «прогноз за добу: радіація» замість day1__shortwave_radiation."""

SETS = {"archive": "факт (ERA5)", "forecast": "свіжий прогноз", "day1": "прогноз за добу", "day2": "прогноз за 2 доби"}
VARS = {
    "shortwave_radiation": "радіація", "direct_radiation": "пряма радіація", "diffuse_radiation": "розсіяна радіація",
    "cloud_cover": "хмарність", "precipitation": "опади", "snowfall": "снігопад", "snow_depth": "сніг лежить",
    "temperature_2m": "температура", "relative_humidity_2m": "вологість", "wind_speed_10m": "вітер",
}
OWN = {
    "sun_elevation": "висота сонця", "sun_azimuth": "напрямок сонця",
    "doy_sin": "день року (sin)", "doy_cos": "день року (cos)", "snow_3d_cm": "снігопад за 3 доби",
}
DYN = {"d1_minus_d2": "day1 − day2", "fc_minus_d1": "свіжий − day1", "spread": "розкид трьох версій"}
DERIVED = {"csi": "індекс ясності", "dfrac": "частка розсіяної"}
OWN_GROUP = "сонце і дата (рахуємо самі)"
DYN_GROUP = "зміна прогнозу"
GROUP_ORDER = [OWN_GROUP, *SETS.values(), DYN_GROUP]


def label(col):
    """Людська назва ознаки."""
    if col in OWN:
        return OWN[col]
    if col == "clearsky_ghi":
        return "радіація ясного неба"
    head, *rest = col.split("__")
    if head in DERIVED:
        return f"{SETS[rest[0]]}: {DERIVED[head]}"
    if head == "dyn":
        var, kind = rest
        return f"{DYN_GROUP}: {VARS[var]}, {DYN[kind]}"
    return f"{SETS[head]}: {VARS[rest[0]]}"


def group(col):
    """Група ознаки: сонце і дата / набір погоди / зміна прогнозу."""
    if col in OWN:
        return OWN_GROUP
    head, *rest = col.split("__")
    if head == "dyn":
        return DYN_GROUP
    return SETS[rest[0] if head in DERIVED else head]
