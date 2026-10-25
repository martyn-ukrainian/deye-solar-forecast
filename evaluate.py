"""Спільне для 04_baseline і 05_model: поділ за часом, метрики, прості методи, варіанти ознак."""
import numpy as np
import pandas as pd
import pvlib

import data_io as io
import dataset

# Вчимо на першому році з панелями, перевіряємо на місяцях, яких модель не бачила
SPLIT = "2025-12-23"


def load():
    df = pd.read_parquet(io.DATA / "dataset.parquet")
    use = df.enough & ~df.clipped
    train = df[use & (df.index < SPLIT)]
    test = df[use & (df.index >= SPLIT)]
    return df, train, test


def metrics(y, pred):
    """MAE — середній промах, Вт; nMAE — він же у % від середньої генерації; bias — систематичний зсув (+ завищує)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    err = pred - y
    return {
        "MAE, Вт": round(float(np.abs(err).mean())),
        "nMAE, %": round(float(np.abs(err).mean() / y.mean() * 100), 1),
        "RMSE, Вт": round(float(np.sqrt((err ** 2).mean()))),
        "зсув, Вт": round(float(err.mean())),
    }


def by_month(index, y, pred):
    """MAE по місяцях: чи виграш стабільний, чи лише в один сезон."""
    loc = index.tz_convert(io.TZ)
    e = pd.Series(np.abs(np.asarray(pred) - np.asarray(y)), index=index)
    m = e.groupby([loc.year, loc.month]).mean().round(0)
    months = ["січ", "лют", "бер", "кві", "тра", "чер", "лип", "сер", "вер", "жов", "лис", "гру"]
    m.index = [f"{months[mo - 1]} {yr}" for yr, mo in m.index]
    return m


def daily_bootstrap(index, y, pred_a, pred_b, n=2000, seed=0):
    """Частка випадкових вибірок днів, де A точніше за B. ~50 % — різниці нема, ≥ 95 % — A справді краще."""
    loc = index.tz_convert(io.TZ).date
    ea = pd.Series(np.abs(np.asarray(pred_a) - np.asarray(y)), index=loc).groupby(level=0).sum()
    eb = pd.Series(np.abs(np.asarray(pred_b) - np.asarray(y)), index=loc).groupby(level=0).sum()
    rng = np.random.default_rng(seed)
    days = len(ea)
    idx = rng.integers(0, days, size=(n, days))
    return float((ea.to_numpy()[idx].sum(1) < eb.to_numpy()[idx].sum(1)).mean())


# ---------- прості методи (без ML) ----------

def persistence(df, index):
    """«Як учора»: генерація в цю ж годину добою раніше (з усієї таблиці, навіть обрізана — так було б і насправді)."""
    return df.pv_w.reindex(index - pd.Timedelta(days=1)).to_numpy()


def fit_scale(x, y):
    """Коефіцієнт k для pv ≈ k·x методом найменших квадратів через нуль."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float((x[ok] * y[ok]).sum() / (x[ok] ** 2).sum())


def poa(df, kind, tilt, azimuth):
    """Радіація на площину панелей (pvlib, ізотропна модель неба) з прогнозу набору kind."""
    st = io.station()
    loc = pvlib.location.Location(st["locationLat"], st["locationLng"], altitude=357)
    pos = loc.get_solarposition(df.index + pd.Timedelta(minutes=30))
    zen = pos["apparent_zenith"].to_numpy()
    ghi = df[f"{kind}__shortwave_radiation"].to_numpy()
    dhi = df[f"{kind}__diffuse_radiation"].to_numpy()
    # Open-Meteo дає пряму на горизонталь; на площину, перпендикулярну променям, — ділимо на cos зеніту
    dni = np.where(zen < 87, df[f"{kind}__direct_radiation"].to_numpy() / np.cos(np.radians(zen)), 0)
    irr = pvlib.irradiance.get_total_irradiance(tilt, azimuth, zen, pos["azimuth"].to_numpy(), dni, ghi, dhi, model="isotropic")
    return np.asarray(irr["poa_global"], float)


def fit_orientation(train, kind="forecast"):
    """Підбір нахилу і напрямку панелей на першому році: де «радіація на площину × k» найточніша."""
    best = None
    for tilt in range(0, 65, 5):
        for az in range(90, 285, 15):
            x = poa(train, kind, tilt, az)
            k = fit_scale(x, train.pv_w)
            mae = np.nanmean(np.abs(k * x - train.pv_w.to_numpy()))
            if best is None or mae < best[0]:
                best = (mae, tilt, az, k)
    return {"tilt": best[1], "azimuth": best[2], "k": best[3]}


# ---------- варіанти ознак для моделі ----------

def variants():
    """Набори ознак для порівняння «з ними і без них». Ключ — назва для таблиці."""
    full = dataset.features("today")
    doy = ["doy_sin", "doy_cos"]
    temp = ["forecast__temperature_2m"]
    dyn = [c for c in full if c.startswith("dyn__")]
    idx = ["csi__forecast", "dfrac__forecast"]
    # До розплутування: сирі три версії прогнозу, пряма/розсіяна, радіація ясного неба замість індексів
    raw = (dataset.SUN + doy + ["clearsky_ghi"]
           + [f"{k}__{v}" for k in ("forecast", "day1", "day2")
              for v in ("shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
                        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "snowfall")])
    return {
        "повний (20 ознак)": full,
        "без дня року": [c for c in full if c not in doy],
        "без температури": [c for c in full if c not in temp],
        "без дня року і температури": [c for c in full if c not in doy + temp],
        "без динаміки прогнозу": [c for c in full if c not in dyn],
        "без індексів (ясності, розсіяної)": [c for c in full if c not in idx],
        "сирі ознаки, до розплутування": raw,
    }


# ---------- як працюватиме насправді: перенавчання щомісяця + поправка на останні дні ----------

def recent_factor(pred, y, days=7):
    """Відношення факт / прогноз за останні `days` днів (лише минуле). Ловить бруд, тінь, зношення панелей."""
    day = pred.index.tz_convert(io.TZ).normalize()
    a = y.groupby(day).sum().rolling(f"{days}D", closed="left").sum()
    p = pred.groupby(day).sum().rolling(f"{days}D", closed="left").sum()
    fac = (a / p).clip(0.5, 1.5)
    return pd.Series(fac.reindex(day).fillna(1).to_numpy(), index=pred.index)


def walk_forward(df, fit_predict, features, start=SPLIT, correct_days=7):
    """Для кожного місяця тесту: вчимо на всіх чистих годинах до його початку, прогнозуємо місяць.

    fit_predict(train_X, train_y, test_X) -> прогноз. Повертає (без поправки, з поправкою) на чистих годинах тесту.
    """
    clean = df[df.enough & ~df.clipped]
    start_ts = pd.Timestamp(start, tz="UTC")
    end = (clean.index.max() + pd.offsets.MonthBegin()).tz_convert("UTC")
    months = pd.DatetimeIndex([start_ts]).append(pd.date_range(start_ts + pd.offsets.MonthBegin(), end, freq="MS"))
    pred = pd.Series(np.nan, index=clean.index)
    first = clean.index < pd.Timestamp(start, tz="UTC")
    # Для поправки потрібен прогноз і на навчальному періоді: беремо модель, навчену до старту тесту
    pred[first] = fit_predict(clean[first][features], clean[first].pv_w, clean[first][features])
    for a, b in zip(months[:-1], months[1:]):
        tr = clean[clean.index < a]
        mask = (clean.index >= a) & (clean.index < b)
        if mask.any():
            pred[mask] = fit_predict(tr[features], tr.pv_w, clean[mask][features])
    pred = pred.clip(lower=0)
    corrected = pred * recent_factor(pred, clean.pv_w, correct_days)
    test = clean.index >= pd.Timestamp(start, tz="UTC")
    return pred[test], corrected[test]


# ---------- нейромережа (PyTorch) для порівняння ----------

def mlp_fit_predict(X, y, Xt, epochs=300, seed=0):
    """Невелика повнозв'язна мережа. Пропуски (світанок) → медіана + прапорець; ознаки нормуємо — мережі це потрібно, деревам ні."""
    import torch
    torch.set_num_threads(1)  # інакше на Mac б’ється з потоками LightGBM (libomp) і висне
    torch.manual_seed(seed)
    X, Xt = X.copy(), Xt.copy()
    for c in X.columns:
        if X[c].isna().any() or Xt[c].isna().any():
            med = X[c].median()
            X[c + "__na"], Xt[c + "__na"] = X[c].isna().astype(float), Xt[c].isna().astype(float)
            X[c], Xt[c] = X[c].fillna(med), Xt[c].fillna(med)
    mu, sd = X.mean(), X.std().replace(0, 1)
    tx = torch.tensor(((X - mu) / sd).to_numpy(), dtype=torch.float32)
    tt = torch.tensor(((Xt - mu) / sd).to_numpy(), dtype=torch.float32)
    scale = float(y.std())
    ty = torch.tensor(y.to_numpy() / scale, dtype=torch.float32).unsqueeze(1)
    net = torch.nn.Sequential(
        torch.nn.Linear(tx.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
    loss_fn = torch.nn.L1Loss()  # вчимо на тій самій мірі, що й оцінюємо (середній промах)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(tx)).split(256):
            opt.zero_grad()
            loss_fn(net(tx[idx]), ty[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(tt).squeeze(1).numpy() * scale


# ---------- варіанти динамічної поправки ----------

def _daily(pred, y, mask=None):
    day = pred.index.tz_convert(io.TZ).normalize()
    if mask is not None:
        pred, y, day = pred[mask], y[mask], day[mask]
    return y.groupby(day).sum(), pred.groupby(day).sum()


def group_factor(pred, y, groups, days=7, min_share=0.2):
    """Окрема поправка для кожної групи годин (напр. сонячні / хмарні). Якщо в групі за тиждень мало енергії
    (< min_share від усієї) — беремо спільну поправку: з 1–2 годин відношення ненадійне."""
    common = recent_factor(pred, y, days)
    out = common.copy()
    day = pred.index.tz_convert(io.TZ).normalize()
    a_all, p_all = _daily(pred, y)
    p_all = p_all.rolling(f"{days}D", closed="left").sum()
    for g in pd.unique(groups):
        m = groups == g
        a, p = _daily(pred, y, m)
        a = a.reindex(p_all.index, fill_value=0).rolling(f"{days}D", closed="left").sum()
        p = p.reindex(p_all.index, fill_value=0).rolling(f"{days}D", closed="left").sum()
        fac = (a / p).clip(0.5, 1.5).where(p >= min_share * p_all)
        f = pd.Series(fac.reindex(day).to_numpy(), index=pred.index)
        out[m] = f[m].fillna(common[m])
    return out


def weighted_factor(pred, y, half_life=2.0, days=14):
    """Поправка, де свіжі дні важать більше: вага дня = 0.5^(днів тому / half_life)."""
    a, p = _daily(pred, y)
    full = pd.date_range(a.index.min(), a.index.max(), freq="D")
    a, p = a.reindex(full, fill_value=0), p.reindex(full, fill_value=0)
    w = 0.5 ** (np.arange(1, days + 1) / half_life)
    av = np.array([np.nan] * len(full)); pv = av.copy()
    for i in range(len(full)):
        lo = max(0, i - days)
        if i - lo < 1:
            continue
        ww = w[: i - lo][::-1]
        av[i] = (a.to_numpy()[lo:i] * ww).sum()
        pv[i] = (p.to_numpy()[lo:i] * ww).sum()
    fac = pd.Series(np.clip(av / pv, 0.5, 1.5), index=full)
    day = pred.index.tz_convert(io.TZ).normalize()
    return pd.Series(fac.reindex(day).fillna(1).to_numpy(), index=pred.index)


# ---------- модель поправки ----------

CORR_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
CORR_CONTEXT = ["csi__forecast", "dfrac__forecast", "sun_elevation", "sun_azimuth", "snow_3d_cm", "forecast__temperature_2m"]


def correction_frame(df, pred):
    """Таблиця з ознаками поправки: база і поправки з минулих днів (лише те, що відомо зранку)."""
    y = df.loc[pred.index, "pv_w"]
    d = df.copy()
    fw = weighted_factor(pred, y, 2)
    cols = {"base": pred, "f1": recent_factor(pred, y, 1), "f7": recent_factor(pred, y, 7), "fw2": fw, "base_fw2": pred * fw}
    for k, v in cols.items():
        d[k] = v.reindex(d.index)
    return d


def correction_model(df, pred, kind="multiplier", honest=False):
    """Модель, що вчиться поправляти базу. walk-forward, щомісяця заново.

    kind="on_top": прогнозує генерацію, знаючи базу і поправки.
    kind="multiplier": прогнозує множник факт/база (вага години = база).
    honest=True: вчиться лише на годинах, де база справді не бачила відповіді (з SPLIT);
    до того — динамічна формула.
    """
    import lightgbm as lgb
    d = correction_frame(df, pred)
    split = pd.Timestamp(SPLIT, tz="UTC")
    feats = (["base", "base_fw2"] if kind == "on_top" else []) + ["f1", "f7", "fw2"] + CORR_CONTEXT

    def fp(X, y, Xt):
        fallback = d.loc[Xt.index, "base_fw2"].to_numpy()
        if honest:
            keep = X.index >= split
            X, y = X[keep], y[keep]
            if len(X) < 300:
                return fallback
        if kind == "on_top":
            return lgb.LGBMRegressor(**CORR_PARAMS).fit(X, y).predict(Xt)
        b, bt = d.loc[X.index, "base"].to_numpy(), d.loc[Xt.index, "base"].to_numpy()
        m = b > 50
        k = lgb.LGBMRegressor(**CORR_PARAMS).fit(X[m], (y.to_numpy()[m] / b[m]).clip(0, 2), sample_weight=b[m]).predict(Xt)
        return bt * np.clip(k, 0.3, 1.7)

    out, _ = walk_forward(d, fp, feats)
    return out


def physics_fit_predict(X, y, Xt):
    """«Фізика панелей» з 04: нахил і напрямок підбираємо на навчанні, далі k × радіація на площині панелей."""
    o = fit_orientation(X.assign(pv_w=y))
    return o["k"] * poa(Xt, "forecast", o["tilt"], o["azimuth"])


PHYSICS_COLS = ["forecast__shortwave_radiation", "forecast__direct_radiation", "forecast__diffuse_radiation"]


def raw_all(df, fit_predict, features):
    """Прогноз бази без поправки на всіх чистих годинах: перший рік — моделлю першого року, далі щомісяця (як walk_forward)."""
    clean = df[df.enough & ~df.clipped]
    first = clean.index < pd.Timestamp(SPLIT, tz="UTC")
    raw, _ = walk_forward(df, fit_predict, features)
    pred = pd.Series(np.nan, index=clean.index)
    pred[first] = fit_predict(clean[first][features], clean[first].pv_w, clean[first][features])
    pred[~first] = np.asarray(raw)
    return pred.clip(lower=0)


def daily_model_factor(pred, y, X, days=7, half_life=None, ridge=0.3, coefs=False):
    """Щоденна модель поправки: кожен день заново вчиться лише на останніх `days` днях.

    Модель: факт ≈ база × (a + b1·x1 + b2·x2 …), x — умови години (стовпці X, центровані).
    Тобто поправка не одне число, а залежить від умов години. Вчиться зваженими найменшими квадратами;
    ridge тягне b до 0 (на ~70 годинах інакше скаче). half_life — свіжі дні важать більше.
    Поправка обмежена 0.5–1.5; якщо даних мало — 1.
    """
    day = pred.index.tz_convert(io.TZ).normalize()
    Xc = (X - X.mean()).fillna(0).to_numpy()
    b, yv = pred.to_numpy(), y.to_numpy()
    out = np.ones(len(pred))
    cf = np.full((len(pred), 1 + Xc.shape[1]), np.nan)
    days_u = day.unique().sort_values()
    for d in days_u:
        win = (day >= d - pd.Timedelta(days=days)) & (day < d)
        win &= b > 50
        today = day == d
        if win.sum() < 20:
            continue
        age = ((d - day[win]).days).to_numpy()
        w = 0.5 ** (age / half_life) if half_life else np.ones(win.sum())
        # регресія y = a·база + Σ bi·база·xi
        A = np.column_stack([b[win], b[win, None] * Xc[win]])
        W = w[:, None]
        G = A.T @ (A * W)
        G[1:, 1:] += ridge * np.eye(Xc.shape[1]) * G[0, 0]
        coef = np.linalg.solve(G, A.T @ (yv[win] * w))
        out[today] = coef[0] + Xc[today] @ coef[1:]
        cf[today] = coef
    fac = pd.Series(np.clip(out, 0.5, 1.5), index=pred.index)
    if coefs:  # ще й коефіцієнти маленької моделі дня: рівень, вплив кожної умови
        return fac, pd.DataFrame(cf, index=pred.index, columns=["small_level"] + [f"small_{c}" for c in X.columns])
    return fac


def stacked_correction(df, pred, X_small):
    """Стекінг: велика модель на результатах щоденних маленьких моделей.

    Маленька модель дня (7 днів, півжиття 2) дає поправку години і свої коефіцієнти. Велика LightGBM вчиться
    множнику факт/база з них + умов години. Вчиться лише чесно (з SPLIT), щомісяця заново; поки даних мало — динамічна формула.
    """
    import lightgbm as lgb
    clean = df.loc[pred.index]
    y = clean.pv_w
    fac, cf = daily_model_factor(pred, y, X_small, days=7, half_life=2, ridge=1.0, coefs=True)
    d = correction_frame(df, pred)
    for c in cf.columns:
        d[c] = cf[c].reindex(d.index)
    d["small_factor"] = fac.reindex(d.index)
    feats = ["small_factor", *cf.columns, "fw2", "f1"] + CORR_CONTEXT
    split = pd.Timestamp(SPLIT, tz="UTC")

    def fp(X, yy, Xt):
        fallback = d.loc[Xt.index, "base_fw2"].to_numpy()
        keep = X.index >= split
        X, yy = X[keep], yy[keep]
        if len(X) < 300:
            return fallback
        b, bt = d.loc[X.index, "base"].to_numpy(), d.loc[Xt.index, "base"].to_numpy()
        m = b > 50
        k = lgb.LGBMRegressor(**CORR_PARAMS).fit(X[m], (yy.to_numpy()[m] / b[m]).clip(0, 2), sample_weight=b[m]).predict(Xt)
        return bt * np.clip(k, 0.3, 1.7)

    out, _ = walk_forward(d, fp, feats)
    return out


def sequence_correction(df, pred, days=7):
    """Варіант 3: одна велика модель бачить останні `days` днів окремо — поправку кожного дня (факт/база)
    і скільки тоді обіцяла база (сонячний чи хмарний був день). Вчиться множнику факт/база, лише чесно (з SPLIT)."""
    import lightgbm as lgb
    clean = df.loc[pred.index]
    day = pred.index.tz_convert(io.TZ).normalize()
    a, p = _daily(pred, clean.pv_w)
    full = pd.date_range(a.index.min(), a.index.max(), freq="D")
    a, p = a.reindex(full), p.reindex(full)
    ratio = (a / p).where(p > 500).clip(0.3, 2)
    d = correction_frame(df, pred)
    feats = []
    for k in range(1, days + 1):
        for name, s in ((f"day{k}_factor", ratio.shift(k)), (f"day{k}_base_kwh", p.shift(k) / 1000)):
            d[name] = pd.Series(s.reindex(day).to_numpy(), index=pred.index).reindex(d.index)
            feats.append(name)
    feats += CORR_CONTEXT
    split = pd.Timestamp(SPLIT, tz="UTC")

    def fp(X, yy, Xt):
        fallback = d.loc[Xt.index, "base_fw2"].to_numpy()
        keep = X.index >= split
        X, yy = X[keep], yy[keep]
        if len(X) < 300:
            return fallback
        b, bt = d.loc[X.index, "base"].to_numpy(), d.loc[Xt.index, "base"].to_numpy()
        m = b > 50
        k = lgb.LGBMRegressor(**CORR_PARAMS).fit(X[m], (yy.to_numpy()[m] / b[m]).clip(0, 2), sample_weight=b[m]).predict(Xt)
        return bt * np.clip(k, 0.3, 1.7)

    out, _ = walk_forward(d, fp, feats)
    return out


# ---------- поправка, навчена на 2 роках чесних прогнозів (out-of-fold) ----------

def oof_predict(df, fit_predict, features):
    """Прогноз бази на кожен місяць моделлю, навченою на всіх ІНШИХ місяцях (out-of-fold).
    Так кожна година має прогноз, який її не бачив, — на всіх 2 роках. Для навчання поправки, не для перевірки."""
    clean = df[df.enough & ~df.clipped]
    month = clean.index.tz_convert(io.TZ).to_period("M")
    pred = pd.Series(np.nan, index=clean.index)
    for m in month.unique():
        mask = month == m
        pred[mask] = fit_predict(clean[~mask][features], clean[~mask].pv_w, clean[mask][features])
    return pred.clip(lower=0)


def corr_features(df, pred, kind, X_small=None):
    """Ознаки моделі поправки з прогнозу бази `pred` (лише минулі дні + умови години).
    kind: "summary" — поправка вчора / 7 днів / динамічна; "sequence" — кожен з 7 днів окремо;
    "stacking" — результати щоденних маленьких моделей + summary."""
    clean = df.loc[pred.index]
    y = clean.pv_w
    F = pd.DataFrame(index=pred.index)
    F["base"] = pred
    if kind in ("summary", "stacking"):
        F["f1"], F["f7"], F["fw2"] = recent_factor(pred, y, 1), recent_factor(pred, y, 7), weighted_factor(pred, y, 2)
    if kind == "stacking":
        fac, cf = daily_model_factor(pred, y, X_small, days=7, half_life=2, ridge=1.0, coefs=True)
        F["small_factor"] = fac
        F[cf.columns] = cf
    if kind == "sequence":
        day = pred.index.tz_convert(io.TZ).normalize()
        a, p = _daily(pred, y)
        full = pd.date_range(a.index.min(), a.index.max(), freq="D")
        a, p = a.reindex(full), p.reindex(full)
        ratio = (a / p).where(p > 500).clip(0.3, 2)
        for k in range(1, 8):
            F[f"day{k}_factor"] = ratio.shift(k).reindex(day).to_numpy()
            F[f"day{k}_base_kwh"] = (p.shift(k) / 1000).reindex(day).to_numpy()
    F[CORR_CONTEXT] = clean[CORR_CONTEXT]
    return F


def corr_walk(df, live, train_pred, kind, X_small=None, since=None):
    """Модель поправки (множник факт/база), щомісяця заново.

    Вчиться на ознаках з `train_pred` (out-of-fold — 2 роки; або сама база — лише чесні дні з `since`),
    застосовується до справжньої бази `live` (як на Pi: база щомісяця на минулому). Повертає прогноз на тесті."""
    import lightgbm as lgb
    clean = df.loc[live.index]
    y = clean.pv_w
    Ftr, Flive = corr_features(df, train_pred, kind, X_small), corr_features(df, live, kind, X_small)
    feats = [c for c in Ftr.columns if c != "base"]
    start = pd.Timestamp(SPLIT, tz="UTC")
    end = (clean.index.max() + pd.offsets.MonthBegin()).tz_convert("UTC")
    months = pd.DatetimeIndex([start]).append(pd.date_range(start + pd.offsets.MonthBegin(), end, freq="MS"))
    out = live * weighted_factor(live, y, 2)  # запас, поки мало даних
    for a, b in zip(months[:-1], months[1:]):
        tr = (Ftr.index < a) & (Ftr.base > 50)
        if since is not None:
            tr &= Ftr.index >= pd.Timestamp(since, tz="UTC")
        mask = (Flive.index >= a) & (Flive.index < b)
        if tr.sum() < 300 or not mask.any():
            continue
        base_tr = Ftr.base[tr].to_numpy()
        k = lgb.LGBMRegressor(**CORR_PARAMS).fit(Ftr.loc[tr, feats], (y[tr].to_numpy() / base_tr).clip(0, 2),
                                                 sample_weight=base_tr).predict(Flive.loc[mask, feats])
        out[mask] = live[mask].to_numpy() * np.clip(k, 0.3, 1.7)
    return out[out.index >= start]


# ---------- додаткові ознаки з розбору помилок (06) ----------

EXTRA = {
    "snow_on_panels_cm": "сніг на панелях (модель: падає — накопичується, тане від тепла і сонця), см",
    "days_since_snow": "днів після снігопаду",
    "forecast__snow_depth": "прогноз: глибина снігу, м",
    "night_min_temp": "мінімум температури за 12 год до того, °C",
    "night_max_humidity": "максимум вологості за 12 год до того, %",
    "days_since_start": "днів від початку роботи панелей (зношення)",
}


def extra_features(df):
    """Ознаки, відомі наперед (з прогнозу погоди), — про сніг, іній і час. Рахуються по всіх годинах, не лише денних."""
    w = io.weather("forecast")
    snow, temp, rad = w.snowfall.to_numpy(), w.temperature_2m.to_numpy(), w.shortwave_radiation.to_numpy()
    acc, panels = 0.0, np.zeros(len(w))
    for i in range(len(w)):
        acc += snow[i]
        acc -= max(temp[i], 0) * 0.3          # тане від тепла: 0.3 см на градус-годину
        if temp[i] > -5:
            acc -= rad[i] / 1000 * 0.5         # і від сонця, навіть у легкий мороз (темне скло гріється)
        acc = min(max(acc, 0.0), 30.0)
        panels[i] = acc
    out = pd.DataFrame(index=w.index)
    out["snow_on_panels_cm"] = panels
    last = pd.Series(np.where(w.snowfall > 0.1, np.arange(len(w)), np.nan), index=w.index).ffill()
    out["days_since_snow"] = ((np.arange(len(w)) - last) / 24).clip(upper=30).fillna(30)
    out["forecast__snow_depth"] = w.snow_depth
    out["night_min_temp"] = w.temperature_2m.rolling(12, min_periods=1).min()
    out["night_max_humidity"] = w.relative_humidity_2m.rolling(12, min_periods=1).max()
    out["days_since_start"] = (w.index - pd.Timestamp(io.PANELS_SINCE, tz="UTC")).days
    return df.join(out)


# ---------- орієнтація, тінь, карта сонця (07) ----------

AZ_BINS = np.arange(45, 316, 15)   # напрямок сонця, ° (90 схід, 180 південь, 270 захід)
EL_BINS = np.arange(0, 71, 5)      # висота сонця, °


def clear_days(df, min_ratio=0.8, min_csi=0.6):
    """Ясні дні без снігу: весь день сонце за фактом погоди (ERA5)."""
    g = df.assign(day=df.index.tz_convert(io.TZ).normalize()).groupby("day")
    ok = (g.archive__shortwave_radiation.sum() / g.clearsky_ghi.sum() >= min_ratio) & \
         (g.archive__snow_depth.max().fillna(0) < 0.001) & (g.csi__archive.min() >= min_csi)
    return ok[ok].index


def aoi(df, tilt, azimuth):
    """Кут падіння променів на площину панелей, ° (0 — прямо в панель, 90 — по дотичній)."""
    st = io.station()
    loc = pvlib.location.Location(st["locationLat"], st["locationLng"], altitude=357)
    pos = loc.get_solarposition(df.index + pd.Timedelta(minutes=30))
    return np.asarray(pvlib.irradiance.aoi(tilt, azimuth, pos["apparent_zenith"], pos["azimuth"]), float)


def sun_map(rows, num, den, prior=10):
    """Карта «факт / прогноз» по положенню сонця (напрямок × висота). Мало годин у клітинці — тягнемо до 1."""
    az = pd.cut(rows.sun_azimuth, AZ_BINS); el = pd.cut(rows.sun_elevation, EL_BINS)
    g = pd.DataFrame({"n": num, "d": den, "az": az, "el": el}).groupby(["el", "az"], observed=False)
    s_n, s_d, cnt = g.n.sum(), g.d.sum(), g.n.size()
    mean_d = s_d.sum() / max(cnt.sum(), 1)
    return ((s_n + prior * mean_d) / (s_d + prior * mean_d)).unstack(), cnt.unstack()


def apply_map(rows, m):
    """Множник з карти для кожної години; що більше прямого сонця — то сильніше діє (тінь і кут важать для прямого)."""
    az = pd.cut(rows.sun_azimuth, AZ_BINS); el = pd.cut(rows.sun_elevation, EL_BINS)
    v = np.array([m.at[e, a] if isinstance(e, pd.Interval) and isinstance(a, pd.Interval) else 1.0 for e, a in zip(el, az)])
    direct = 1 - rows.dfrac__forecast.fillna(1).clip(0, 1).to_numpy()
    return 1 + (np.nan_to_num(v, nan=1.0) - 1) * direct


def map_walk(df, base):
    """Шлях Б: стара модель × карта сонця. Карта — з ясних днів до початку кожного місяця (чесно)."""
    clean = df.loc[base.index]
    days = clear_days(df)
    day = clean.index.tz_convert(io.TZ).normalize()
    on_clear = day.isin(days)
    start = pd.Timestamp(SPLIT, tz="UTC")
    end = (clean.index.max() + pd.offsets.MonthBegin()).tz_convert("UTC")
    months = pd.DatetimeIndex([start]).append(pd.date_range(start + pd.offsets.MonthBegin(), end, freq="MS"))
    fac = pd.Series(1.0, index=clean.index)
    first = clean.index < start
    m, _ = sun_map(clean[first & on_clear], clean.pv_w[first & on_clear], base[first & on_clear])
    fac[first] = apply_map(clean[first], m)
    for a, b in zip(months[:-1], months[1:]):
        tr = (clean.index < a) & on_clear
        mask = (clean.index >= a) & (clean.index < b)
        if mask.any():
            m, _ = sun_map(clean[tr], clean.pv_w[tr], base[tr])
            fac[mask] = apply_map(clean[mask], m)
    return base * fac


def orient_factor(index, tilt, azimuth):
    """Ефективність положення панелей: ясне небо на площині панелей (з втратами на відбиття від скла під кутом)
    / ясне небо на горизонталі. Лише дата і час — погода не потрібна. 1.0 — як горизонтальне скло."""
    st = io.station()
    loc = pvlib.location.Location(st["locationLat"], st["locationLng"], altitude=357)
    mid = index + pd.Timedelta(minutes=30)
    pos = loc.get_solarposition(mid)
    cs = loc.get_clearsky(mid, model="ineichen", solar_position=pos)
    irr = pvlib.irradiance.get_total_irradiance(tilt, azimuth, pos["apparent_zenith"], pos["azimuth"],
                                                cs["dni"], cs["ghi"], cs["dhi"], model="isotropic")
    theta = pvlib.irradiance.aoi(tilt, azimuth, pos["apparent_zenith"], pos["azimuth"])
    iam = pvlib.iam.physical(theta)  # скло: що гостріший кут, то більше відбивається
    poa = irr["poa_direct"] * iam + irr["poa_diffuse"]
    return pd.Series(np.where(cs["ghi"] > 20, poa / cs["ghi"], np.nan), index=index)


def solar_noon_side(index, buffer_h=1.0):
    """Ранок / вечір відносно сонячного полудня; ±buffer_h навколо полудня — «полудень» (кут з обох боків однаковий)."""
    st = io.station()
    days = index.tz_convert(io.TZ).normalize()
    u = days.unique()
    # полудень дня: опівдні за Києвом — той самий день і в UTC
    tr = pvlib.solarposition.sun_rise_set_transit_spa((u + pd.Timedelta(hours=12)).tz_convert("UTC"), st["locationLat"], st["locationLng"])["transit"]
    noon = pd.DatetimeIndex(pd.Series(pd.DatetimeIndex(tr).tz_convert("UTC"), index=u).reindex(days))
    dh = np.asarray((index + pd.Timedelta(minutes=30) - noon).total_seconds()) / 3600
    return np.where(dh < -buffer_h, "ранок", np.where(dh > buffer_h, "вечір", "полудень"))


def weighted_side_factor(pred, y, side, half_life=2.0):
    """Динамічна поправка окремо для ранку і вечора; полудень — спільна."""
    out = weighted_factor(pred, y, half_life)
    for s in ("ранок", "вечір"):
        m = side == s
        out[m] = weighted_factor(pred[m], y[m], half_life).to_numpy()
    return out
