"""Збирає 04_4_oof.ipynb: модель поправки на 2 роках чесних прогнозів (out-of-fold) — і чому це не працює."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 04_4 · Поправка на 2 роках чесних прогнозів

У `04_1`–`04_3` моделі поправки вчились лише на ~10 місяцях: на першому році база прогнозує дні, на яких сама вчилась, і там «помиляється» штучно мало. Ідея: зробити чесні прогнози бази на **обох роках**, щоб модель поправки бачила дві зими і два літа.

Два способи:
- **по місяцях (out-of-fold):** для кожного місяця база вчиться на всіх інших місяцях і прогнозує цей;
- **дві моделі:** модель другого року прогнозує перший рік, другий рік — справжня база (вчилась лише на минулому).

Модель поправки (ті самі три з `04_1`/`04_3`: підсумок 7 днів, кожен з 7 днів окремо, стекінг на маленьких моделях) вчиться на цих прогнозах, а застосовується до справжньої бази — як буде на Pi. Перевірка та сама: щомісяця заново, лише на минулому."""),
    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
import data_io as io, evaluate as ev
pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', None)
df, train, test = ev.load()
clean = df[df.enough & ~df.clipped]; y = clean.pv_w; t = test.index
S = pd.Timestamp(ev.SPLIT, tz='UTC'); year1 = clean.index < S
saved = pd.read_pickle(io.DATA / 'corrections.pkl')
PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
fp_lgb = lambda X, yy, Xt: lgb.LGBMRegressor(**PARAMS).fit(X, yy).predict(Xt)
X_small = clean[['csi__forecast']].fillna(0.5).assign(pm=(clean.sun_azimuth >= 180).astype(float))
BASES = {'фізика панелей': (saved['physics']['без поправки'], ev.physics_fit_predict, ev.PHYSICS_COLS),
         'LightGBM (20 ознак)': (saved['lgb']['без поправки'], fp_lgb, ev.variants()['повний (20 ознак)'])}

prep = {}
for name, (live, fp, feats) in BASES.items():
    live = live.reindex(clean.index)
    oof = ev.oof_predict(df, fp, feats)
    two = live.copy(); two[year1] = np.clip(fp(clean[~year1][feats], y[~year1], clean[year1][feats]), 0, None)
    prep[name] = (live, oof, two)"""),
    md("""## Що бачить модель поправки на кожному році

Факт / база за рік: 1 — база в середньому влучає, < 1 — завищує, > 1 — занижує."""),
    code("""r = lambda p, m: round(y[m].sum() / p[m].sum(), 2)
pd.DataFrame({name: {'1-й рік: база бачила ці дні': r(live, year1),
                     '1-й рік: по місяцях (out-of-fold)': r(oof, year1),
                     '1-й рік: модель 2-го року': r(two, year1),
                     '2-й рік: по місяцях (out-of-fold)': r(oof, ~year1),
                     '2-й рік: справжня база (як на Pi)': r(live, ~year1)}
              for name, (live, oof, two) in prep.items()})"""),
    md("## Результат"),
    code("""def compare(name):
    live, oof, two = prep[name]
    V = {'динамічна формула, півжиття 2 дні': live * ev.weighted_factor(live, y, 2)}
    for kind, label in (('summary', 'підсумок 7 днів'), ('sequence', 'кожен з 7 днів окремо'), ('stacking', 'стекінг')):
        V[f'{label}: 10 міс чесних'] = ev.corr_walk(df, live, live, kind, X_small, since=ev.SPLIT)
        V[f'{label}: 2 роки, по місяцях'] = ev.corr_walk(df, live, oof, kind, X_small)
        V[f'{label}: 2 роки, дві моделі'] = ev.corr_walk(df, live, two, kind, X_small)
    ref = V['динамічна формула, півжиття 2 дні'].reindex(t)
    rows = {k: {**ev.metrics(test.pv_w, v.reindex(t)),
                'краще за формулу, % вибірок': '—' if v is V['динамічна формула, півжиття 2 дні'] else round(ev.daily_bootstrap(t, test.pv_w, v.reindex(t), ref) * 100)}
            for k, v in V.items()}
    out = pd.DataFrame(rows).T.sort_values('MAE, Вт'); out.index.name = 'поправка'
    for c in ('MAE, Вт', 'RMSE, Вт', 'зсув, Вт'):
        out[c] = out[c].astype(int)
    out['nMAE, %'] = out['nMAE, %'].astype(float)
    return out
compare('фізика панелей')"""),
    code("compare('LightGBM (20 ознак)')"),
    md("""## Висновки

1. **На базі «фізика» 2 роки чесних прогнозів допомогли:** варіант 3 (кожен з 7 днів окремо) — **293 Вт** проти 332 у формули (і 313 на 10 місяцях), краще у 100 % вибірок. Це майже рівень LightGBM (300) — бо, як і в `04_3`, модель поправки доробляє фізику до ML. Але зсув +33…56 Вт: вона трохи завищує.
2. **На базі LightGBM — ні:** будь-яка модель поправки гірша за формулу (316–337 проти 300), зсув +66…93 Вт.
3. **Чому завищує.** У чесних прогнозах перший рік база трохи *занижує* (факт / база 1.01–1.07), другий — *завищує* (0.94–0.95). Модель поправки бачить умови години і вчиться з першого року «множник ≈ 1.05», а на другому треба 0.95. Свіжі дні їй підказують правду, але історія перетягує.
4. **Поправка — не закономірність, а дрейф.** Перша зима — поправка ~1, друга — 0.6 (сніг). З історії це не передбачити. Тому поверх LightGBM найкраще найпростіше — свіжі дні (динамічна формула).
5. **Чого справді бракує — свіжіших даних, а не довшої історії:** поправка всередині дня (з ранку видно, як ідуть сьогоднішні години), окремо два стрінги pv1/pv2 з Pi, сніг на панелях (камера чи датчик).
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "04_4_oof.ipynb")
