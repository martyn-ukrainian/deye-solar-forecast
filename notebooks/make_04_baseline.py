"""Збирає 04_baseline.ipynb: прості методи без ML — планка, яку модель має побити."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 04 · Планка: прості методи без ML

Що дає найпростіший прогноз сонця на «сьогодні». Модель у `05` має це побити, інакше вона не потрібна.

**Перевірка чесна, як буде на Pi:** перший прогноз — на моделі, навченій на першому році (23 гру 2024 – 22 гру 2025); далі щомісяця перенавчаємо на всьому, що було до початку місяця. Перевіряємо на 23 гру 2025 – 24 жов 2026 — ці години метод на момент прогнозу не бачив. Лише чисті години (не обрізані, з достатньою кількістю записів).

**Міри:** MAE — середній промах за годину, Вт; nMAE — він же у % від середньої генерації; зсув — систематичне завищення (+) чи заниження (−)."""),
    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import data_io as io, evaluate as ev
plt.rcParams.update({'figure.figsize': (11, 3.5), 'axes.grid': True, 'grid.alpha': .3})
pd.set_option('display.max_columns', 60)
df, train, test = ev.load()
print(f'навчання (перший рік): {len(train):,} год; перевірка: {len(test):,} год, {test.index.min():%d.%m.%Y} – {test.index.max():%d.%m.%Y}')
print(f'середня генерація в перевірці: {test.pv_w.mean():.0f} Вт')"""),
    md("""## Методи

1. **«Як учора»** — скільки панелі дали вчора в цю годину.
2. **«Радіація × k»** — свіжий прогноз радіації помножити на коефіцієнт, підібраний на минулому.
3. **«Фізика панелей»** — радіацію з прогнозу перерахувати на нахилену площину панелей (pvlib) і помножити на k. Нахил і напрямок панелей підбираємо на першому році. Так роблять готові бібліотеки на кшталт `open-meteo-solar-forecast`.

Кожен — без поправки і **з поправкою на останні 7 днів**: прогноз × (факт / прогноз за останній тиждень). Поправка ловить бруд, тінь, зношення панелей."""),
    code("""orient = ev.fit_orientation(train)
print(f"панелі (найкраще пояснює перший рік): нахил {orient['tilt']}°, напрямок {orient['azimuth']}° (180 — південь, 225 — південний захід)")

clean = df[df.enough & ~df.clipped]
POA_COLS = ['forecast__shortwave_radiation', 'forecast__direct_radiation', 'forecast__diffuse_radiation']

def fp_scale(X, y, Xt):
    return ev.fit_scale(X.forecast__shortwave_radiation, y) * Xt.forecast__shortwave_radiation.to_numpy()

def fp_physics(X, y, Xt):
    o = ev.fit_orientation(X.assign(pv_w=y))
    return o['k'] * ev.poa(Xt, 'forecast', o['tilt'], o['azimuth'])

def persistence_series():
    p = pd.Series(ev.persistence(df, clean.index), index=clean.index)
    cor = p * ev.recent_factor(p.fillna(0), clean.pv_w, 7)
    t = clean.index >= pd.Timestamp(ev.SPLIT, tz='UTC')
    return p[t], cor[t]

results, preds = {}, {}
p, c = persistence_series(); ok = p.notna()
results['як учора'] = ev.metrics(test.pv_w[ok], p[ok]); preds['як учора'] = p
for name, fp in (('радіація × k', fp_scale), ('фізика панелей', fp_physics)):
    raw, cor = ev.walk_forward(df, fp, POA_COLS)
    results[name] = ev.metrics(test.pv_w, raw); preds[name] = raw
    results[name + ' + поправка 7 днів'] = ev.metrics(test.pv_w, cor); preds[name + ' + поправка 7 днів'] = cor
table = pd.DataFrame(results).T.sort_values('MAE, Вт'); table.index.name = 'метод'; table"""),
    md("## По місяцях\n\nСередній промах, Вт. Взимку сонця мало — і промахи менші в абсолюті."),
    code("""bm = pd.DataFrame({k: ev.by_month(test.index, test.pv_w, v.reindex(test.index).fillna(0)) for k, v in preds.items()}).astype(int)
bm.T"""),
    code("""best = table.index[0]
pd.to_pickle({'table': table, 'preds': preds, 'best': best, 'orient': orient}, io.DATA / 'baseline.pkl')
print('найкращий простий метод:', best)"""),
    md("""## Висновки

1. **«Як учора»** — найгірше (518 Вт): погода міняється щодня, вчорашнє сонце мало каже про сьогоднішнє.
2. **«Фізика панелей» краща за «радіація × k»** (363 проти 375 Вт без поправки, 335 проти 348 з поправкою): перерахунок на нахил і напрямок панелей допомагає. Найкраще перший рік пояснює площина з нахилом ~5° на південний захід (240°).
3. **Поправка на останні 7 днів допомагає** (фізика 363 → 335) і прибирає більшу частину завищення (+63 → +27 Вт). Найбільше — взимку (січ 408 → 271, лют 644 → 380): сніг. Влітку (чер–сер) поправка навіть трохи шкодить — тоді панелі працюють як у перший рік, і поправці нічого виправляти.
4. **Планка для моделі: «фізика панелей + поправка 7 днів», 335 Вт** середнього промаху за годину (28 % від середньої генерації). Модель у `05` корисна, лише якщо помиляється менше.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "04_baseline.ipynb")
