"""Збирає 03_features.ipynb: які ознаки є, що з чим дублюється, фінальний набір «сьогодні»."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 03 · Ознаки моделі сонця\n\nЩо з погоди і положення сонця пов'язане з генерацією, які ознаки дублюють одна одну і як їх розплутали. В кінці — набір для моделі **«сьогодні»** (скільки сонце дасть сьогодні). Таблиця — з `02_dataset`."),
    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import data_io as io, dataset
from labels import SETS, VARS, OWN, GROUP_ORDER, label, group
plt.rcParams.update({'figure.figsize': (11, 3.5), 'axes.grid': True, 'grid.alpha': .3})
pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', None)

df = pd.read_parquet(io.DATA / 'dataset.parquet')
use = df.enough & ~df.clipped
print(f'годин для навчання: {use.sum():,}')"""),
    md("## Які ознаки є\n\nУ навчанні буде лише те, що відомо **наперед**: прогнози, сонце, дата. Факт ERA5 — тільки для порівняння (скільки точності губить прогноз)."),
    code("""feats = [c for c in df.columns if c not in dataset.NOT_FEATURES]
f = pd.DataFrame({'група': [group(c) for c in feats], 'ознака': [label(c).split(': ')[-1] for c in feats], 'змінна в коді': feats})
f.sort_values('група', key=lambda g: g.map({v: i for i, v in enumerate(GROUP_ORDER)}), kind='stable').set_index(['група', 'ознака'])"""),
    md("## Що найбільше пов'язане з генерацією\n\nКореляція з генерацією на чистих годинах (від −1 до 1; ближче до ±1 — сильніший зв'язок). Погода — по наборах: видно, скільки зв'язку губиться, коли прогноз дальший."),
    code("""c = df[use].corr(numeric_only=True)['pv_w']
w = pd.DataFrame({SETS[k]: {**{VARS[v]: c.get(f'{k}__{v}') for v in VARS},
                            'індекс ясності': c.get(f'csi__{k}'), 'частка розсіяної': c.get(f'dfrac__{k}')} for k in SETS}).dropna(how='all')
w = w.reindex(w['факт (ERA5)'].abs().sort_values(ascending=False).index).round(2)
w.index.name = 'ознака'
w['змінна в коді'] = [next((k for k, v in VARS.items() if v == n), {'індекс ясності': 'csi', 'частка розсіяної': 'dfrac'}.get(n)) for n in w.index]
w"""),
    code("""o = pd.DataFrame({'ознака': list(OWN.values()), 'кореляція з генерацією': [round(c[k], 2) for k in OWN], 'змінна в коді': list(OWN)})
o.reindex(o['кореляція з генерацією'].abs().sort_values(ascending=False).index).set_index('ознака')"""),
    md("## Ознаки, що дублюють одна одну\n\nДекілька пар кажуть майже одне й те саме:"),
    code("""pair = ['sun_elevation', 'clearsky_ghi', 'forecast__shortwave_radiation', 'forecast__direct_radiation', 'day1__shortwave_radiation',
        'doy_cos', 'forecast__temperature_2m']
df[use][pair].rename(columns=label).corr().round(2)"""),
    md("""Як розплутали:

- **Висота сонця ↔ радіація ясного неба 0.99** (ясне небо рахується з висоти сонця). Замість ясного неба — **індекс ясності** (clear-sky index) = прогноз радіації / радіація ясного неба: 1 — ясно, 0.1 — суцільні хмари. Стандартний прийом у сонячних прогнозах: готова функція `pvlib.irradiance.clearsky_index`; [Lauret et al., Solar 2022](https://hal.univ-reunion.fr/hal-03964785v1) — більшість прогнозистів моделюють саме індекс, і нормування на ясне небо дає кращі прогнози незалежно від типу моделі.
- **Загальна ↔ пряма радіація 0.94** (загальна = пряма + розсіяна). Замість прямої і розсіяної — **частка розсіяної** = розсіяна / загальна: ~1 — димка чи хмари, ~0.2 — пряме сонце.
- **Свіжий ↔ за добу 0.96** (і так для всіх величин, 0.83–1.0). Сирі day1/day2 відновлюються зі свіжого + змін, тож беремо **свіжий + зміни** (day1 − day2, свіжий − day1, розкид).
- **День року ↔ температура −0.80** — пора року. Лишаємо обидві, але перевіримо в моделі з ними і без них: температура може нести своє (гарячі панелі дають менше).
- День року ↔ висота сонця лише −0.37: висота міняється протягом дня, день року — ні. Не дублікати."""),
    code("""pd.DataFrame({'висота сонця ↔ індекс ясності': [round(df[use].sun_elevation.corr(df[use].csi__forecast), 2)],
              'загальна радіація ↔ частка розсіяної': [round(df[use].forecast__shortwave_radiation.corr(df[use].dfrac__forecast), 2)]},
             index=['кореляція після розплутування'])"""),
    md("## Набір ознак моделі «сьогодні»\n\nСкільки сонце дасть сьогодні. Основа — свіжий прогноз (він є лише на найближчі години). Модель «завтра» (на прогнозі за добу) — окремо, пізніше."),
    code("""today = dataset.features('today')
TRY = {'doy_sin', 'doy_cos', 'forecast__temperature_2m'}
pd.DataFrame({'ознака': [label(x) for x in today], 'змінна в коді': today,
              'перевірити з/без': ['так' if x in TRY else '' for x in today]}).set_index('ознака')"""),
    md("Пари, що лишились з кореляцією ≥ 0.7:"),
    code("""cc = df[use][today].corr()
pairs = [(label(a), label(b), round(cc.loc[a, b], 2)) for i, a in enumerate(today) for b in today[i + 1:] if abs(cc.loc[a, b]) >= .7]
pd.DataFrame(pairs, columns=['ознака', 'ознака ', 'кореляція']).sort_values('кореляція', key=abs, ascending=False).reset_index(drop=True)"""),
    md("Вище 0.9 нема. Залишились природні: висота сонця ↔ радіація (вище сонце — більше світла), індекс ясності ↔ хмарність ↔ частка розсіяної (три погляди на хмари), день року ↔ температура (перевіримо)."),
    md("## Чи показує розкид прогнозів, наскільки прогноз помилиться\n\nРозкид — наскільки три версії прогнозу (за 2 доби, за добу, свіжий) розходились на ту саму годину. Промах — наскільки свіжий прогноз радіації розійшовся з фактом ERA5.\n\nПорівнюємо **у відсотках від ясного неба**: інакше полудень завищує і розкид, і промах просто тому, що там більше сонця."),
    code("""u = df[use].copy()
LEVELS = ['майже однакові', 'мало', 'середньо', 'помітно', 'сильно']
u['miss'] = (u.archive__shortwave_radiation - u.forecast__shortwave_radiation).abs()
u['spread_rel'] = u.dyn__shortwave_radiation__spread / u.clearsky_ghi * 100
u['miss_rel'] = u.miss / u.clearsky_ghi * 100
u['версії розходились'] = pd.qcut(u.spread_rel, 5, labels=LEVELS)
t = u.groupby('версії розходились', observed=True)[['spread_rel', 'miss_rel']].median().round(1)
t.columns = ['розкид, % ясного неба', 'промах, % ясного неба']; t"""),
    md("Те саме всередині годин зі схожим сонцем (за радіацією ясного неба), промах у Вт/м²:"),
    code("""rows = {}
for lo, hi, name in ((150, 350, 'слабке 150–350'), (350, 550, 'середнє 350–550'), (550, 900, 'сильне 550–900')):
    s = u[(u.clearsky_ghi >= lo) & (u.clearsky_ghi < hi)].copy()
    s['g'] = pd.qcut(s.dyn__shortwave_radiation__spread, 5, labels=LEVELS)
    rows[f'сонце {name} ({len(s)} год)'] = s.groupby('g', observed=True).miss.median().round(0).astype(int)
r = pd.DataFrame(rows).T; r.columns.name = 'версії розходились →'; r"""),
    code("""print(f"рангова кореляція розкид ↔ промах: {u.spread_rel.corr(u.miss_rel, method='spearman'):.2f}")"""),
    md("""## Висновки

1. **Головна ознака — радіація**: кореляція з генерацією 0.91 з фактом і зі свіжим прогнозом, 0.90 за добу, 0.88 за 2 доби. Прогноз за 2 доби губить небагато. Далі — пряма радіація, висота сонця. Вологість пов'язана обернено (вологість ≈ хмари, туман).
2. **Дублікати розплутано.** Радіацію ясного неба замінили індексом ясності (з висотою сонця тепер −0.12); пряму/розсіяну — часткою розсіяної; сирі day1/day2 — змінами.
3. **Модель «сьогодні» — 20 ознак**, пар вище 0.9 нема. День року і температура — перевірено в `05`: температура потрібна, день року — ні.
4. **Розкид прогнозів працює, але помірно.** У % від ясного неба: коли версії майже однакові, промах ~6 %; коли сильно розходяться — ~25 % (у 4 рази). Всередині схожого сонця — у 2–6 разів, найсильніше в яскраві години. Рангова кореляція 0.34. Беремо як ознаку і для меж прогнозу («у гіршому разі»).
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "03_features.ipynb")
