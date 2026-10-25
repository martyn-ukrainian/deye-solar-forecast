"""Збирає 01_data.ipynb з комірок нижче. Ноутбук можна далі правити руками."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 01 · Що є в даних Deye\n\nДва роки станції з хмари Deye, крок 5 хв. Мета: зрозуміти дірки, обрізання сонця, тінь і відключення, перш ніж будувати датасет. Час усюди київський."),
    code("""import sys; sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import data_io as io
import warnings; warnings.filterwarnings('ignore', category=UserWarning)
plt.rcParams.update({'figure.figsize': (11, 3.5), 'axes.grid': True, 'grid.alpha': .3})
pd.set_option('display.max_columns', 60)  # таблиці по місяцях широкі — показувати всі колонки
MONTHS = ['січ', 'лют', 'бер', 'кві', 'тра', 'чер', 'лип', 'сер', 'вер', 'жов', 'лис', 'гру']
NAMES = {'pv_w': 'сонце, Вт', 'grid_w': 'мережа, Вт', 'battery_w': 'батарея, Вт (+ віддає)', 'load_w': 'дім, Вт', 'soc': 'заряд, %'}

def when(t):
    \"\"\"Дата для людини: «29 січ 2026 12:40», київський час.\"\"\"
    t = pd.Timestamp(t).tz_convert(io.TZ)
    return f'{t.day} {MONTHS[t.month - 1]} {t.year} {t:%H:%M}'

def month(t):
    \"\"\"«січ 2025».\"\"\"
    return f'{MONTHS[t.month - 1]} {t.year}'

st = io.station()
d = io.deye_frames()
print(st['installedCapacity'], 'кВт')
print(f'{len(d):,} кадрів, {when(d.index.min())} .. {when(d.index.max())}')
d.rename(columns=NAMES).describe().round(0).rename(index={
    'count': 'кадрів', 'mean': 'середнє', 'std': 'розкид', 'min': 'мін', '25%': '25 %', '50%': 'медіана', '75%': '75 %', 'max': 'макс'}).astype(int)"""),
    md("## Дірки\n\nНормальний крок 5 хв, часто 10 (хмара пропускає кадр). Довші дірки — стік був без зв'язку."),
    code("""gap = d.index.to_series().diff().dt.total_seconds() / 60
print(gap.round().value_counts().head(5).rename_axis('крок, хв').rename('кадрів').to_string())
big = gap[gap > 30].sort_values(ascending=False)
print(f'\\nдірок > 30 хв: {len(big)}, разом {big.sum() / 60:.0f} год')
top = big.head(8)
pd.DataFrame({
    'з': [when(t - pd.Timedelta(minutes=m)) for t, m in top.items()],
    'по': [when(t) for t in top.index],
    'годин': (top / 60).round(1).to_numpy(),
})"""),
    md("**Відомі події.** 17 жов 2024: станцію (інвертор + батарея) підключено до хмари, тому жовтень 2024 неповний. До 23 гру 2024 сонця 0: панелі, схоже, ще доробляли, підключили пізніше. 26–29 січ 2026: ожеледиця, згоріли проводи і впали 7 стовпів разом з лінією інтернету, зв'язку не було кілька днів; короткі обриви 5 і 8 лют — ремонти після неї. У навчання ці години не йдуть: беремо лише години, де є ≥ 6 записів."),
    md("## Покриття по місяцях\n\nСкільки кадрів є від можливих (288 на добу)."),
    code("""n = d.pv_w.resample('MS').count()
cov = (n / (n.index.days_in_month * 288) * 100).round(0).astype(int)
# Зверху рік, під ним місяці
cov.index = pd.MultiIndex.from_arrays([cov.index.year, [MONTHS[t.month - 1] for t in cov.index]], names=['рік', 'місяць'])
cov.rename('% кадрів').to_frame().T"""),
    md("## Енергія по місяцях\n\nПрийшло (сонце + мережа) і спожито (дім). Різниця між ними — батарея і втрати інвертора."),
    code("""# Кожен кадр — до наступного, але не довше 15 хв: довшу дірку не вигадуємо
dt_h = d.index.to_series().diff().shift(-1).dt.total_seconds().clip(upper=900).fillna(300) / 3600
# Прийшло (сонце + мережа) і спожито (дім); різниця — батарея і втрати інвертора
kwh = d[['pv_w', 'grid_w', 'load_w']].mul(dt_h, axis=0).div(1000).resample('MS').sum()
kwh.index = [month(t) for t in kwh.index]; kwh.index.name = 'місяць'
x = np.arange(len(kwh)); fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(x - .2, kwh.pv_w, .4, color='gold', label='прийшло: сонце')
ax.bar(x - .2, kwh.grid_w, .4, bottom=kwh.pv_w, color='tab:blue', label='прийшло: мережа')
ax.bar(x + .2, kwh.load_w, .4, color='tab:orange', label='спожито: дім')
ax.set_xticks(x); ax.set_xticklabels(kwh.index, rotation=90, fontsize=8)
ax.set_ylabel('кВт·год'); ax.legend(); plt.show()
table = kwh.round(0).astype(int)
table.columns = pd.MultiIndex.from_tuples([('прийшло, кВт·год', 'сонце'), ('прийшло, кВт·год', 'мережа'), ('спожито, кВт·год', 'дім')])
table[('', 'сонце покрило, %')] = (kwh.pv_w / kwh.load_w * 100).clip(upper=100).fillna(0).round(0).astype(int)
table"""),
    md("## Сонце проти радіації: тінь і орієнтація\n\nПанелі працюють з 23.12.2024 (до того сонця 0), тож беремо лише цей період. Ділимо генерацію на радіацію ERA5 за кожну годину. Беремо лише чисті години (не обрізані). Де відношення падає при тій самій радіації — тінь або панелі туди не дивляться."),
    code("""h = d.resample('1h').mean()
w = io.weather('archive')
h = h.join(w[['shortwave_radiation', 'temperature_2m', 'snow_depth']], how='inner')
# Обрізання: батарея майже повна і з мережі не беремо — інвертор віддає сонця рівно стільки, скільки бере дім
h['clipped'] = (h.soc >= 97) & (h.grid_w < 30) & (h.shortwave_radiation > 50)
day = h[(h.shortwave_radiation > 50) & (h.index >= io.PANELS_SINCE)]
print(f'денних годин {len(day):,}, з них обрізаних {day.clipped.mean():.0%}')
clean = day[~day.clipped]
local = clean.index.tz_convert(io.TZ)
ratio = (clean.pv_w / clean.shortwave_radiation).groupby([local.month, local.hour]).median().unstack()
fig, ax = plt.subplots(figsize=(11, 4))
im = ax.imshow(ratio, aspect='auto', cmap='viridis', vmin=0, vmax=ratio.stack().quantile(.95))
ax.set_yticks(range(len(ratio))); ax.set_yticklabels([MONTHS[m - 1] for m in ratio.index])
ax.set_xticks(range(len(ratio.columns))); ax.set_xticklabels(ratio.columns)
ax.set_xlabel('година'); ax.grid(False)
plt.colorbar(im, label='Вт панелей на 1 Вт/м² радіації'); plt.show()"""),
    code("""prof = clean.groupby(local.hour)[['pv_w', 'shortwave_radiation']].median()
ax = prof.pv_w.plot(label='сонце, Вт')
(prof.shortwave_radiation * prof.pv_w.max() / prof.shortwave_radiation.max()).plot(ax=ax, ls='--', label='радіація (масштаб)')
ax.set_xlabel('година'); ax.legend(); plt.show()
print('пік генерації о', prof.pv_w.idxmax(), ':00, пік радіації о', prof.shortwave_radiation.idxmax(), ':00')"""),
    md("## Обрізання по місяцях\n\nZero export: скільки денних годин сонце було обрізане через повну батарею."),
    code("""dl = day.index.tz_convert(io.TZ)
cl = day.clipped.groupby([dl.year, dl.month]).mean() * 100
cl.index = [f'{MONTHS[m - 1]} {y}' for y, m in cl.index]
ax = cl.plot.bar(width=.8, rot=90, fontsize=8); ax.set_ylabel('% денних годин'); plt.show()"""),
    md("## Розряд батареї (спроба знайти відключення)\n\nХмара не пише стан мережі. Спроба: з мережі 0, батарея розряджається, сонця мало. **Результат: так відключення не знайти.** Більшість епізодів — щоденний розряд 07:00–22:00 (99 → 50 %): тоді інвертор був у режимі, коли дім живе з батареї, а вночі вона заряджається з мережі. Зараз режим інший (з 24.10.2026 видно: при мережі батарея стоїть на 98 %). Справжні відключення беремо лише з Pi (реле + напруга), з 25.10.2026."),
    code("""off = (d.grid_w < 30) & (d.battery_w > 80) & (d.pv_w < d.load_w * .5)
# Склеюємо в епізоди: розрив до 15 хв — та сама подія
t = d.index[off]
brk = t.to_series().diff().dt.total_seconds().gt(15 * 60).to_numpy()
starts = t[brk | (np.arange(len(t)) == 0)]
ends = t[np.r_[brk[1:], True]]
ep = pd.DataFrame({'start': starts.tz_convert(io.TZ), 'end': ends.tz_convert(io.TZ)})
ep['hours'] = (ep.end - ep.start).dt.total_seconds() / 3600 + 5 / 60
ep = ep[ep.hours >= .5]
ep['soc_start'] = d.soc.reindex(ep.start.dt.tz_convert('UTC')).values
ep['soc_end'] = d.soc.reindex(ep.end.dt.tz_convert('UTC')).values
print(f'епізодів розряду ≥ 30 хв: {len(ep)}, разом {ep.hours.sum():.0f} год, найдовше {ep.hours.max():.1f} год')
m = ep.groupby([ep.start.dt.year, ep.start.dt.month]).hours.agg(['count', 'sum']).round(1)
m.index = [f'{MONTHS[mo - 1]} {y}' for y, mo in m.index]; m.index.name = 'місяць'
m.columns = ['епізодів', 'годин']; m"""),
    code("""ax = ep.hours.plot.hist(bins=30); ax.set_xlabel('тривалість, год'); ax.set_ylabel('епізодів'); plt.show()
top = ep.sort_values('hours', ascending=False).head(10)
pd.DataFrame({'з': top.start.map(when), 'по': top.end.map(when), 'годин': top.hours.round(1),
              'заряд на початку, %': top.soc_start, 'в кінці, %': top.soc_end}).reset_index(drop=True)"""),
    md("""## Висновки

1. **Дані.** 738 днів, ~89–96 % кадрів (хмара часто пише раз на 10 хв). Дірок > 30 хв 40, разом 169 год, найдовша 66 год (29 січ 2026). Мережа порожня = 0 з мережі (баланс сходиться).
2. **Панелі з 23 гру 2024.** Для моделі сонця — лише цей період, ~22 місяці.
3. **Форма дня повторює радіацію**, пік разом об 11:00. Відхилення: рано вранці (6–7) і 13–15 генерація нижча за радіацію — схоже на часткову тінь; влітку ввечері (17–20) — вища, панелі ловлять західне сонце краще, ніж горизонталь. Орієнтацію рядків перевірити на pv1/pv2 з Pi.
4. **Зима слабка непропорційно.** Груд–лют ~2–3 Вт на 1 Вт/м², літо ~5–6. Низьке сонце (тінь) і сніг. Моделі потрібні висота сонця і сніг.
5. **Обрізання** 22 % денних годин, влітку до 35 %. Без фільтра модель недооцінить панелі якраз у найсонячніші дні.
6. **Відключень з хмари не дістати**: режим батареї в минулому був інший. Для прогнозу батареї спиратимемось на історію Pi.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "01_data.ipynb")
