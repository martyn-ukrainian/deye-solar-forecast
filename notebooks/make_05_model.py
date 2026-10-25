"""Збирає 05_model.ipynb: LightGBM з різними наборами ознак, нейромережа, порівняння з планкою з 04."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 05 · Модель сонця «сьогодні»

LightGBM на ознаках з `03`, порівняння наборів ознак «з ними і без них», невелика нейромережа (PyTorch) і зведена таблиця проти простих методів з `04`.

Перевірка та сама, що в `04`: щомісяця перенавчання на всьому минулому, перевірка на 23 гру 2025 – 24 жов 2026, чисті години. Кожен варіант — з поправкою на останні 7 днів (так працюватиме на Pi).

**Як читати «краще за повний у X % вибірок»:** беремо випадкові набори днів перевірки 2000 разів і рахуємо, як часто варіант точніший за повний набір. ~50 % — різниці нема (шум), ≥ 95 % — варіант справді кращий, ≤ 5 % — справді гірший."""),
    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, lightgbm as lgb
import data_io as io, evaluate as ev
from labels import label
plt.rcParams.update({'figure.figsize': (11, 3.5), 'axes.grid': True, 'grid.alpha': .3})
pd.set_option('display.max_columns', 60)
df, train, test = ev.load()
base = pd.read_pickle(io.DATA / 'baseline.pkl')

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
def fp_lgb(X, y, Xt):
    return lgb.LGBMRegressor(**PARAMS).fit(X, y).predict(Xt)
print('планка з 04:', base['best'], base['table'].loc[base['best'], 'MAE, Вт'], 'Вт')"""),
    md("## Варіанти ознак\n\nПовний набір «сьогодні» (20 ознак) і без окремих груп — те, що ми обговорювали: день року, температура, динаміка прогнозу, індекси. Плюс «сирі ознаки» — як було до розплутування дублікатів (три версії прогнозу як є, пряма/розсіяна, радіація ясного неба)."),
    code("""results, preds = {}, {}
for name, feats in ev.variants().items():
    raw, cor = ev.walk_forward(df, fp_lgb, feats)
    preds[name] = cor
    results[name] = {**ev.metrics(test.pv_w, cor), 'ознак': len(feats)}
full = preds['повний (20 ознак)']
for name in results:
    results[name]['краще за повний, % вибірок'] = '—' if name.startswith('повний') else round(ev.daily_bootstrap(test.index, test.pv_w, preds[name], full) * 100)
var = pd.DataFrame(results).T; var.index.name = 'LightGBM + поправка 7 днів'; var"""),
    md("## Стеля: якби погоду знали точно\n\nТа сама модель, але замість прогнозу — факт погоди ERA5 (без динаміки: у факту версій нема). Скільки точності губиться через неточний прогноз погоди."),
    code("""ceil_feats = [c.replace('forecast__', 'archive__').replace('csi__forecast', 'csi__archive').replace('dfrac__forecast', 'dfrac__archive')
              for c in ev.variants()['повний (20 ознак)'] if not c.startswith('dyn__')]
_, ceil = ev.walk_forward(df, fp_lgb, ceil_feats)
print('стеля (факт погоди):', ev.metrics(test.pv_w, ceil))"""),
    md("## Нейромережа (PyTorch)\n\nДва шари по 64 і 32 нейрони, вчиться на тому самому середньому промаху. Пропуски на світанку → медіана + прапорець; ознаки нормовані. Ті самі 20 ознак і та сама перевірка."),
    code("""raw_mlp, mlp = ev.walk_forward(df, ev.mlp_fit_predict, ev.variants()['повний (20 ознак)'])
print('нейромережа + поправка 7 днів:', ev.metrics(test.pv_w, mlp))"""),
    md("## Зведене порівняння: хто кращий\n\nУсі методи з `04` і `05` поруч, від найточнішого."),
    code("""allp = {**{f'простий: {k}': v for k, v in base['preds'].items()},
        **{f'LightGBM: {k}': v for k, v in preds.items()},
        'нейромережа (PyTorch) + поправка 7 днів': mlp,
        'стеля: LightGBM на факті погоди (недосяжно)': ceil}
summary = pd.DataFrame({k: ev.metrics(test.pv_w[v.reindex(test.index).notna()], v.reindex(test.index).dropna()) for k, v in allp.items()}).T
summary = summary.sort_values('MAE, Вт'); summary.index.name = 'метод'
best_base = base['preds'][base['best']]
best_name = f"простий: {base['best']}"
summary['краще за найкращий простий, % вибірок'] = [
    '—' if k == best_name else round(ev.daily_bootstrap(test.index, test.pv_w, allp[k].reindex(test.index).fillna(0), best_base) * 100)
    for k in summary.index]
summary"""),
    code("""top = summary.index[summary.index.str.startswith('LightGBM')][0]
fig, ax = plt.subplots(figsize=(8, 4))
s = summary.drop(index=[i for i in summary.index if i.startswith('стеля')])['MAE, Вт']
colors = ['tab:green' if i == top else 'tab:blue' if i.startswith(('LightGBM', 'нейро')) else 'tab:gray' for i in s.index]
ax.barh(range(len(s)), s, color=colors); ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
ax.invert_yaxis(); ax.set_xlabel('середній промах за годину, Вт (менше — краще)')
ax.axvline(summary.loc[summary.index.str.startswith('стеля'), 'MAE, Вт'].iloc[0], ls='--', c='k', lw=1, label='стеля (факт погоди)')
ax.legend(); plt.show()"""),
    md("## По місяцях: найкраща модель проти найкращого простого методу"),
    code("""bm = pd.DataFrame({
    f"простий: {base['best']}": ev.by_month(test.index, test.pv_w, best_base.reindex(test.index)),
    top: ev.by_month(test.index, test.pv_w, allp[top].reindex(test.index)),
}).astype(int)
bm['виграш моделі, Вт'] = bm.iloc[:, 0] - bm.iloc[:, 1]
bm.T"""),
    md("## Що найбільше впливає на прогноз\n\nВажливість ознак у моделі, навченій на всіх чистих годинах (скільки разом зменшила помилку кожна ознака)."),
    code("""clean = df[df.enough & ~df.clipped]
feats = ev.variants()['повний (20 ознак)']
final = lgb.LGBMRegressor(**PARAMS, importance_type='gain').fit(clean[feats], clean.pv_w)
imp = pd.Series(final.feature_importances_, index=feats)
imp = (imp / imp.sum() * 100).sort_values(ascending=False).round(1)
pd.DataFrame({'ознака': [label(x) for x in imp.index], 'частка, %': imp.to_numpy(), 'змінна в коді': imp.index}).set_index('ознака')"""),
    md("## Панелі на другий рік дають менше\n\nЗа того самого сонця (радіація ERA5 > 150 Вт/м², чисті години) — скільки Вт панелі дають на 1 Вт/м², по місяцях двох років. Звідси частина завищення без поправки."),
    code("""u = df[df.enough & ~df.clipped & (df.archive__shortwave_radiation > 150)].copy()
loc = u.index.tz_convert(io.TZ)
r = (u.pv_w / u.archive__shortwave_radiation).groupby([loc.month, loc.year]).median().unstack().round(2)
MONTHS = ['січ', 'лют', 'бер', 'кві', 'тра', 'чер', 'лип', 'сер', 'вер', 'жов', 'лис', 'гру']
r = r[[2025, 2026]].dropna(); r.index = [MONTHS[m - 1] for m in r.index]; r.index.name = 'місяць'
r['2026 від 2025, %'] = (r[2026] / r[2025] * 100).round(0).astype(int); r.T"""),
    md("""## Висновки

1. **Модель потрібна: LightGBM + поправка — 304 Вт** (25 % від середньої генерації) проти 335 у найкращого простого методу, краще у 100 % вибірок. Найбільший виграш — січ (+76 Вт), чер (+65), сер (+59).
2. **До стелі лишилось 13 Вт.** З ідеальною погодою (факт ERA5) було б 291. Решта промаху — неточний прогноз погоди і те, чого в погоді нема (бруд, сніг на панелях, шум усередині години).
3. **Варіанти ознак:** без температури гірше (вибірки 4 %) — температура потрібна (гарячі панелі дають менше); без динаміки прогнозу гірше (3 %); без дня року — те саме (50 %); сирі ознаки до розплутування — 301, різниця шум (84 %). Лишаємо 20 ознак.
4. **Нейромережа (PyTorch) — 334 Вт**, на рівні простої фізики. На ~6 тис. годин дерева працюють краще.
5. **Важливість:** радіація зі свіжого прогнозу — 67 %, висота сонця — 14 %, індекс ясності — 4 %, решта по 1–2 %.
6. **Панелі на другий рік** — див. таблицю: різниця з першим роком помітно менша, ніж до очищення даних (частина «слабкості» була притиснутим батареєю сонцем). Стан панелей розбирається в `06`.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "05_model.ipynb")
