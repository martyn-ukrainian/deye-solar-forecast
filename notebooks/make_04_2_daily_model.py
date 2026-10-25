"""Збирає 04_2_daily_model.ipynb: щоденна модель поправки на ковзному вікні останніх днів."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 04_2 · Щоденна модель поправки

Продовження `04_1`. Там поправка — формула (одне число на день) або модель, навчена на всій історії. Тут — **щодня заново** вчимо маленьку модель лише на останніх 7 днях; її результат і є поправка.

**Вікно ковзне:** на 25 жовтня модель вчиться на 18–24 жовтня, на 26-те — на 19–25 (6 старих днів + 1 новий) і т.д. Ціль моделі — наскільки факт розійшовся з базою в ці дні, тож вона бачить саме помилки бази.

На 7 днях лише ~70 сонячних годин, тож модель крихітна: поправка = a + b·(індекс ясності) [+ c·(після полудня)]. Поправка вже не одне число на день, а залежить від години: можна навчитись «у ясні години бруд з'їдає більше» або «після полудня тінь». Свіжі дні важать більше (півжиття 2 дні); ridge (λ = 1) не дає коефіцієнтам скакати.

Перебирали також 14 днів, без ваг і λ = 0.1 — результати близькі (±3 Вт)."""),
    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import data_io as io, evaluate as ev
pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', None)
df, train, test = ev.load()
clean = df[df.enough & ~df.clipped]; y = clean.pv_w; t = test.index
saved = pd.read_pickle(io.DATA / 'corrections.pkl')  # з 04_1
base, lbase = saved['physics']['без поправки'], saved['lgb']['без поправки']

def compare(V, ref_name):
    ref = V[ref_name].reindex(t)
    rows = {k: {**ev.metrics(test.pv_w, v.reindex(t)),
                f'краще за «{ref_name}», % вибірок': '—' if k == ref_name else round(ev.daily_bootstrap(t, test.pv_w, v.reindex(t), ref) * 100)}
            for k, v in V.items()}
    r = pd.DataFrame(rows).T.sort_values('MAE, Вт'); r.index.name = 'поправка'
    for c in ('MAE, Вт', 'RMSE, Вт', 'зсув, Вт'):
        r[c] = r[c].astype(int)
    r['nMAE, %'] = r['nMAE, %'].astype(float)
    r.iloc[:, -1] = [v if v == '—' else int(v) for v in r.iloc[:, -1]]
    return r

X_csi = clean[['csi__forecast']].fillna(0.5)
X_csi_pm = X_csi.assign(pm=(clean.sun_azimuth >= 180).astype(float))
def daily(b, X):
    return b * ev.daily_model_factor(b, y, X, days=7, half_life=2, ridge=1.0)
def table(b, name):
    return compare({'звичайна, 7 днів': b * ev.recent_factor(b, y, 7),
                    'динамічна формула, півжиття 2 дні': b * ev.weighted_factor(b, y, 2),
                    'щоденна модель: ясність': daily(b, X_csi),
                    'щоденна модель: ясність + після полудня': daily(b, X_csi_pm)}, 'динамічна формула, півжиття 2 дні')"""),
    md("## База «фізика панелей»"),
    code("table(base, 'фізика')"),
    md("## База LightGBM (20 ознак)"),
    code("table(lbase, 'LightGBM')"),
    md("""## Висновки

1. **Щоденна модель = формула.** На базі «фізика» 332 проти 332, на базі LightGBM 300 проти 300 — різниця шум (вибірки 33–49 %). Зсув у щоденної моделі менший (на LightGBM +2 проти +15 Вт), але промах той самий.
2. Після очищення даних (обрізання з 93 %) ранок і вечір більше не поводяться по-різному — раніше «тінь після обіду» була батареєю, і щоденна модель її ловила. Тепер ловити нічого.
3. Беремо формулу: простіше і так само точно.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "04_2_daily_model.ipynb")
