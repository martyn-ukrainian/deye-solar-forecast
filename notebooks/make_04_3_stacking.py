"""Збирає 04_3_stacking.ipynb: велика модель поправки на результатах щоденних маленьких моделей (стекінг)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# 04_3 · Стекінг: велика модель на маленьких

Продовження `04_2`. Там щодня — окрема маленька модель на останніх 7 днях (306 моделей на перевірці, кожна не знає про інші). Тут їхні результати стають ознаками **однієї великої моделі**, яка накопичує досвід: «маленька модель казала таке — а поправка виявилась такою».

**Що бачить велика модель (LightGBM) для кожної години:**
- поправку маленької моделі дня на цю годину;
- коефіцієнти маленької моделі: загальний рівень, вплив ясності, різниця після полудня;
- динамічну поправку і поправку за вчора (з `04_1`);
- умови години: індекс ясності, частка розсіяної, висота і напрямок сонця, сніг, температура.

Для порівняння — **варіант 3**: така сама велика модель, але без маленьких. Замість них бачить кожен з 7 днів окремо: поправку дня (факт / база) і скільки тоді обіцяла база (сонячний чи хмарний був день), плюс умови години. Сама має прийти до того, як з 7 днів зробити поправку.

Обидві вчаться множнику факт / база. **Чесно:** лише на днях з 23 гру 2025, де база не бачила відповіді (на першому році база прогнозувала дні, на яких сама вчилась — поправки там нічого не значать). Щомісяця заново на всьому чесному минулому; поки чесних годин < 300 — динамічна формула. Тобто на початку перевірки великій моделі майже нема на чому вчитись, далі — більше."""),
    code("""import sys; sys.path.insert(0, '..')
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import data_io as io, evaluate as ev
pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', None)
df, train, test = ev.load()
clean = df[df.enough & ~df.clipped]; y = clean.pv_w; t = test.index
saved = pd.read_pickle(io.DATA / 'corrections.pkl')  # з 04_1
X_small = clean[['csi__forecast']].fillna(0.5).assign(pm=(clean.sun_azimuth >= 180).astype(float))
MONTHS = ['січ', 'лют', 'бер', 'кві', 'тра', 'чер', 'лип', 'сер', 'вер', 'жов', 'лис', 'гру']

def variants(b):
    return {'динамічна формула, півжиття 2 дні': b * ev.weighted_factor(b, y, 2),
            'маленькі моделі (04_2)': b * ev.daily_model_factor(b, y, X_small, days=7, half_life=2, ridge=1.0),
            'велика модель на маленьких (стекінг)': ev.stacked_correction(df, b, X_small),
            'велика модель на 7 днях окремо (варіант 3)': ev.sequence_correction(df, b)}

def compare(V, ref_name='динамічна формула, півжиття 2 дні'):
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

def months(V):
    bm = pd.DataFrame({k: ev.by_month(t, test.pv_w, v.reindex(t)) for k, v in V.items()}).astype(int)
    return bm.T

P = variants(saved['physics']['без поправки'])
L = variants(saved['lgb']['без поправки'])"""),
    md("## База «фізика панелей»"),
    code("compare(P)"),
    md("По місяцях, середній промах, Вт. Велика модель на початку вчиться на кількох тижнях, далі — на дедалі більшому чесному минулому:"),
    code("months(P)"),
    md("## База LightGBM (20 ознак)"),
    code("compare(L)"),
    code("months(L)"),
    md("""## Висновки

1. **На базі «фізика» велика модель на 7 днях окремо (варіант 3) — найкраща: 313 Вт** проти 332 у формули, краще у 100 % вибірок. Стекінг — 328. Але й тут вони виграють не поправкою, а тим, що бачать умови години і доробляють фізику до ML: виграш улітку (чер 262 проти 299, сер 320 проти 384), а навесні (бер–кві) вони гірші.
2. **На базі LightGBM — гірші за формулу** (варіант 3 — 321, стекінг — 335 проти 300), зсув +36…72 Вт. LightGBM уже знає сонце, хмари і сезон; поправці лишається загальний рівень, а велика модель додає шум.
3. **Зараз беремо LightGBM + динамічна формула (300 Вт).** Великі моделі поправки варто перевірити знову через рік, коли кожен сезон буде вдруге.
"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "solar-forecast", "display_name": "solar-forecast", "language": "python"}})
nbf.write(nb, "04_3_stacking.ipynb")
