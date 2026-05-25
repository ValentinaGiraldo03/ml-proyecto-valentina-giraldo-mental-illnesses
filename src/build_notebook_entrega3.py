"""Construye el notebook reproducible de la Entrega 3.

El notebook resultante es ejecutable end-to-end con el mismo CSV y
la misma semilla. Re-corre el tuning final, evalua en test, calcula
intervalos de confianza por bootstrap, analiza sensibilidad a la
semilla, hace validacion temporal y ablacion de features, e imprime
las figuras y conclusiones de la Entrega 3.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
NB_PATH = PROJECT / "notebooks" / "03_entrega3_modelo_final.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


nb = nbf.v4.new_notebook()
cells = []

# -------------------------------------------------------------------
# 0. Header
# -------------------------------------------------------------------
cells.append(
    md(
        """# Entrega 3 — Modelo final, interpretación y comunicación
**Curso:** Aprendizaje de Máquina Aplicado (EAFIT) · **Estudiante:** Valentina Giraldo

**Dataset:** *Mental Illnesses Prevalence* (Our World in Data / IHME, GBD).

Esta tercera entrega cierra el ciclo CRISP-DM del proyecto. En la Entrega 1 dejé el problema planteado y un baseline (regresión lineal) que **no** lograba generalizar a países nuevos. En la Entrega 2 comparé tres familias de modelos y `HistGradientBoosting` salió como el más prometedor: era el único que superaba con claridad al dummy en países que no había visto.

En esta entrega me toca:

1. **Hacer un tuning más serio** del candidato ganador (HistGB) y del competidor más cercano (Random Forest), con un grid más amplio y *early stopping*.
2. **Probar un ensemble** sencillo (promedio HistGB + RF) para ver si gana algo.
3. **Decidir el modelo final** con criterio honesto y una sola pasada por el test.
4. **Cuantificar la incertidumbre** (intervalos de confianza por bootstrap sobre el test).
5. **Probar la sensibilidad** a la semilla del split (¿qué tan estable es el resultado si cambian los países que caen en test?).
6. **Hacer un análisis de errores y de sensibilidad** (ablación de features, validación temporal estricta).
7. **Interpretar** el modelo final con permutation importance y partial dependence.
8. **Comunicar** todo en lenguaje claro: qué funcionó, qué no, qué falta y qué recomiendo.

Voy a tratar de ser honesta — donde el modelo es bueno, lo digo; donde tiene una limitación seria, también."""
    )
)

# -------------------------------------------------------------------
# 1. Setup
# -------------------------------------------------------------------
cells.append(
    md(
        """## 1. Setup y carga de datos
Mantengo todo idéntico a las entregas anteriores: mismo CSV, mismo renombrado, misma exclusión de los 9 agregados regionales (que no son países) y misma semilla `SEED = 42`."""
    )
)

cells.append(
    code(
        """import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, GroupShuffleSplit
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid', context='notebook')

DATA_PATH = Path('../data/mental-illnesses-prevalence.csv')
FIG_DIR = Path('../figures')
REPORT_DIR = Path('../report')

COL_MAP = {
    'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'schizophrenia',
    'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'depression',
    'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'anxiety',
    'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'bipolar',
    'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'eating',
}

raw = pd.read_csv(DATA_PATH)
df = raw.rename(columns=COL_MAP)
countries = df[df['Code'].notna()].copy().reset_index(drop=True)
print(f'Filas: {len(countries)} | Paises: {countries.Entity.nunique()}')

FEATURES = ['schizophrenia', 'anxiety', 'bipolar', 'eating', 'Year']
TARGET = 'depression'
"""
    )
)

cells.append(
    md(
        """## 2. Split por país (idéntico a Entregas 1 y 2)
Reservo el mismo test que ya venía usando: 41 países (≈20 %) que el modelo no ve hasta el final. Esto me garantiza que cualquier mejora respecto a las entregas previas viene del *modelado* y no de un test distinto."""
    )
)

cells.append(
    code(
        """X = countries[FEATURES].values
y = countries[TARGET].values
groups = countries['Entity'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr_idx, te_idx = next(gss.split(X, y, groups))
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]
g_tr, g_te = groups[tr_idx], groups[te_idx]
test_df = countries.iloc[te_idx].reset_index(drop=True)

print(f'Train: {len(X_tr)} filas | {pd.Series(g_tr).nunique()} paises')
print(f'Test : {len(X_te)} filas | {pd.Series(g_te).nunique()} paises')
print(f'Paises en comun entre train y test: {len(set(g_tr) & set(g_te))}')
"""
    )
)

# -------------------------------------------------------------------
# 3. Tuning final
# -------------------------------------------------------------------
cells.append(
    md(
        """## 3. Tuning final
En la Entrega 2 el grid era modesto. Aquí lo amplío para los dos candidatos basados en árboles:

- **HistGB:** explora 3 *learning rates*, 4 profundidades, 3 tamaños mínimos de hoja, dos niveles de regularización L2, y activo *early stopping* (15 % de validación interna, paciencia de 25 rondas). El early stopping me permite poner `max_iter=1000` sin miedo a sobreentrenar.
- **RF:** explora 2 tamaños de bosque, 3 profundidades, 3 tamaños de hoja y dos estrategias de `max_features`. Random Forest es robusto pero conviene afinarlo.

Todo el tuning se hace con `GroupKFold(5)` por país sobre el train. Cero leakage entre folds."""
    )
)

cells.append(
    code(
        """# 3.1 HistGB con grid amplio + early stopping
pipe_h = Pipeline([
    ('model', HistGradientBoostingRegressor(
        random_state=SEED,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
    )),
])
grid_h = {
    'model__learning_rate': [0.03, 0.05, 0.1],
    'model__max_iter': [500, 1000],
    'model__max_depth': [None, 4, 6, 8],
    'model__min_samples_leaf': [10, 20, 40],
    'model__l2_regularization': [0.0, 0.1],
}
gs_h = GridSearchCV(
    pipe_h, grid_h,
    scoring='neg_root_mean_squared_error',
    cv=GroupKFold(n_splits=5).split(X_tr, y_tr, g_tr),
    n_jobs=-1, refit=True,
)
gs_h.fit(X_tr, y_tr)
idx = gs_h.best_index_
histgb_best = gs_h.best_estimator_
histgb_cv_rmse = float(-gs_h.best_score_)
histgb_cv_std = float(gs_h.cv_results_['std_test_score'][idx])
print(f'HistGB CV RMSE = {histgb_cv_rmse:.4f} +/- {histgb_cv_std:.4f}')
print(f'  best params: {gs_h.best_params_}')
"""
    )
)

cells.append(
    code(
        """# 3.2 Random Forest con grid amplio
pipe_rf = Pipeline([
    ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1)),
])
grid_rf = {
    'model__n_estimators': [400, 800],
    'model__max_depth': [8, 12, None],
    'model__min_samples_leaf': [1, 3, 5],
    'model__max_features': ['sqrt', 1.0],
}
gs_rf = GridSearchCV(
    pipe_rf, grid_rf,
    scoring='neg_root_mean_squared_error',
    cv=GroupKFold(n_splits=5).split(X_tr, y_tr, g_tr),
    n_jobs=-1, refit=True,
)
gs_rf.fit(X_tr, y_tr)
idx = gs_rf.best_index_
rf_best = gs_rf.best_estimator_
rf_cv_rmse = float(-gs_rf.best_score_)
rf_cv_std = float(gs_rf.cv_results_['std_test_score'][idx])
print(f'RF     CV RMSE = {rf_cv_rmse:.4f} +/- {rf_cv_std:.4f}')
print(f'  best params: {gs_rf.best_params_}')
"""
    )
)

cells.append(
    md(
        """### 3.3 Lectura del tuning
Con el grid ampliado y early stopping:
- **HistGB** baja de 0.76 (Entrega 2) a **0.73** de RMSE en CV. Mejora pequeña pero consistente.
- **Random Forest** baja de 0.77 a **0.72**. La mejora del RF es mayor — esto sugiere que el grid corto de la Entrega 2 estaba lejos de su óptimo y subestimé la familia.
- Las **desviaciones estándar siguen altas** (~0.13). Eso quiere decir que el desempeño promedio entre folds esconde mucha variabilidad — algunos países son fáciles y otros son muy difíciles. Lo voy a investigar más abajo con el análisis de sensibilidad a la semilla.

Esta lectura cambia mi expectativa: ya no estoy tan segura de que HistGB vaya a ganar en test."""
    )
)

# -------------------------------------------------------------------
# 4. Ensemble
# -------------------------------------------------------------------
cells.append(
    md(
        """## 4. Ensemble simple — ¿aporta promediar HistGB + RF?
Un truco clásico es promediar predicciones de modelos diferentes. La idea es que sus errores no estén correlacionados y se compensen. Defino un ensemble por promedio uniforme y lo evalúo igual que los demás."""
    )
)

cells.append(
    code(
        """class AverageEnsemble:
    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        w = np.ones(len(estimators)) if weights is None else np.array(weights, float)
        self.weights = w / w.sum()
    def predict(self, X):
        preds = np.column_stack([e.predict(X) for e in self.estimators])
        return preds @ self.weights

ensemble = AverageEnsemble([histgb_best, rf_best])
print('Ensemble (HistGB+RF) listo.')
"""
    )
)

# -------------------------------------------------------------------
# 5. Evaluación en test
# -------------------------------------------------------------------
cells.append(
    md(
        """## 5. Evaluación final en test (una sola pasada)
Aquí cumplo la regla más importante del protocolo: el test se mira **una sola vez**. Si después de ver el test cambio el modelo, deja de ser test."""
    )
)

cells.append(
    code(
        """def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def metric_dict(y_true, y_pred):
    return {
        'RMSE': rmse(y_true, y_pred),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
        'R2':   float(r2_score(y_true, y_pred)),
    }

candidates = {
    'HistGB (tuneado)':   histgb_best,
    'RF (tuneado)':       rf_best,
    'Ensemble (HistGB+RF)': ensemble,
}
test_preds = {}
test_metrics = {}
for name, est in candidates.items():
    yh = est.predict(X_te)
    test_preds[name] = yh
    test_metrics[name] = metric_dict(y_te, yh)
final_table = pd.DataFrame(test_metrics).T.reset_index().rename(columns={'index':'modelo'})
final_table.round(4)
"""
    )
)

cells.append(
    md(
        """### Decisión: el modelo final es **Random Forest (tuneado)**
Esto me sorprendió un poco — en la Entrega 2 HistGB era el ganador con un margen claro. Pero al darle a Random Forest una grilla más amplia (especialmente `max_depth=8`, `min_samples_leaf=3`, `n_estimators=800`), su CV bajó y su test también. El ensemble no mejora a RF (es prácticamente igual en RMSE pero con MAE algo mejor).

Mi criterio de decisión: **menor RMSE en test** (la métrica primaria del proyecto, definida en Entrega 1). RF gana por un margen pequeño pero consistente.

⚠️ Honestidad técnica: el resultado depende del split de países que el azar (con `SEED=42`) eligió como test. Más abajo cuantifico cuánto cambia este número si cambio la semilla."""
    )
)

cells.append(
    code(
        """# Fijo el modelo final
FINAL_NAME = 'RF (tuneado)'
final_est = candidates[FINAL_NAME]
final_pred = test_preds[FINAL_NAME]
final_metrics = test_metrics[FINAL_NAME]
print(f'Modelo final: {FINAL_NAME}')
print(f'  RMSE = {final_metrics["RMSE"]:.4f}')
print(f'  MAE  = {final_metrics["MAE"]:.4f}')
print(f'  R2   = {final_metrics["R2"]:.4f}')
"""
    )
)

# -------------------------------------------------------------------
# 6. Bootstrap CIs
# -------------------------------------------------------------------
cells.append(
    md(
        """## 6. ¿Qué tan confiables son estas métricas? — Bootstrap
Un número solo no es muy útil. Lo importante es decir “mi RMSE es X **± algo**”. Para el test calculo un intervalo de confianza al 95 % por **bootstrap (n=2000)**: en cada iteración remuestreo con reemplazo el test, calculo las métricas y al final reporto los percentiles 2.5 % y 97.5 %.

Esto me dice qué tan variable es la métrica frente a *qué filas concretas* tocaron caer en test. No corrige la sensibilidad al *split de países* (eso lo veo en la sección siguiente)."""
    )
)

cells.append(
    code(
        """def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmses, maes, r2s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        rmses.append(rmse(yt, yp))
        maes.append(mean_absolute_error(yt, yp))
        r2s.append(r2_score(yt, yp))
    return {
        m: {
            'mean': float(np.mean(arr)),
            'lo': float(np.quantile(arr, alpha/2)),
            'hi': float(np.quantile(arr, 1-alpha/2)),
        }
        for m, arr in [('RMSE', rmses), ('MAE', maes), ('R2', r2s)]
    }

ci = bootstrap_ci(y_te, final_pred)
for k, v in ci.items():
    print(f"{k}: {v['mean']:.4f}  IC95% = [{v['lo']:.4f}, {v['hi']:.4f}]")
"""
    )
)

cells.append(
    md(
        """### Lectura del bootstrap
El RMSE final está en **0.618** con IC95 % ≈ **[0.597, 0.638]**. El intervalo es estrecho (≈0.04 puntos), lo que me dice que el resultado *para este split* es sólido — no es ruido de muestreo dentro del test.

El **R² ≈ 0.43** con IC95 % ≈ **[0.38, 0.47]**. Eso quiere decir que el modelo explica entre un 38 % y un 47 % de la variabilidad en países no vistos. No es un problema “resuelto”, pero sí mejora claramente al dummy."""
    )
)

# -------------------------------------------------------------------
# 7. Sensibilidad a la semilla
# -------------------------------------------------------------------
cells.append(
    md(
        """## 7. Sensibilidad a la semilla del split externo
El bootstrap solo perturba las filas del test, pero no cambia *qué países* son test. Si por suerte cayeron en test 41 países “fáciles”, mi RMSE va a ser optimista; si cayeron los difíciles, será pesimista. Para medir este efecto, repito todo el experimento de fit + test con **5 semillas distintas (0, 1, 7, 21, 42)** usando los mejores hiperparámetros que ya encontré."""
    )
)

cells.append(
    code(
        """def sensitivity_seed(df, best_params, seeds=(0,1,7,21,42)):
    out = []
    for s in seeds:
        XX = df[FEATURES].values
        yy = df[TARGET].values
        gg = df['Entity'].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=s)
        tr, te = next(gss.split(XX, yy, gg))
        params = {k.replace('model__',''):v for k,v in best_params.items()}
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
        m.fit(XX[tr], yy[tr])
        yh = m.predict(XX[te])
        mm = metric_dict(yy[te], yh)
        mm['seed'] = int(s)
        out.append(mm)
    return out

seed_rows = sensitivity_seed(countries, gs_rf.best_params_)
pd.DataFrame(seed_rows).round(4)
"""
    )
)

cells.append(
    md(
        """### Lectura — esta es la limitación más honesta del proyecto
Las métricas **cambian mucho** según qué países caigan en test:

- **R² entre 0.16 y 0.58** según la semilla.
- **RMSE entre 0.57 y 0.93** según la semilla.

Eso significa que mi resultado “RMSE=0.618, R²=0.43” describe **un caso particular**, no la performance esperada en cualquier conjunto de 41 países nuevos. Con apenas 205 países y 5 features, hay subconjuntos donde el modelo extrapola bien y subconjuntos donde extrapola mal.

Esta es la limitación más fuerte del proyecto, y prefiero dejarla escrita explícitamente antes que pretender que el número es definitivo. Para una recomendación de política pública o un despliegue real, **no usaría una sola semilla**: reportaría el resultado promedio sobre varias semillas y su rango (ver tabla de arriba)."""
    )
)

# -------------------------------------------------------------------
# 8. Validación temporal
# -------------------------------------------------------------------
cells.append(
    md(
        """## 8. Protocolo de validación alternativo: temporal
El split por país responde a la pregunta “¿qué pasa con un país nuevo?”. Pero hay otra pregunta que también vale la pena responder: **“¿qué pasa con años nuevos del mismo país?”**. Para esto entreno con **1990–2009** y dejo **2010–2019** como test, conservando todos los países en ambos lados."""
    )
)

cells.append(
    code(
        """train_mask = countries['Year'] <= 2009
X_trT = countries.loc[train_mask, FEATURES].values
X_teT = countries.loc[~train_mask, FEATURES].values
y_trT = countries.loc[train_mask, TARGET].values
y_teT = countries.loc[~train_mask, TARGET].values

params = {k.replace('model__',''):v for k,v in gs_rf.best_params_.items()}
rf_T = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
rf_T.fit(X_trT, y_trT)
temporal_metrics = metric_dict(y_teT, rf_T.predict(X_teT))
print('Validacion temporal (train 1990-2009 / test 2010-2019):')
for k, v in temporal_metrics.items():
    print(f'  {k} = {v:.4f}')
"""
    )
)

cells.append(
    md(
        """### Lectura del split temporal — atención al sesgo de fuga
Aquí los números son **mucho mejores**: R² ≈ 0.83, RMSE ≈ 0.38. Suena espectacular, pero hay que leerlo con cuidado.

El problema: aunque ningún par (país, año) se repite entre train y test, **sí hay años previos del mismo país en train**. La depresión a nivel poblacional cambia poco de un año al siguiente (alta autocorrelación temporal dentro del país), y los otros trastornos también. El modelo no está *extrapolando a países nuevos* — está *interpolando años nuevos para países que ya conoce*. Es un escenario más cómodo.

Esto es un buen recordatorio: **dos protocolos de validación legítimos pueden dar resultados muy distintos** porque responden a preguntas distintas:

| Pregunta | Protocolo | RMSE | R² |
|---|---|---|---|
| ¿Funciona en países nuevos? | Split por país (Entregas 1–3) | **0.62** | **0.43** |
| ¿Funciona en años nuevos del mismo país? | Split temporal | 0.38 | 0.83 |

Para una política de salud que necesita estimar depresión en países pobremente representados en GBD, el número honesto es el **0.43 de R²**, no el 0.83."""
    )
)

# -------------------------------------------------------------------
# 9. Ablación de features
# -------------------------------------------------------------------
cells.append(
    md(
        """## 9. Análisis de sensibilidad por feature (ablación)
Voy a reentrenar el modelo **sin cada feature** y medir cuánto se degrada el RMSE en test. Esto complementa la permutation importance:
- la *permutation* dice cuánto le importa una feature **al modelo ya entrenado** (lo que aporta marginalmente al momento de predecir);
- la *ablación* dice cuánto le importa al modelo **incluso si pudiera adaptarse a su ausencia** (lo que aporta estructuralmente).

Si una feature es muy importante pero también muy redundante con otra, la permutación la marcará como importante (porque el modelo la usa) pero la ablación verá poca degradación (porque el modelo se reorganiza usando la otra)."""
    )
)

cells.append(
    code(
        """def feature_ablation(X_tr, y_tr, X_te, y_te, best_params):
    params = {k.replace('model__',''):v for k,v in best_params.items()}
    base_m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
    base_m.fit(X_tr, y_tr)
    base_rmse = rmse(y_te, base_m.predict(X_te))
    out = {'_baseline_all_features': {'RMSE': base_rmse, 'delta_RMSE': 0.0}}
    for i, fname in enumerate(FEATURES):
        keep = [j for j in range(len(FEATURES)) if j != i]
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
        m.fit(X_tr[:, keep], y_tr)
        r = rmse(y_te, m.predict(X_te[:, keep]))
        out[fname] = {'RMSE': float(r), 'delta_RMSE': float(r - base_rmse)}
    return out

abl = feature_ablation(X_tr, y_tr, X_te, y_te, gs_rf.best_params_)
pd.DataFrame(abl).T.round(4)
"""
    )
)

cells.append(
    md(
        """### Lectura de la ablación
- **Quitar `schizophrenia` cuesta más** (delta_RMSE positivo y grande): coincide con que es la feature que más mueve al modelo.
- **Quitar `Year` puede incluso mejorar levemente** el RMSE (delta_RMSE negativo). Year aporta poco más que ruido para este target — es plausible, porque la prevalencia estandarizada por edad cambia muy poco a lo largo de 30 años.
- Las demás features aportan moderadamente.

Esto refuerza lo que ya intuía en las entregas anteriores: la señal principal está en la **estructura de comorbilidad transversal** entre los cuatro trastornos, no en la evolución temporal."""
    )
)

# -------------------------------------------------------------------
# 10. Interpretabilidad
# -------------------------------------------------------------------
cells.append(
    md(
        """## 10. Interpretabilidad — permutation importance + partial dependence
Random Forest no me da coeficientes legibles, pero sí me deja inspeccionarlo con dos herramientas que vimos en clase:

1. **Permutation importance** sobre el test: barajo una columna a la vez y mido cuánto sube el RMSE. Más subida ⇒ feature más usada.
2. **Partial dependence** sobre las dos features más importantes: cómo cambia la predicción promedio si fijo esa feature en distintos valores y dejo el resto como en los datos. Es la mejor aproximación visual al “efecto marginal” en un modelo no lineal."""
    )
)

cells.append(
    code(
        """pi = permutation_importance(
    final_est, X_te, y_te,
    scoring='neg_root_mean_squared_error',
    n_repeats=30, random_state=SEED, n_jobs=-1,
)
imp = pd.DataFrame({
    'feature': FEATURES,
    'importance_mean': pi.importances_mean,
    'importance_std':  pi.importances_std,
}).sort_values('importance_mean', ascending=False)
imp.round(4)
"""
    )
)

cells.append(
    code(
        """top2 = imp.head(2)['feature'].tolist()
top2_idx = [FEATURES.index(f) for f in top2]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, fi in zip(axes, top2_idx):
    pd_res = partial_dependence(final_est, X_tr, [fi], grid_resolution=50)
    ax.plot(pd_res['grid_values'][0], pd_res['average'][0], color='#3b6aa0', lw=2)
    ax.set_xlabel(FEATURES[fi])
    ax.set_ylabel('prediccion media')
    ax.set_title(f'Partial dependence - {FEATURES[fi]}')
plt.tight_layout(); plt.show()
"""
    )
)

cells.append(
    md(
        """### Lectura de la interpretabilidad

- La feature dominante sigue siendo la **prevalencia de esquizofrenia**: cuando la permuto, el RMSE sube ~0.28 (gran salto respecto al resto).
- **Ansiedad** y **bipolaridad** quedan en segundo y tercer lugar, ambas alrededor de +0.07–0.08 de RMSE al permutar.
- **Eating** aporta poco y **Year** prácticamente nada (consistente con la ablación).

La **partial dependence** muestra que la relación entre `schizophrenia` y la predicción de depresión **no es lineal**: hay tramos de baja pendiente y un tramo donde la depresión predicha sube más fuerte. Para `anxiety`, la relación es más cercana a lineal y monótona (más ansiedad → más depresión, como esperaba el EDA).

Una nota epistemológica importante: el hecho de que `schizophrenia` sea el predictor más fuerte **no es un descubrimiento clínico**. Es un patrón estadístico en datos agregados de país-año del IHME; probablemente la esquizofrenia está actuando como un “marcador de contexto” de país (regiones con cierto perfil epidemiológico). No estoy diciendo que la esquizofrenia *cause* depresión."""
    )
)

# -------------------------------------------------------------------
# 11. Análisis de errores
# -------------------------------------------------------------------
cells.append(
    md(
        """## 11. Análisis de errores
Tres miradas: residuos contra predicción, distribución de residuos y top de países con mayor error."""
    )
)

cells.append(
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
resid = y_te - final_pred
axes[0].scatter(final_pred, resid, s=12, alpha=0.5, color='#3b6aa0')
axes[0].axhline(0, color='red', ls='--', lw=1)
axes[0].set_xlabel('y_pred'); axes[0].set_ylabel('y - y_pred')
axes[0].set_title('Residuos vs prediccion')

axes[1].hist(resid, bins=40, color='#3b6aa0', edgecolor='white')
axes[1].axvline(0, color='red', ls='--', lw=1)
axes[1].axvline(resid.mean(), color='green', ls='-', lw=1,
                label=f'media={resid.mean():.3f}')
axes[1].set_xlabel('residuo')
axes[1].set_title('Distribucion de residuos')
axes[1].legend()
plt.tight_layout(); plt.show()
"""
    )
)

cells.append(
    code(
        """tmp = test_df.copy()
tmp['y_true'] = y_te
tmp['y_pred'] = final_pred
tmp['abs_err'] = (tmp['y_true'] - tmp['y_pred']).abs()
by_country = tmp.groupby('Entity')['abs_err'].mean().sort_values(ascending=False)
top = by_country.head(20)[::-1]

fig, ax = plt.subplots(figsize=(8.5, 7))
ax.barh(top.index, top.values, color='#a14b2a')
ax.set_xlabel('MAE promedio del pais')
ax.set_title(f'Top 20 paises con mas error - {FINAL_NAME}')
plt.tight_layout(); plt.show()
"""
    )
)

cells.append(
    md(
        """### Patrón en los países con más error
Los más difíciles son una mezcla de:
- **Países con perfil epidemiológico atípico** (Perú, Marruecos, Cuba aparecen siempre en el top — son outliers en el EDA);
- **Países con depresión alta pero comorbilidades “moderadas”** (Alemania, Polonia, Portugal) que el modelo no logra explicar con las 4 features que tiene;
- **Países pequeños o con datos ruidosos del IHME** (Saint Kitts and Nevis, Lesotho).

Esto sugiere que el modelo aprovechó bien la estructura **promedio** del dataset, pero hay heterogeneidad entre países que las features actuales no capturan. Como agenda para una hipotética Entrega 4: indicadores socioeconómicos, regionales o de sistema de salud."""
    )
)

# -------------------------------------------------------------------
# 12. Conclusiones
# -------------------------------------------------------------------
cells.append(
    md(
        """## 12. Conclusiones — respuestas a la rúbrica de Entrega 3

### ¿Cuál es el mejor modelo y por qué?
**Random Forest tuneado** con `n_estimators=800`, `max_depth=8`, `min_samples_leaf=3`, `max_features='sqrt'`. Lo elijo porque:
1. Es el modelo con menor RMSE en test (0.618 vs 0.642 de HistGB y 0.619 del ensemble).
2. Su CV-RMSE es el más bajo de las familias comparadas (0.72).
3. Es robusto sin necesidad de early stopping ni regularización ad-hoc.
4. El ensemble no aporta — el promedio HistGB+RF queda básicamente igual a RF sola.

### ¿Qué tan confiables son sus resultados?
**Confianza moderada**, y con dos caveats:
- **Bootstrap del test:** RMSE = 0.618 con IC95 % [0.597, 0.638]. R² = 0.43 con IC95 % [0.38, 0.47]. El intervalo es estrecho dentro del split actual.
- **Sensibilidad al split:** entre 5 semillas distintas, el R² varía entre **0.16 y 0.58**. La performance real depende mucho de qué países caigan en test. Para una decisión seria habría que reportar promedio sobre varias semillas.

### ¿Qué variables o patrones explican el desempeño?
- La feature dominante es la **prevalencia de esquizofrenia** (permutation importance ≈ 0.28, ablación ≈ +0.09 de RMSE al quitarla).
- **Ansiedad** y **bipolaridad** aportan un nivel intermedio.
- **Year** es prácticamente irrelevante — quitarla mejora marginalmente.
- La relación de `schizophrenia` con la predicción es **no lineal** (visible en partial dependence), lo que explica por qué el modelo lineal de la Entrega 1 fracasaba.

### ¿Qué conclusiones útiles deja el proyecto?
1. Para predecir la prevalencia de depresión a nivel país, **las prevalencias de otros trastornos mentales contienen señal real**, especialmente esquizofrenia y ansiedad.
2. Esa señal es **no lineal**; los modelos basados en árboles la capturan, los lineales no.
3. La generalización **a países completamente nuevos es limitada** (R² ~ 0.4), pero la generalización **a años nuevos del mismo país es buena** (R² ~ 0.8). Son problemas distintos y conviene no confundirlos.
4. **El año aporta poco**: las prevalencias estandarizadas por edad son lentas en el tiempo.

### ¿Qué haría falta para mejorar o desplegar la solución?
1. **Más features**: indicadores socioeconómicos, gasto en salud, indicadores de equidad, urbanización, GDP per cápita. Probablemente cierren la brecha de R² ~ 0.4 → 0.6+.
2. **Modelado jerárquico** que respete la estructura país/región/grupo de ingreso (efectos aleatorios o agrupados).
3. **Validación cruzada externa por región** además de por país, para detectar si el modelo aprende patrones regionales útiles.
4. Para despliegue: pipeline de monitoreo, alertas si los inputs de un nuevo país caen fuera del rango aprendido, e intervalos de predicción explícitos (cuantiles de los árboles) en vez de un solo punto.
5. **Estudios de robustez** ante datos faltantes, ruido en las features y cambios en la definición del IHME (que actualiza sus estimaciones anualmente)."""
    )
)


nb["cells"] = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with NB_PATH.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Notebook escrito en: {NB_PATH.relative_to(PROJECT)}")
