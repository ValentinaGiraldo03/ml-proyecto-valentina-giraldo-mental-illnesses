# Reproducibility checklist — Entrega 3

Este checklist documenta los puntos que un tercero debería poder verificar para reproducir el proyecto end-to-end. Se sigue el espíritu de los *reproducibility checklists* que se usan en conferencias de ML (NeurIPS, ICML), adaptado al alcance del curso.

## 1. Datos

- [x] **Fuente única y verificable.** El CSV `data/mental-illnesses-prevalence.csv` proviene de Our World in Data (compilación del IHME · Global Burden of Disease). No fue modificado.
- [x] **Conteo de filas y columnas documentado.** 6 420 filas × 8 columnas en el CSV original. Tras excluir los 9 agregados regionales (Africa, America, Europe, EU27, 4 grupos de ingreso por nivel del Banco Mundial) quedan **6 150 filas / 205 países / 30 años (1990–2019)**.
- [x] **Esquema de columnas explícito.** El renombrado de columnas largas a nombres cortos (`schizophrenia`, `depression`, `anxiety`, `bipolar`, `eating`) está centralizado en el diccionario `COL_MAP` y es idéntico en los scripts de las 3 entregas.
- [x] **Exclusiones justificadas.** Se excluyen las 270 filas con `Code` vacío porque corresponden a agregados regionales, no a países. Documentado en Entrega 1.
- [x] **Sin overlap entre train y test.** Los 164 países de train y los 41 de test son disjuntos por construcción (verificado en notebook con `set(g_tr) & set(g_te)`).

## 2. Código

- [x] **Semilla fija.** `SEED = 42` en todos los scripts (`build_entrega2.py`, `build_entrega3.py`, etc.).
- [x] **Sin descargas en runtime.** El CSV está versionado en el repositorio (`data/`). Los scripts no descargan nada.
- [x] **Sin estado oculto.** Los scripts se ejecutan limpios desde cero; toda la salida es función de `(CSV, SEED, parámetros del grid)`.
- [x] **Notebooks reproducibles.** Cada notebook usa rutas relativas (`../data`, `../figures`, `../report`) y la misma semilla.
- [x] **Funciones reutilizables.** `rmse`, `metric_dict`, `bootstrap_ci`, `sensitivity_seed`, `feature_ablation`, `temporal_split` están encapsuladas y testeables.

## 3. Entorno

- [x] **`requirements.txt` versionado** con cotas mínimas: `pandas>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`, `seaborn>=0.12`, `scikit-learn>=1.3`, `nbformat>=5.0`, `nbclient>=0.9`, `reportlab>=4.0`.
- [x] **Sin dependencias opcionales.** No se usan XGBoost, LightGBM, SHAP, ni nada fuera de la biblioteca estándar de scikit-learn. La elección está alineada con el contenido visto en clase (slides 01–05).
- [x] **Python 3.10+ recomendado.** Las pruebas se hicieron en Python 3.14 sobre Windows 11.

## 4. Protocolo experimental

- [x] **Split externo determinístico:** `GroupShuffleSplit(test_size=0.2, random_state=42)` por `Entity`. Heredado de Entrega 1.
- [x] **Tuning sin leakage:** `GridSearchCV` con `GroupKFold(5)` por país. `StandardScaler` (para modelos lineales) dentro del `Pipeline` para que cada fold lo ajuste solo con su train.
- [x] **Test mirado una sola vez** para cada candidato final.
- [x] **Métricas declaradas a priori:** RMSE (principal), MAE, R². RMSE elegido en Entrega 1 y mantenido a lo largo del proyecto.
- [x] **Intervalos de confianza** por bootstrap (n=2000) sobre el test.
- [x] **Sensibilidad documentada** sobre 5 semillas distintas del split externo.

## 5. Resultados

- [x] **Single source of truth.** Todas las cifras del reporte PDF, póster, resumen ejecutivo y notebook se leen de `report/entrega3_metrics.json`. No hay números hardcodeados en los reportes.
- [x] **Figuras reproducibles.** Cada figura PNG en `figures/` se regenera desde `build_entrega3.py`.
- [x] **Tabla de hiperparámetros ganadores** en el JSON (`tuning.HistGB.best_params`, `tuning.RF.best_params`).
- [x] **Resultados intermedios guardados:** CV-RMSE de cada familia, métricas en test de cada candidato, IC bootstrap, sensibilidad por semilla, ablación por feature, permutation importance, top-20 países con mayor error.

## 6. Limitaciones reportadas

- [x] **Sensibilidad al split** documentada como limitación principal (no como hallazgo).
- [x] **Diferencia entre split por país y temporal** explicada (no se promete que R²=0.83 sea generalizable a países nuevos).
- [x] **No-causalidad** declarada explícitamente.
- [x] **Limitaciones del target (IHME)** declaradas en el reporte y en el resumen ejecutivo.

## 7. Comunicación

- [x] **Reporte PDF completo** (`report/entrega3_reporte.pdf`).
- [x] **Resumen ejecutivo** (`report/entrega3_executive_summary.pdf`).
- [x] **Póster** (`poster/entrega3_poster.pdf`).
- [x] **Notebook final** con narrativa en primera persona y código ejecutable (`notebooks/03_entrega3_modelo_final.ipynb`).
- [x] **README** del repositorio con instrucciones de reproducción paso a paso.

## 8. Cómo verificar la reproducción

```bash
cd project
pip install -r requirements.txt
python src/build_entrega3.py             # genera figuras 14-23 + entrega3_metrics.json
python src/build_notebook_entrega3.py    # genera notebooks/03_*.ipynb
python src/build_pdf_entrega3.py         # genera report/entrega3_reporte.pdf
python src/build_executive_summary.py    # genera report/entrega3_executive_summary.pdf
python src/build_poster.py               # genera poster/entrega3_poster.pdf
```

Los valores numéricos clave que un tercero debería obtener (sobre Python 3.10+ y scikit-learn 1.3+):

| Métrica | Valor esperado |
|---|---|
| Modelo final | Random Forest (tuneado) |
| Test RMSE (SEED=42) | ≈ 0.618 |
| Test MAE (SEED=42) | ≈ 0.517 |
| Test R² (SEED=42) | ≈ 0.426 |
| Bootstrap IC95 % R² | aprox. [0.38, 0.47] |
| Sensibilidad R² (5 seeds) | rango aprox. 0.16–0.58 |

Pequeñas diferencias numéricas (en el orden del 0.001) son aceptables y suelen deberse a cambios menores entre versiones de scikit-learn en cómo se inicializa el bosque. Los rankings y conclusiones cualitativas no deberían cambiar.
