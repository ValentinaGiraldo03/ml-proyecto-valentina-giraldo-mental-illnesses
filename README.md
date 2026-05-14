# Proyecto aplicado — Aprendizaje de Máquina (EAFIT)

**Dataset:** *Mental Illnesses Prevalence* (IHME — Global Burden of Disease, publicado por Our World in Data).
**Marco metodológico:** CRISP-DM.
**Estado:** Entrega 1 ✅ · **Entrega 2 ✅** · Entrega 3 ⏳

---

## 1. ¿Qué hay en esta carpeta?

```
project/
├── README.md                                   ← este archivo
├── requirements.txt                            ← dependencias de Python
├── data/
│   └── mental-illnesses-prevalence.csv
├── notebooks/
│   ├── 01_entrega1_eda_baseline.ipynb          ← Entrega 1: EDA + baseline
│   └── 02_entrega2_modelos_validacion.ipynb    ← Entrega 2: comparación + validación
├── figures/
│   ├── 01_distribuciones.png                   (Entrega 1)
│   ├── 02_correlaciones.png
│   ├── 03_tendencia_global.png
│   ├── 04_top_bottom_depresion.png
│   ├── 05_scatter_ansiedad_depresion.png
│   ├── 06a_residuos_row.png   ·  06b_residuos_group.png
│   ├── 07a_pvt_row.png        ·  07b_pvt_group.png
│   ├── 08_cv_comparison.png                    (Entrega 2 — desde aquí)
│   ├── 09_test_metrics.png
│   ├── 10_residuos_familias.png
│   ├── 11_pvt_familias.png
│   ├── 12_permutation_importance.png
│   └── 13_top_errores.png
├── report/
│   ├── entrega1_reporte.pdf   ·  entrega1_metrics.json
│   └── entrega2_reporte.pdf   ·  entrega2_metrics.json
└── src/
    ├── build_entrega2.py                       ← corre todos los experimentos de la Entrega 2
    ├── build_notebook_entrega2.py              ← regenera el notebook 02_*.ipynb
    └── build_pdf_entrega2.py                   ← genera el PDF de la Entrega 2
```

## 2. ¿Cómo reproducir los resultados?

```bash
# (dentro de la carpeta project/)
pip install -r requirements.txt
```

### Entrega 1
Abrir `notebooks/01_entrega1_eda_baseline.ipynb` en VS Code y ejecutar todas las celdas. Lee `../data/mental-illnesses-prevalence.csv` con ruta relativa y deja figuras en `../figures/`.

### Entrega 2
Hay dos formas equivalentes:

1. **Vía notebook (recomendada para revisar el flujo):**
   abrir `notebooks/02_entrega2_modelos_validacion.ipynb` en VS Code y ejecutar todas las celdas. Re-corre el split por país, los `GridSearchCV` con `GroupKFold` por país, evalúa en test e imprime tablas + figuras.

2. **Vía scripts (reproducibilidad end-to-end):**
   ```bash
   # 1. Corre experimentos, guarda figuras y JSON
   python src/build_entrega2.py
   # 2. Regenera el notebook con la narrativa
   python src/build_notebook_entrega2.py
   # 3. Genera el reporte PDF
   python src/build_pdf_entrega2.py
   ```

La semilla está fija en `SEED = 42` en todos los scripts.

## 3. Decisiones que se mantienen entre entregas

- **Tarea:** regresión supervisada.
- **Target:** `depression` (share de población, estandarizado por edad).
- **Features:** `schizophrenia`, `anxiety`, `bipolar`, `eating`, `Year`.
- **Unidad de observación:** par (país, año). 6 150 filas tras excluir 9 agregados regionales.
- **Métrica principal:** RMSE (en puntos porcentuales del target). Complementarias: MAE y R².
- **Split honesto:** `GroupShuffleSplit` por `Entity` (80/20). El test (41 países) no se toca hasta el final.
- **Validación interna:** `GroupKFold(5)` por `Entity` para CV y tuning.

## 4. Qué se hizo en la Entrega 2 (resumen ejecutivo)

- Se compararon **tres familias de modelos**: ElasticNet (lineal regularizado), Random Forest y Histogram Gradient Boosting, contra dos baselines (Dummy y `LinReg` de la Entrega 1).
- Cada familia se ajustó con `GridSearchCV` usando `GroupKFold(5)` por país.
- El preprocesamiento (`StandardScaler` para los lineales) está dentro del `Pipeline` para evitar fuga entre folds.
- En CV las tres familias entrenadas mejoran al dummy; HistGB queda primero (RMSE ≈ 0.76).
- En **test** (41 países nunca vistos), la diferencia se acentúa: HistGB obtiene **RMSE ≈ 0.65 y R² ≈ 0.37**, RandomForest queda en R² ≈ 0.15 y las familias lineales caen por debajo del dummy.
- La feature más informativa según permutation importance es la **prevalencia de esquizofrenia**, seguida de la ansiedad. El año es prácticamente irrelevante.
- **Decisión provisional:** seguir con HistGB como candidato a modelo final. Pendientes para Entrega 3: features regionales, validación temporal y tuning más fino.

Detalles en `notebooks/02_entrega2_modelos_validacion.ipynb` y `report/entrega2_reporte.pdf`.
