# Proyecto aplicado. Aprendizaje de Máquina

**Dataset:** *Mental Illnesses Prevalence* (IHME — Global Burden of Disease, publicado por Our World in Data).
**Marco metodológico:** CRISP-DM.
**Estado:** Entrega 1 ✅ · Entrega 2 ✅ · **Entrega 3 ✅** (versión final).

Proyecto de regresión supervisada para estimar la **prevalencia de depresión**
a nivel país-año a partir de la prevalencia de otros trastornos mentales
(esquizofrenia, ansiedad, trastorno bipolar, trastornos alimentarios) y el año.

La salud mental es un componente esencial del bienestar humano: se estima que
1 de cada 3 mujeres y 1 de cada 5 hombres experimentarán depresión mayor a lo
largo de su vida. Entender cómo se relacionan los distintos trastornos puede
ayudar a anticipar la carga de enfermedad en países con menos datos disponibles
y orientar la planificación de recursos en salud pública.

---

## 1. ¿Qué hay en esta carpeta?

```
project/
├── README.md                                       ← este archivo
├── requirements.txt                                ← dependencias de Python
├── data/
│   └── mental-illnesses-prevalence.csv
├── notebooks/
│   ├── 01_entrega1_eda_baseline.ipynb              ← Entrega 1: EDA + baseline
│   ├── 02_entrega2_modelos_validacion.ipynb        ← Entrega 2: comparación + validación
│   └── 03_entrega3_modelo_final.ipynb              ← Entrega 3: modelo final + interpretación
├── figures/
│   ├── 01-07*.png                                  (Entrega 1)
│   ├── 08-13*.png                                  (Entrega 2)
│   └── 14-23*.png                                  (Entrega 3 — desde aquí)
├── report/
│   ├── entrega1_reporte.pdf   ·  entrega1_metrics.json
│   ├── entrega2_reporte.pdf   ·  entrega2_metrics.json
│   ├── entrega3_reporte.pdf              ← reporte final
│   ├── entrega3_executive_summary.pdf    ← resumen ejecutivo (~1 página)
│   ├── entrega3_presentation.pdf         ← deck de presentación (8 slides)
│   ├── entrega3_metrics.json             ← métricas finales (fuente única de la verdad)
│   └── reproducibility_checklist.md      ← checklist de reproducibilidad
├── poster/
│   └── entrega3_poster.pdf               ← póster síntesis (A1)
└── src/
    ├── build_entrega2.py            ·  build_notebook_entrega2.py   ·  build_pdf_entrega2.py
    ├── build_entrega3.py            ←   experimentos finales + figuras + JSON
    ├── build_notebook_entrega3.py   ←   regenera notebook 03_*.ipynb
    ├── build_pdf_entrega3.py        ←   regenera el reporte PDF final
    ├── build_executive_summary.py   ←   regenera el resumen ejecutivo
    ├── build_poster.py              ←   regenera el póster
    └── build_presentation.py        ←   regenera el deck de presentación
```

## 2. ¿Cómo reproducir los resultados?

```bash
# (dentro de la carpeta project/)
pip install -r requirements.txt
```

### Entrega 1
Abrir `notebooks/01_entrega1_eda_baseline.ipynb` y ejecutar todas las celdas.

### Entrega 2
```bash
python src/build_entrega2.py            # experimentos + figuras + JSON
python src/build_notebook_entrega2.py   # regenera el notebook
python src/build_pdf_entrega2.py        # regenera el PDF
```

### Entrega 3 (versión final)
```bash
python src/build_entrega3.py             # tuning final + bootstrap + sensibilidad + ablación
python src/build_notebook_entrega3.py    # regenera el notebook
python src/build_pdf_entrega3.py         # regenera el reporte PDF
python src/build_executive_summary.py    # regenera el resumen ejecutivo
python src/build_poster.py               # regenera el póster
python src/build_presentation.py         # regenera el deck de presentación
```

Semilla fija `SEED=42` en todos los scripts.

## 3. Decisiones que se mantienen entre entregas

- **Tarea:** regresión supervisada.
- **Target:** `depression` (share de población, estandarizado por edad).
- **Features:** `schizophrenia`, `anxiety`, `bipolar`, `eating`, `Year`.
- **Unidad de observación:** par (país, año). 6 150 filas tras excluir 9 agregados regionales.
- **Métrica principal:** RMSE (en puntos porcentuales del target). Complementarias: MAE y R².
- **Split honesto:** `GroupShuffleSplit` por `Entity` (80/20). El test (41 países) no se toca hasta el final.
- **Validación interna:** `GroupKFold(5)` por `Entity` para CV y tuning. Cero leakage entre folds.

## 4. Qué se hizo en la Entrega 3 (resumen ejecutivo)

- **Tuning final** con grid ampliado para HistGradientBoosting (con early stopping) y Random Forest, y un **ensemble** simple por promedio de los dos.
- **Modelo final elegido:** Random Forest tuneado (`n_estimators=800`, `max_depth=8`, `min_samples_leaf=3`, `max_features='sqrt'`). Es el de menor RMSE en test entre los candidatos finales.
- **Resultado en test (41 países no vistos):** RMSE ≈ **0.618** (IC95 % bootstrap [0.597, 0.638]); MAE ≈ **0.517**; R² ≈ **0.426** (IC95 % [0.380, 0.467]).
- **Sensibilidad al split**: probado con 5 semillas distintas, el R² varía entre 0.16 y 0.58. **Es la limitación más fuerte del proyecto** y está documentada explícitamente.
- **Validación temporal estricta** (train 1990–2009 / test 2010–2019): R² ≈ 0.83. Mejor número, pero menos honesto — el modelo interpola años de países que ya conoce.
- **Ablación de features** y **permutation importance** coinciden: la feature dominante es la prevalencia de esquizofrenia, seguida de ansiedad y bipolaridad. `Year` no aporta.
- **Partial dependence** muestra que la relación de `schizophrenia` con la predicción es no lineal — explica por qué la regresión lineal de la Entrega 1 fallaba.
- **Análisis de errores**: los países con mayor error coinciden con outliers epidemiológicos identificados en el EDA (Perú, Marruecos, Cuba), países europeos con depresión alta pero comorbilidades moderadas (Alemania, Polonia, Portugal) y países pequeños con estimaciones ruidosas en GBD (Saint Kitts and Nevis, Lesotho).

Detalle completo en `notebooks/03_entrega3_modelo_final.ipynb` y `report/entrega3_reporte.pdf`. Síntesis visual en `poster/entrega3_poster.pdf`. Lectura rápida en `report/entrega3_executive_summary.pdf`. Deck de presentación en `report/entrega3_presentation.pdf`.

## 5. Limitaciones documentadas

1. **Sensibilidad al split** (la limitación principal): el resultado depende del subconjunto de países que cae en test.
2. **Target derivado**: las prevalencias son estimaciones del IHME, no observaciones brutas.
3. **No es inferencia causal**: la importancia estadística de una feature no implica causalidad.
4. **Features limitadas**: solo 5 columnas; agregar contexto socioeconómico probablemente cierre la brecha de R².
5. **No transferibilidad fuera del IHME**: si otra fuente estimara las prevalencias con criterios distintos, el modelo necesitaría recalibración.
