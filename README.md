# Proyecto aplicado — Aprendizaje de Máquina (EAFIT)

**Dataset:** *Mental Illnesses Prevalence* (IHME — Global Burden of Disease, publicado por Our World in Data).
**Marco metodológico:** CRISP-DM.
**Entrega 1:** Problema, datos, EDA y baseline.

---

## 1. ¿Qué hay en esta carpeta?

```
project/
├── README.md                          ← este archivo (explicación de todo)
├── requirements.txt                   ← dependencias de Python
├── data/
│   └── mental-illnesses-prevalence.csv
├── notebooks/
│   └── 01_entrega1_eda_baseline.ipynb ← notebook de la Entrega 1 (se abre en VS Code)
├── figures/                           ← todas las figuras PNG
│   ├── 01_distribuciones.png
│   ├── 02_correlaciones.png
│   ├── 03_tendencia_global.png
│   ├── 04_top_bottom_depresion.png
│   ├── 05_scatter_ansiedad_depresion.png
│   ├── 06a_residuos_row.png     ← residuos con split aleatorio (con fuga)
│   ├── 06b_residuos_group.png   ← residuos con split por país (sin fuga)
│   ├── 07a_pvt_row.png          ← y vs ŷ con split aleatorio
│   └── 07b_pvt_group.png        ← y vs ŷ con split por país
└── report/
    ├── entrega1_reporte.pdf           ← reporte PDF de la Entrega 1
    └── entrega1_metrics.json          ← métricas numéricas exactas
```

## 2. ¿Cómo reproducir los resultados?

```bash
# (dentro de la carpeta project/)
pip install -r requirements.txt
```

Luego abrir `notebooks/01_entrega1_eda_baseline.ipynb` directamente en VS Code
(con la extensión de Python que soporta notebooks `.ipynb`) y ejecutar todas las celdas.

- La semilla está fija en `SEED = 42` dentro del notebook.
- El notebook lee `../data/mental-illnesses-prevalence.csv` (ruta relativa).
- Las figuras y el reporte PDF ya están generados en `figures/` y `report/`.

La depresión es el trastorno mental más prevalente; conocer la estructura de
  comorbilidad a nivel poblacional es relevante para políticas públicas, y proporciona una
  referencia honesta de dificultad antes de usar modelos más complejos.
