# Proyecto aplicado — Aprendizaje de Máquina (EAFIT)

**Dataset:** *Mental Illnesses Prevalence* (IHME — Global Burden of Disease, publicado por Our World in Data).
**Marco metodológico:** CRISP-DM.
**Estado:** Entrega 1 · **Entrega 2* 

---
## 1. ¿Cómo reproducir los resultados?

```bash
# (dentro de la carpeta project/)
pip install -r requirements.txt
```

### Entrega 1
Abrir `notebooks/01_entrega1_eda_baseline.ipynb` en VS Code y ejecutar todas las celdas. Lee `../data/mental-illnesses-prevalence.csv` con ruta relativa y deja figuras en `../figures/`.

### Entrega 2
Hay dos formas equivalentes:

1. **Vía notebook:**
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
