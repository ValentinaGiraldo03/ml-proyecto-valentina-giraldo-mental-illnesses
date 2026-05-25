"""Genera el poster PDF de la Entrega 3.

Poster A1 vertical (841 x 594 mm) compuesto con matplotlib:
  - Cabecera con titulo y autor.
  - Problema y datos (texto).
  - Metodologia y modelo final (texto).
  - 3 figuras clave (comparacion final, permutation importance, top errores).
  - Resultado clave en grande.
  - Limitaciones y conclusiones.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
METRICS = PROJECT / "report" / "entrega3_metrics.json"
FIG = PROJECT / "figures"
OUT = PROJECT / "poster" / "entrega3_poster.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


def fmt(x, d=3):
    return f"{x:.{d}f}"


def add_box(ax, title, text, fontsize=11):
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#f4f7fb",
            edgecolor="#2a5a8a",
            lw=1.2,
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.03,
        0.95,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize + 4,
        fontweight="bold",
        color="#1f3a5f",
    )
    ax.text(
        0.03,
        0.85,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="#1d1d1d",
        wrap=True,
    )


def add_image_box(ax, title, image_path, fontsize=12):
    ax.axis("off")
    img = mpimg.imread(image_path)
    ax.imshow(img)
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", color="#1f3a5f", loc="left")


def main():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    fm = m["final_test_metrics"]
    ci = m["bootstrap_ci"]

    # A1 vertical en pulgadas: 23.4 x 33.1
    fig = plt.figure(figsize=(23.4, 33.1))
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        nrows=10,
        ncols=2,
        figure=fig,
        height_ratios=[0.6, 0.45, 1.2, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6],
        hspace=0.45,
        wspace=0.12,
        left=0.04,
        right=0.96,
        top=0.97,
        bottom=0.03,
    )

    # ---------- Header ----------
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    ax_head.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#1f3a5f",
            transform=ax_head.transAxes,
        )
    )
    ax_head.text(
        0.5,
        0.72,
        "Predicción de la prevalencia de depresión a nivel país-año",
        transform=ax_head.transAxes,
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        color="white",
    )
    ax_head.text(
        0.5,
        0.35,
        "a partir de la prevalencia de otros trastornos mentales (IHME · GBD)",
        transform=ax_head.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        color="#dde6f0",
        style="italic",
    )
    ax_head.text(
        0.5,
        0.10,
        "Valentina Giraldo · Aprendizaje de Máquina Aplicado · EAFIT · 2026",
        transform=ax_head.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="#b3c9e0",
    )

    # ---------- Big result row ----------
    ax_res = fig.add_subplot(gs[1, :])
    ax_res.axis("off")
    ax_res.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#e8eef7",
            edgecolor="#2a5a8a",
            lw=1.5,
            transform=ax_res.transAxes,
        )
    )
    res_text = (
        f"Modelo final: {m['final_model']}    |    "
        f"RMSE = {fmt(fm['RMSE'])} "
        f"[IC95% {fmt(ci['RMSE']['lo'])}–{fmt(ci['RMSE']['hi'])}]    |    "
        f"MAE = {fmt(fm['MAE'])}    |    "
        f"R² = {fmt(fm['R2'])} "
        f"[IC95% {fmt(ci['R2']['lo'])}–{fmt(ci['R2']['hi'])}]"
    )
    ax_res.text(
        0.5,
        0.65,
        res_text,
        transform=ax_res.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#1f3a5f",
    )
    ax_res.text(
        0.5,
        0.25,
        f"sobre {m['n_countries_test']} países nunca vistos durante el entrenamiento — "
        f"split por país, GroupShuffleSplit 80/20, SEED={m['seed']}",
        transform=ax_res.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        color="#3b6aa0",
        style="italic",
    )

    # ---------- Row: Problema + Método ----------
    ax_prob = fig.add_subplot(gs[2, 0])
    add_box(
        ax_prob,
        "1. Problema y datos",
        "Tarea de regresión supervisada: estimar la prevalencia de "
        "depresión (share de población, estandarizado por edad) por país y "
        "año, usando como features la prevalencia de esquizofrenia, ansiedad, "
        "trastorno bipolar, trastornos alimentarios y el año.\n\n"
        f"Dataset: Mental Illnesses Prevalence (IHME · GBD, vía Our World in "
        f"Data). {m['n_train']+m['n_test']} filas, "
        f"{m['n_countries_train']+m['n_countries_test']} países, 1990–2019.\n\n"
        "Se excluyen 9 agregados regionales del CSV original (Africa, Europe, "
        "EU27, grupos de ingreso) que no son países.",
        fontsize=13,
    )

    ax_met = fig.add_subplot(gs[2, 1])
    add_box(
        ax_met,
        "2. Metodología (CRISP-DM)",
        "• Marco: CRISP-DM en 3 entregas acumulativas (problema/baseline → "
        "comparación de familias → modelo final).\n\n"
        "• Split externo: GroupShuffleSplit 80/20 por Entity (164 / 41 países).\n\n"
        "• Tuning: GridSearchCV con GroupKFold(5) por país sobre el train. "
        "Pipeline con StandardScaler para los modelos lineales.\n\n"
        "• Métrica primaria: RMSE en puntos porcentuales del target. "
        "Complementarias: MAE y R².\n\n"
        "• Validación complementaria: bootstrap (n=2000) sobre el test para IC95% "
        "y repetición con 5 semillas del split externo para medir robustez.",
        fontsize=13,
    )

    # ---------- Row: Comparación final + Permutation ----------
    ax_cmp = fig.add_subplot(gs[3, 0])
    add_image_box(
        ax_cmp,
        "3. Resultado en test (41 países no vistos)",
        FIG / "14_final_comparison.png",
        fontsize=12,
    )

    ax_pi = fig.add_subplot(gs[3, 1])
    add_image_box(
        ax_pi,
        "4. ¿Qué features explican la predicción?",
        FIG / "23_permutation_final.png",
        fontsize=12,
    )

    # ---------- Row: Bootstrap CI + Seed sensitivity ----------
    ax_ci = fig.add_subplot(gs[4, 0])
    add_image_box(
        ax_ci,
        "5. Confianza del resultado (bootstrap)",
        FIG / "15_bootstrap_ci.png",
        fontsize=12,
    )

    ax_seed = fig.add_subplot(gs[4, 1])
    add_image_box(
        ax_seed,
        "6. Sensibilidad al split de países (5 semillas)",
        FIG / "16_seed_sensitivity.png",
        fontsize=12,
    )

    # ---------- Row: PvT + Top errors ----------
    ax_pvt = fig.add_subplot(gs[5, 0])
    add_image_box(
        ax_pvt,
        "7. y real vs y predicho",
        FIG / "22_pvt_final.png",
        fontsize=12,
    )

    ax_err = fig.add_subplot(gs[5, 1])
    add_image_box(
        ax_err,
        "8. Países con mayor error en test",
        FIG / "20_top_errors_final.png",
        fontsize=12,
    )

    # ---------- Row: Partial dep + Temporal ----------
    ax_pd = fig.add_subplot(gs[6, 0])
    add_image_box(
        ax_pd,
        "9. Partial dependence — relación no lineal",
        FIG / "18_partial_dependence.png",
        fontsize=12,
    )

    ax_tmp = fig.add_subplot(gs[6, 1])
    add_image_box(
        ax_tmp,
        "10. Dos protocolos de validación dan respuestas distintas",
        FIG / "19_temporal_vs_country.png",
        fontsize=12,
    )

    # ---------- Row: Hallazgos ----------
    ax_find = fig.add_subplot(gs[7, 0])
    add_box(
        ax_find,
        "11. Hallazgos clave",
        "• Las prevalencias de los otros trastornos mentales SÍ contienen señal "
        "predictiva sobre la depresión, pero la relación es NO LINEAL.\n\n"
        "• La feature dominante es la prevalencia de esquizofrenia, seguida de "
        "ansiedad y bipolaridad. El año aporta casi nada.\n\n"
        "• Generalizar a países nuevos (R² ≈ 0.43) es más difícil que generalizar "
        "a años nuevos del mismo país (R² ≈ 0.83). Son problemas distintos.\n\n"
        "• El modelo Random Forest con grid amplio supera ligeramente a HistGB en "
        "este split. El ensemble no aporta sobre RF sola.",
        fontsize=12,
    )

    ax_lim = fig.add_subplot(gs[7, 1])
    add_box(
        ax_lim,
        "12. Limitaciones explícitas",
        "• ALTA sensibilidad al split externo: R² entre 0.16 y 0.58 según "
        "la semilla. Para uso real, reportar el rango sobre varias semillas.\n\n"
        "• El target son estimaciones del IHME, no observaciones directas: el "
        "modelo aprende patrones de esa fuente.\n\n"
        "• NO es inferencia causal. Que la esquizofrenia sea la feature más útil "
        "no implica que cause depresión.\n\n"
        "• Núcleo persistente de países con error alto (Perú, Marruecos, Cuba, "
        "Alemania, Polonia). Las 5 features actuales no capturan suficiente "
        "contexto regional/socioeconómico.",
        fontsize=12,
    )

    # ---------- Row: Conclusiones y recomendaciones ----------
    ax_concl = fig.add_subplot(gs[8, :])
    add_box(
        ax_concl,
        "13. Conclusiones y recomendaciones",
        "El proyecto muestra que un Random Forest tuneado puede capturar la "
        "estructura no lineal de comorbilidad entre trastornos mentales a nivel "
        "país-año y mejorar sustancialmente al baseline lineal, alcanzando "
        f"R² = {fmt(fm['R2'])} en {m['n_countries_test']} países nunca vistos. "
        "Para uso académico o exploratorio el modelo es utilizable como referencia "
        "comparativa entre perfiles de comorbilidad. Antes de pensar en "
        "despliegue, las acciones prioritarias son: (1) agregar features "
        "socioeconómicas (PIB per cápita, gasto en salud, urbanización) que "
        "probablemente cierren la brecha hasta R² ≈ 0.6+; (2) reportar siempre el "
        "rango entre varias semillas y no un punto único; (3) validar fuera del "
        "IHME para descartar dependencia de la fuente. El modelo NO debe usarse "
        "para decisiones individuales: opera a nivel país-año.",
        fontsize=13,
    )

    # ---------- Footer ----------
    ax_foot = fig.add_subplot(gs[9, :])
    ax_foot.axis("off")
    ax_foot.text(
        0.5,
        0.55,
        "Reproducible end-to-end: SEED=42 · scikit-learn · "
        "python src/build_entrega3.py + build_notebook_entrega3.py + build_pdf_entrega3.py",
        transform=ax_foot.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#2a5a8a",
        style="italic",
    )
    ax_foot.text(
        0.5,
        0.20,
        "Repo: ml-proyecto-valentina-giraldo-mental-illnesses · "
        "Dataset: Our World in Data / IHME GBD",
        transform=ax_foot.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="grey",
    )

    fig.savefig(OUT, dpi=150, bbox_inches=None)
    plt.close(fig)
    print(f"Poster generado en {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
