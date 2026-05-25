"""Genera el deck de presentacion final (PDF de ~8 slides) de la Entrega 3.

Cada slide es una pagina apaisada con titulo + contenido visual y/o
viñetas. Se pensó como apoyo a la presentación oral.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
METRICS = PROJECT / "report" / "entrega3_metrics.json"
FIG = PROJECT / "figures"
OUT = PROJECT / "report" / "entrega3_presentation.pdf"


def fmt(x, d=3):
    return f"{x:.{d}f}"


def new_slide():
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    return fig


def add_header(fig, title, subtitle=None):
    ax = fig.add_axes([0, 0.88, 1, 0.12])
    ax.axis("off")
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#1f3a5f",
            transform=ax.transAxes,
        )
    )
    ax.text(
        0.03,
        0.55,
        title,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color="white",
        va="center",
    )
    if subtitle:
        ax.text(
            0.03,
            0.20,
            subtitle,
            transform=ax.transAxes,
            fontsize=12,
            color="#dde6f0",
            va="center",
            style="italic",
        )


def add_footer(fig, page, total):
    ax = fig.add_axes([0, 0, 1, 0.04])
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"Valentina Giraldo · Aprendizaje de Máquina Aplicado · EAFIT · {page}/{total}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="grey",
    )


def text_block(fig, x, y, w, h, lines, fontsize=14, color="#1d1d1d"):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    txt = "\n".join(lines) if isinstance(lines, list) else lines
    ax.text(
        0.02,
        0.96,
        txt,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        va="top",
        ha="left",
    )


def image_block(fig, x, y, w, h, image_path):
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    img = mpimg.imread(image_path)
    ax.imshow(img)


def main():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    fm = m["final_test_metrics"]
    ci = m["bootstrap_ci"]
    total = 8

    with PdfPages(OUT) as pdf:
        # ---------- Slide 1: Title ----------
        fig = new_slide()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="#1f3a5f",
                transform=ax.transAxes,
            )
        )
        ax.text(
            0.5,
            0.62,
            "Predicción de la prevalencia de depresión",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=34,
            fontweight="bold",
            color="white",
        )
        ax.text(
            0.5,
            0.52,
            "a nivel país-año (IHME · GBD)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=22,
            color="#b3c9e0",
            style="italic",
        )
        ax.text(
            0.5,
            0.35,
            "Entrega 3 — Modelo final, interpretación y comunicación",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color="white",
        )
        ax.text(
            0.5,
            0.22,
            "Valentina Giraldo",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color="white",
        )
        ax.text(
            0.5,
            0.15,
            "Aprendizaje de Máquina Aplicado · EAFIT · 2026",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#b3c9e0",
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 2: Problema y datos ----------
        fig = new_slide()
        add_header(fig, "1. Problema y datos", "Tarea: regresión supervisada a nivel país-año")
        text_block(
            fig,
            0.05,
            0.10,
            0.42,
            0.75,
            [
                "Predecir la prevalencia de depresión",
                "(share de población, estandarizado por edad)",
                "",
                "Features:",
                "   • schizophrenia",
                "   • anxiety",
                "   • bipolar",
                "   • eating",
                "   • Year",
                "",
                "Dataset: Mental Illnesses Prevalence",
                "(IHME · Global Burden of Disease,",
                "vía Our World in Data)",
                "",
                f"{m['n_train']+m['n_test']} filas",
                f"{m['n_countries_train']+m['n_countries_test']} países",
                "1990–2019",
            ],
            fontsize=15,
        )
        image_block(fig, 0.52, 0.10, 0.45, 0.75, FIG / "02_correlaciones.png")
        add_footer(fig, 2, total)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 3: Metodología ----------
        fig = new_slide()
        add_header(fig, "2. Metodología", "CRISP-DM, sin fuga entre países")
        text_block(
            fig,
            0.05,
            0.10,
            0.90,
            0.75,
            [
                "• Split externo: GroupShuffleSplit 80/20 por país (SEED=42)",
                "    → 164 países train · 41 países test (nunca vistos)",
                "",
                "• Tuning interno: GridSearchCV con GroupKFold(5) por país",
                "    → cero leakage entre folds; StandardScaler dentro del Pipeline",
                "",
                "• Familias re-evaluadas en Entrega 3:",
                "    – HistGradientBoosting (grid amplio + early stopping)",
                "    – Random Forest (grid amplio)",
                "    – Ensemble por promedio HistGB + RF",
                "",
                "• Validación complementaria:",
                "    – Bootstrap (n=2000) sobre el test → intervalos de confianza",
                "    – 5 semillas distintas del split externo → sensibilidad",
                "    – Split temporal estricto (train ≤2009 / test 2010–2019)",
                "",
                "• Métrica primaria: RMSE en puntos porcentuales del target",
                "    – Complementarias: MAE (robusta) y R² (intuitiva)",
            ],
            fontsize=15,
        )
        add_footer(fig, 3, total)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 4: Comparación + decisión ----------
        fig = new_slide()
        add_header(
            fig,
            "3. Comparación en test y decisión del modelo final",
            "41 países que el modelo no vio nunca",
        )
        image_block(fig, 0.05, 0.10, 0.55, 0.75, FIG / "14_final_comparison.png")
        text_block(
            fig,
            0.62,
            0.15,
            0.35,
            0.70,
            [
                f"Modelo final:",
                f"  {m['final_model']}",
                "",
                f"RMSE = {fmt(fm['RMSE'])}",
                f"MAE  = {fmt(fm['MAE'])}",
                f"R²   = {fmt(fm['R2'])}",
                "",
                "El ensemble no aporta sobre",
                "RF sola. HistGB queda en 2º.",
                "",
                "Mejora claramente al dummy",
                "y al LinReg de la Entrega 1",
                "(R² < 0 con el mismo split).",
            ],
            fontsize=15,
        )
        add_footer(fig, 4, total)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 5: Confianza y sensibilidad ----------
        fig = new_slide()
        add_header(
            fig,
            "4. ¿Qué tan confiable es el resultado?",
            "Bootstrap (n=2000) + sensibilidad a la semilla del split",
        )
        image_block(fig, 0.03, 0.10, 0.45, 0.75, FIG / "15_bootstrap_ci.png")
        image_block(fig, 0.52, 0.10, 0.45, 0.75, FIG / "16_seed_sensitivity.png")
        ax = fig.add_axes([0, 0.05, 1, 0.06])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Bootstrap: estrecho (IC95% RMSE ≈ [0.60, 0.64]).   "
            "Pero entre semillas: R² varía entre 0.16 y 0.58 — esta es la limitación principal.",
            transform=ax.transAxes,
            ha="center",
            fontsize=13,
            color="#1f3a5f",
            fontweight="bold",
        )
        add_footer(fig, 5, total)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 6: Interpretabilidad ----------
        fig = new_slide()
        add_header(
            fig,
            "5. ¿Qué features explican la predicción?",
            "Permutation importance + partial dependence",
        )
        image_block(fig, 0.03, 0.10, 0.45, 0.75, FIG / "23_permutation_final.png")
        image_block(fig, 0.52, 0.10, 0.45, 0.75, FIG / "18_partial_dependence.png")
        ax = fig.add_axes([0, 0.05, 1, 0.06])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Feature dominante: prevalencia de esquizofrenia.   Relación NO lineal.   "
            "El año aporta casi nada.",
            transform=ax.transAxes,
            ha="center",
            fontsize=13,
            color="#1f3a5f",
            fontweight="bold",
        )
        add_footer(fig, 6, total)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 7: Errores + temporal ----------
        fig = new_slide()
        add_header(
            fig,
            "6. Dónde falla el modelo + dos protocolos de validación",
            "El núcleo de error coincide con outliers del EDA",
        )
        image_block(fig, 0.03, 0.10, 0.50, 0.75, FIG / "20_top_errors_final.png")
        image_block(fig, 0.55, 0.10, 0.42, 0.75, FIG / "19_temporal_vs_country.png")
        ax = fig.add_axes([0, 0.05, 1, 0.06])
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Países pesados: Perú, Marruecos, Cuba, Alemania, Polonia.   "
            "Split temporal: R² 0.83 — más fácil porque interpola años, no extrapola países.",
            transform=ax.transAxes,
            ha="center",
            fontsize=12,
            color="#1f3a5f",
            fontweight="bold",
        )
        add_footer(fig, 7, total)
        pdf.savefig(fig)
        plt.close(fig)

        # ---------- Slide 8: Conclusiones ----------
        fig = new_slide()
        add_header(fig, "7. Conclusiones y recomendaciones", None)
        text_block(
            fig,
            0.05,
            0.05,
            0.45,
            0.80,
            [
                "Lo que funciona:",
                "",
                "• Los modelos basados en árboles sí",
                "  capturan la estructura no lineal de",
                "  comorbilidad entre trastornos.",
                "",
                f"• RF final: R² = {fmt(fm['R2'])} en países nuevos.",
                "  Salto claro sobre el dummy y el",
                "  baseline lineal de Entrega 1.",
                "",
                "• La feature dominante es la prevalencia",
                "  de esquizofrenia (no causal, contextual).",
                "",
                "• El año aporta casi nada.",
            ],
            fontsize=14,
        )
        text_block(
            fig,
            0.52,
            0.05,
            0.45,
            0.80,
            [
                "Limitaciones honestas:",
                "",
                "• R² varía entre 0.16 y 0.58 según la",
                "  semilla. Reportar el rango, no el punto.",
                "",
                "• Target son estimaciones del IHME,",
                "  no observaciones directas.",
                "",
                "• No es inferencia causal.",
                "",
                "Próximos pasos:",
                "",
                "• Agregar features socioeconómicas",
                "  → probablemente R² ~ 0.6+.",
                "• Validación cruzada por región.",
                "• Intervalos de predicción explícitos.",
            ],
            fontsize=14,
        )
        add_footer(fig, 8, total)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Presentacion generada en {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
