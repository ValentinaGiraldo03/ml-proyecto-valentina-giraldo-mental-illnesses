"""Genera el resumen ejecutivo PDF de la Entrega 3.

Documento corto (~1-2 paginas) pensado para lectura rapida por un
stakeholder que no necesita el detalle tecnico.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
METRICS = PROJECT / "report" / "entrega3_metrics.json"
FIG = PROJECT / "figures"
OUT = PROJECT / "report" / "entrega3_executive_summary.pdf"


def fmt(x, d=3):
    return f"{x:.{d}f}"


def build():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    base = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontSize=17,
        spaceAfter=8,
        textColor=colors.HexColor("#1f3a5f"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontSize=12,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#2a5a8a"),
    )
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=4,
        alignment=4,
    )
    small = ParagraphStyle(
        "Small",
        parent=base["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.grey,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=12,
        bulletIndent=2,
        spaceAfter=2,
    )

    story = []
    fm = m["final_test_metrics"]
    ci = m["bootstrap_ci"]

    # ---------- Title ----------
    story.append(Paragraph("Resumen ejecutivo", h1))
    story.append(
        Paragraph(
            "Predicción de la prevalencia de depresión a nivel país-año a partir "
            "de la prevalencia de otros trastornos mentales (IHME · GBD)",
            small,
        )
    )
    story.append(
        Paragraph(
            "Proyecto aplicado · Aprendizaje de Máquina Aplicado · EAFIT · "
            "Valentina Giraldo · 2026-05-14",
            small,
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    # ---------- Problema ----------
    story.append(Paragraph("Problema", h2))
    story.append(
        Paragraph(
            "Estimar la prevalencia de depresión (% de la población, estandarizada "
            "por edad) a nivel país-año, con información de la prevalencia de "
            "esquizofrenia, ansiedad, trastorno bipolar y trastornos alimentarios. "
            "Datos: <i>Mental Illnesses Prevalence</i> (IHME · GBD), "
            f"{m['n_train']+m['n_test']} filas, "
            f"{m['n_countries_train']+m['n_countries_test']} países, 1990-2019.",
            body,
        )
    )

    # ---------- Resultado clave ----------
    story.append(Paragraph("Resultado clave", h2))
    tbl = [
        ["Métrica", "Valor", "IC95% (bootstrap, n=2000)"],
        [
            "RMSE",
            fmt(fm["RMSE"]),
            f"[{fmt(ci['RMSE']['lo'])}, {fmt(ci['RMSE']['hi'])}]",
        ],
        [
            "MAE",
            fmt(fm["MAE"]),
            f"[{fmt(ci['MAE']['lo'])}, {fmt(ci['MAE']['hi'])}]",
        ],
        [
            "R²",
            fmt(fm["R2"]),
            f"[{fmt(ci['R2']['lo'])}, {fmt(ci['R2']['hi'])}]",
        ],
    ]
    t = Table(tbl, colWidths=[2.5 * cm, 3 * cm, 7 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            f"<b>Modelo final:</b> {m['final_model']} (n_estimators=800, "
            "max_depth=8, min_samples_leaf=3, max_features='sqrt'), tuneado con "
            "GroupKFold(5) por país. Evaluación final en 41 países nunca vistos. "
            "Supera al baseline lineal de la Entrega 1 (que daba R² &lt; 0 con el "
            "mismo split por país) y al dummy de la media.",
            body,
        )
    )

    # ---------- Hallazgos ----------
    story.append(Paragraph("Hallazgos", h2))
    story.append(
        Paragraph(
            "• <b>La señal es no lineal.</b> La regresión lineal (Entrega 1) "
            "no podía generalizar a países nuevos; los modelos basados en árboles "
            "sí lo logran porque capturan respuestas con tramos.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "• <b>La feature más informativa es la prevalencia de esquizofrenia</b> "
            "(permutation importance ≈ 0.28 de RMSE; ablación: +0.09 al quitarla). "
            "Le siguen ansiedad y bipolaridad. El año aporta casi nada.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "• <b>El año aporta casi nada</b>: la prevalencia estandarizada por edad "
            "cambia poco en 30 años. La señal está en la <i>estructura transversal</i> "
            "de comorbilidad, no en su evolución temporal.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "• <b>Generalizar a países nuevos es más difícil que generalizar a años "
            "nuevos del mismo país.</b> En split temporal estricto (train ≤2009 / "
            f"test 2010-2019) el R² sube a {fmt(m['temporal_validation']['R2'])} "
            "porque el modelo interpola años, no extrapola países.",
            bullet,
        )
    )

    # ---------- Limitaciones ----------
    story.append(Paragraph("Limitaciones que afectan el uso del modelo", h2))
    story.append(
        Paragraph(
            "• <b>Alta sensibilidad al split.</b> Con 5 semillas distintas el R² "
            "varía entre 0.16 y 0.58. El número del recuadro corresponde a SEED=42; "
            "para una decisión real conviene reportar promedio sobre semillas y su "
            "rango.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "• <b>Datos del IHME, no observaciones directas.</b> El target son "
            "estimaciones suavizadas por modelos epidemiológicos. El modelo aprende "
            "patrones de esa fuente, no de mediciones brutas.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "• <b>No es inferencia causal.</b> Que <i>schizophrenia</i> sea la "
            "feature más útil para predecir <i>depression</i> en este modelo "
            "<b>no implica</b> que la cause; las prevalencias se mueven juntas "
            "porque comparten determinantes sociales no observados.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "• <b>Núcleo persistente de países con error alto:</b> Perú, Marruecos, "
            "Cuba, Alemania, Polonia. Las 5 features no capturan suficiente "
            "contexto regional o socioeconómico.",
            bullet,
        )
    )

    # ---------- Recomendaciones ----------
    story.append(Paragraph("Recomendaciones", h2))
    story.append(
        Paragraph(
            "1. <b>Para uso académico o exploratorio:</b> el modelo es utilizable "
            "como referencia para comparar perfiles de comorbilidad entre países. "
            "Reportar siempre el intervalo, no solo el punto.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "2. <b>Antes de pensar en despliegue:</b> agregar features "
            "socioeconómicas (PIB per cápita, gasto en salud, urbanización) "
            "para subir el R² de ~0.4 a 0.6+ y validar con varias semillas.",
            bullet,
        )
    )
    story.append(
        Paragraph(
            "3. <b>No usar para decisiones individuales</b>: el modelo opera a "
            "nivel país-año, no a nivel persona.",
            bullet,
        )
    )

    # ---------- Foto ----------
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Image(str(FIG / "23_permutation_final.png"), width=12 * cm, height=7 * cm)
    )

    SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Entrega 3 - Resumen ejecutivo",
        author="Valentina Giraldo",
    ).build(story)
    print(f"Resumen ejecutivo generado en {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    build()
