"""Genera el reporte PDF final de la Entrega 3.

Lee report/entrega3_metrics.json y las figuras en figures/ para
producir report/entrega3_reporte.pdf con la version definitiva del
informe: resumen ejecutivo, metodologia, resultados, discusion,
limitaciones y recomendaciones.
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
    PageBreak,
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
OUT = PROJECT / "report" / "entrega3_reporte.pdf"


def fmt(x, decimals=4):
    return f"{x:.{decimals}f}"


def make_styles():
    base = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontSize=18,
        spaceAfter=10,
        textColor=colors.HexColor("#1f3a5f"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2a5a8a"),
    )
    h3 = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#3b6aa0"),
    )
    body = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
        alignment=4,
    )
    small = ParagraphStyle(
        "Small",
        parent=base["BodyText"],
        fontSize=9,
        leading=11,
        textColor=colors.grey,
    )
    return h1, h2, h3, body, small


def header_table(rows, col_widths):
    t = Table(rows, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def build():
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    h1, h2, h3, body, small = make_styles()
    story = []

    # ---------- Cover ----------
    story.append(
        Paragraph(
            "Entrega 3 — Modelo final, interpretación y comunicación",
            h1,
        )
    )
    story.append(
        Paragraph(
            "Predicción de la prevalencia de depresión a nivel país-año a partir de "
            "la prevalencia de otros trastornos mentales (IHME · GBD).",
            small,
        )
    )
    story.append(
        Paragraph(
            "Proyecto aplicado · Aprendizaje de Máquina Aplicado · EAFIT · 2026-05-14",
            small,
        )
    )
    story.append(Paragraph("Valentina Giraldo", small))
    story.append(Spacer(1, 0.5 * cm))

    # ---------- Resumen ejecutivo ----------
    story.append(Paragraph("Resumen ejecutivo", h2))
    fm = m["final_test_metrics"]
    ci = m["bootstrap_ci"]
    story.append(
        Paragraph(
            "Este proyecto resuelve una tarea de regresión supervisada: estimar la "
            "prevalencia de depresión (share de la población, estandarizado por edad) "
            "a nivel país-año, a partir de la prevalencia de los otros cuatro "
            "trastornos mentales del dataset IHME/GBD y del año. Tras tres iteraciones "
            "(CRISP-DM), el modelo final es un <b>Random Forest tuneado con "
            "GroupKFold(5) por país</b>. Sobre 41 países nunca vistos durante el "
            f"entrenamiento, alcanza <b>RMSE = {fmt(fm['RMSE'], 3)}</b> "
            f"(IC95% [{fmt(ci['RMSE']['lo'], 3)}, {fmt(ci['RMSE']['hi'], 3)}]), "
            f"<b>MAE = {fmt(fm['MAE'], 3)}</b> y "
            f"<b>R² = {fmt(fm['R2'], 3)}</b> "
            f"(IC95% [{fmt(ci['R2']['lo'], 3)}, {fmt(ci['R2']['hi'], 3)}]). "
            "Mejora claramente al dummy y al baseline lineal de la Entrega 1, y la "
            "prevalencia de esquizofrenia resulta ser la feature más informativa. "
            "El proyecto deja como limitación principal la <b>alta sensibilidad al "
            "split de países</b>: el R² varía entre 0.16 y 0.58 según qué 41 países "
            "caen en test.",
            body,
        )
    )

    # ---------- 1. Problema y datos ----------
    story.append(Paragraph("1. Problema y datos", h2))
    story.append(
        Paragraph(
            "<b>Pregunta:</b> ¿se puede predecir la prevalencia de depresión a nivel "
            "país-año a partir de la prevalencia de otros trastornos mentales? "
            "<b>Tarea:</b> regresión supervisada. <b>Target:</b> <i>depression</i> "
            "(prevalencia, share de la población). <b>Features:</b> "
            "<i>schizophrenia, anxiety, bipolar, eating, Year</i>. "
            "<b>Unidad de observación:</b> par (país, año). Tras excluir los 9 "
            "agregados regionales del CSV (Africa, Europe, EU27, grupos de ingreso, "
            "etc.) trabajo con "
            f"{m['n_train']+m['n_test']} filas y "
            f"{m['n_countries_train']+m['n_countries_test']} países entre 1990 y "
            "2019. La <b>métrica principal es RMSE</b> en puntos porcentuales del "
            "target, complementada por MAE (más robusto a outliers) y R² (lectura "
            "intuitiva). La elección del dataset y del target se justificó en la "
            "Entrega 1.",
            body,
        )
    )

    # ---------- 2. Metodología ----------
    story.append(Paragraph("2. Metodología y protocolo", h2))
    story.append(
        Paragraph(
            "<b>Marco metodológico:</b> CRISP-DM (Entrega 1 = comprensión + datos + "
            "baseline; Entrega 2 = comparación de familias; Entrega 3 = modelo final, "
            "interpretación y comunicación).<br/>"
            f"<b>Split externo:</b> <i>GroupShuffleSplit</i> 80/20 por <i>Entity</i> "
            f"con SEED={m['seed']} (heredado de Entregas 1 y 2). "
            f"Entreno con {m['n_train']} filas / {m['n_countries_train']} países; "
            f"test con {m['n_test']} filas / {m['n_countries_test']} países nunca "
            "vistos hasta la evaluación final.<br/>"
            "<b>Tuning interno:</b> <i>GridSearchCV</i> con <i>GroupKFold(5)</i> "
            "por país sobre el train, scoring = neg_RMSE. Cada candidato se entrena "
            "dentro de un <i>Pipeline</i> para evitar fuga entre folds.<br/>"
            "<b>Familias re-evaluadas en Entrega 3:</b> HistGradientBoosting "
            "(grid amplio + early stopping), Random Forest (grid amplio) y "
            "Ensemble por promedio de los dos. Como referencia se mantiene la "
            "comparación con el Dummy y el LinReg de la Entrega 1.<br/>"
            "<b>Validación adicional:</b> bootstrap de 2000 muestras del test para "
            "calcular intervalos de confianza, repetición con 5 semillas distintas "
            "del split externo para medir sensibilidad y un split temporal estricto "
            "(train 1990-2009 / test 2010-2019) como protocolo alternativo.",
            body,
        )
    )

    # ---------- 3. Tuning ----------
    story.append(Paragraph("3. Tuning final de las familias candidatas", h2))
    tn = m["tuning"]
    tn_rows = [["Familia", "CV-RMSE", "± std", "Mejores hiperparámetros"]]
    for fam, info in tn.items():
        params_str = ", ".join(
            f"{k.replace('model__','')}={v}" for k, v in info["best_params"].items()
        )
        tn_rows.append(
            [
                fam,
                fmt(info["cv_rmse_mean"], 4),
                fmt(info["cv_rmse_std"], 4),
                params_str,
            ]
        )
    story.append(header_table(tn_rows, [3 * cm, 2.5 * cm, 2 * cm, 9 * cm]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Con el grid ampliado, el CV-RMSE de HistGB pasa de 0.76 (Entrega 2) a "
            f"{fmt(tn['HistGB']['cv_rmse_mean'], 3)} y el de RF de 0.77 a "
            f"{fmt(tn['RF']['cv_rmse_mean'], 3)}. La mejora del RF es mayor: "
            "indicaría que el grid corto de la Entrega 2 estaba lejos de su óptimo. "
            "Ambas familias siguen mostrando std de CV alta (≈ 0.13), confirmando "
            "que el desempeño promedio entre folds esconde heterogeneidad fuerte "
            "entre países.",
            body,
        )
    )

    # ---------- 4. Evaluación en test ----------
    story.append(PageBreak())
    story.append(Paragraph("4. Evaluación final en test", h2))
    test_rows = [["Modelo", "RMSE", "MAE", "R²"]]
    for name, mm in m["test_metrics"].items():
        test_rows.append(
            [name, fmt(mm["RMSE"], 4), fmt(mm["MAE"], 4), fmt(mm["R2"], 4)]
        )
    story.append(header_table(test_rows, [5 * cm, 3 * cm, 3 * cm, 3 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Image(str(FIG / "14_final_comparison.png"), width=16 * cm, height=5 * cm)
    )
    story.append(
        Paragraph(
            f"<b>Decisión:</b> el modelo final es <b>{m['final_model']}</b>, por "
            "menor RMSE en test. La diferencia con HistGB es pequeña pero "
            "consistente, y el ensemble por promedio no aporta más que RF sola.",
            body,
        )
    )

    # ---------- 5. Confianza ----------
    story.append(Paragraph("5. Confianza del resultado — bootstrap", h2))
    ci_rows = [["Métrica", "Valor", "IC95% inf.", "IC95% sup."]]
    for met in ["RMSE", "MAE", "R2"]:
        ci_rows.append(
            [
                met,
                fmt(ci[met]["mean"], 4),
                fmt(ci[met]["lo"], 4),
                fmt(ci[met]["hi"], 4),
            ]
        )
    story.append(header_table(ci_rows, [3 * cm, 3 * cm, 3 * cm, 3 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Image(str(FIG / "15_bootstrap_ci.png"), width=14 * cm, height=8 * cm)
    )
    story.append(
        Paragraph(
            "Con 2000 remuestras del test, el intervalo de confianza al 95 % del "
            f"RMSE es estrecho ([{fmt(ci['RMSE']['lo'], 3)}, "
            f"{fmt(ci['RMSE']['hi'], 3)}]). El R² se ubica con confianza por encima "
            "de 0 (el dummy daría 0). Eso significa que, <i>dentro</i> de este "
            "split de países, el resultado es estadísticamente robusto.",
            body,
        )
    )

    # ---------- 6. Sensibilidad a la semilla ----------
    story.append(PageBreak())
    story.append(Paragraph("6. Sensibilidad al split externo", h2))
    sd_rows = [["Seed", "RMSE", "MAE", "R²"]]
    for r in m["seed_sensitivity"]:
        sd_rows.append(
            [
                str(r["seed"]),
                fmt(r["RMSE"], 4),
                fmt(r["MAE"], 4),
                fmt(r["R2"], 4),
            ]
        )
    story.append(header_table(sd_rows, [2 * cm, 3 * cm, 3 * cm, 3 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Image(str(FIG / "16_seed_sensitivity.png"), width=16 * cm, height=5 * cm)
    )
    story.append(
        Paragraph(
            "<b>Esta es la limitación más fuerte del proyecto.</b> Re-entrenando el "
            "modelo con los mismos hiperparámetros sobre 5 semillas distintas del "
            "split externo, el R² varía entre <b>0.16 y 0.58</b> y el RMSE entre "
            "<b>0.57 y 0.93</b>. Con 205 países y solo 5 features, el resultado "
            "depende fuertemente de qué subconjunto cae como test. Para un uso real "
            "habría que reportar la <i>distribución</i> de métricas sobre varias "
            "semillas, no un solo punto. El número que reporto en el resumen "
            "(R² ≈ 0.43) es el caso de SEED=42 — el mismo que vengo usando desde la "
            "Entrega 1 por consistencia, pero no es la performance esperada en "
            "cualquier conjunto de 41 países nuevos.",
            body,
        )
    )

    # ---------- 7. Validación temporal ----------
    story.append(Paragraph("7. Validación alternativa: split temporal", h2))
    tmp = m["temporal_validation"]
    tt_rows = [
        ["Protocolo", "n_train", "n_test", "RMSE", "MAE", "R²"],
        [
            "Por país (final)",
            m["n_train"],
            m["n_test"],
            fmt(fm["RMSE"], 4),
            fmt(fm["MAE"], 4),
            fmt(fm["R2"], 4),
        ],
        [
            "Temporal 1990-2009 / 2010-2019",
            tmp["n_train"],
            tmp["n_test"],
            fmt(tmp["RMSE"], 4),
            fmt(tmp["MAE"], 4),
            fmt(tmp["R2"], 4),
        ],
    ]
    story.append(header_table(tt_rows, [5.5 * cm, 1.8 * cm, 1.8 * cm, 2 * cm, 2 * cm, 2 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Image(str(FIG / "19_temporal_vs_country.png"), width=14 * cm, height=7 * cm)
    )
    story.append(
        Paragraph(
            "El split temporal <b>parece mucho mejor</b> (R² ≈ 0.83), pero hay que "
            "leerlo con cuidado: aunque no se repite ningún par (país, año), el "
            "train contiene años previos del mismo país, y la prevalencia "
            "estandarizada por edad varía poco año a año. El modelo no extrapola a "
            "países nuevos, interpola años nuevos. Son dos preguntas distintas y "
            "ambas son legítimas, pero la honesta para una recomendación de salud "
            "pública en países pobremente representados en GBD es la del split por "
            "país (R² ≈ 0.43), no la temporal.",
            body,
        )
    )

    # ---------- 8. Interpretabilidad ----------
    story.append(PageBreak())
    story.append(Paragraph("8. Interpretabilidad", h2))
    pi = m["permutation_importance"]
    pi_rows = [["Feature", "Aumento RMSE", "± std"]]
    for f, im, isd in zip(
        pi["features"], pi["importance_mean"], pi["importance_std"]
    ):
        pi_rows.append([f, fmt(im, 4), fmt(isd, 4)])
    story.append(header_table(pi_rows, [4.5 * cm, 4 * cm, 3 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Image(str(FIG / "23_permutation_final.png"), width=14 * cm, height=8 * cm)
    )
    story.append(
        Paragraph(
            "La feature más informativa es la <b>prevalencia de esquizofrenia</b> "
            f"(+{fmt(pi['importance_mean'][0], 3)} de RMSE al permutarla), seguida de "
            f"<b>ansiedad</b> (+{fmt(pi['importance_mean'][1], 3)}) y "
            f"<b>bipolaridad</b> (+{fmt(pi['importance_mean'][2], 3)}). El "
            "<b>año</b> es prácticamente irrelevante (importancia centrada en cero). "
            "Este orden es estable entre las semillas que probé.",
            body,
        )
    )
    story.append(Spacer(1, 0.15 * cm))
    story.append(
        Image(str(FIG / "18_partial_dependence.png"), width=15 * cm, height=5.5 * cm)
    )
    story.append(
        Paragraph(
            "Las curvas de <i>partial dependence</i> sobre las dos features "
            "principales muestran que la relación de <i>schizophrenia</i> con la "
            "predicción <b>no es lineal</b>: el modelo aprende una respuesta con "
            "tramos planos y un tramo creciente. La <i>anxiety</i> sigue una "
            "respuesta cercana a lineal y monótona. Esto explica por qué el modelo "
            "lineal de la Entrega 1 era insuficiente.",
            body,
        )
    )

    # ---------- 9. Sensibilidad por ablación ----------
    story.append(PageBreak())
    story.append(Paragraph("9. Análisis de sensibilidad por ablación", h2))
    abl = m["ablation"]
    abl_rows = [["Feature removida", "RMSE", "Δ RMSE"]]
    base = abl["_baseline_all_features"]["RMSE"]
    abl_rows.append(["(ninguna — baseline)", fmt(base, 4), "0.0000"])
    for k, v in abl.items():
        if k == "_baseline_all_features":
            continue
        abl_rows.append([k, fmt(v["RMSE"], 4), fmt(v["delta_RMSE"], 4)])
    story.append(header_table(abl_rows, [5 * cm, 3 * cm, 3 * cm]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Image(str(FIG / "17_ablation.png"), width=14 * cm, height=7 * cm))
    story.append(
        Paragraph(
            "Quitar <i>schizophrenia</i> es lo que más empeora el modelo "
            f"(+{fmt(abl['schizophrenia']['delta_RMSE'], 3)} de RMSE), confirmando "
            "que su contribución no es redundante con las otras features. Quitar "
            "<i>Year</i> mejora levemente, lo que sugiere que aporta más ruido que "
            "señal — consistente con la baja permutation importance. Las demás "
            "features aportan moderadamente.",
            body,
        )
    )

    # ---------- 10. Análisis de errores ----------
    story.append(PageBreak())
    story.append(Paragraph("10. Análisis de errores", h2))
    story.append(
        Paragraph(
            "Los residuos en test no muestran sesgo sistemático (la media de "
            "residuos está cerca de cero) pero sí tienen colas: hay un grupo de "
            "países con error sustancial. Los identifico abajo.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * cm))
    story.append(
        Image(str(FIG / "21_residuals_final.png"), width=15 * cm, height=6 * cm)
    )
    story.append(Spacer(1, 0.1 * cm))
    story.append(
        Image(str(FIG / "22_pvt_final.png"), width=10 * cm, height=9 * cm)
    )
    story.append(
        Image(str(FIG / "20_top_errors_final.png"), width=14 * cm, height=11 * cm)
    )
    top_countries = list(m["top_error_countries"].items())[:10]
    story.append(
        Paragraph(
            "El listado del top 10 incluye " + ", ".join([c for c, _ in top_countries])
            + ". Son una mezcla de países con perfil epidemiológico atípico "
              "(Perú, Marruecos, Cuba — ya identificados como outliers en el EDA de "
              "la Entrega 1), países europeos con depresión alta pero comorbilidades "
              "moderadas (Alemania, Polonia, Portugal) y países pequeños con "
              "estimaciones más ruidosas en GBD (Saint Kitts and Nevis, Lesotho). "
              "Esto sugiere que las 5 features actuales no capturan suficiente "
              "información de contexto regional/socioeconómico.",
            body,
        )
    )

    # ---------- 11. Discusión ----------
    story.append(PageBreak())
    story.append(Paragraph("11. Discusión y respuestas a la rúbrica", h2))

    story.append(Paragraph("¿Cuál es el mejor modelo y por qué?", h3))
    story.append(
        Paragraph(
            f"<b>{m['final_model']}</b> con "
            + ", ".join(
                f"{k.replace('model__','')}={v}"
                for k, v in m["tuning"]["RF"]["best_params"].items()
            )
            + ". Tiene el menor RMSE en test (por encima de HistGB tuneado y del "
              "ensemble), es robusto sin necesitar early stopping ni regularización "
              "explícita, y el ensemble HistGB+RF no mejora sobre RF sola. "
              "La elección del baseline lineal de la Entrega 1 quedó descartada "
              "por su incapacidad de capturar la relación no lineal de "
              "<i>schizophrenia</i> con la prevalencia de depresión.",
            body,
        )
    )

    story.append(Paragraph("¿Qué tan confiables son sus resultados?", h3))
    story.append(
        Paragraph(
            f"<b>Dentro del split fijado</b> el resultado es robusto: bootstrap (n=2000) "
            f"da RMSE = {fmt(ci['RMSE']['mean'], 3)} con IC95% "
            f"[{fmt(ci['RMSE']['lo'], 3)}, {fmt(ci['RMSE']['hi'], 3)}] y R² = "
            f"{fmt(ci['R2']['mean'], 3)} con IC95% "
            f"[{fmt(ci['R2']['lo'], 3)}, {fmt(ci['R2']['hi'], 3)}]. "
            "<b>Entre splits</b> el resultado es mucho más variable: el R² entre "
            "5 semillas distintas varía entre 0.16 y 0.58. Para una decisión real "
            "habría que reportar el promedio sobre varias semillas y su rango.",
            body,
        )
    )

    story.append(Paragraph("¿Qué variables o patrones explican el desempeño?", h3))
    story.append(
        Paragraph(
            "La señal principal está en la <b>estructura transversal de "
            "comorbilidad</b> entre los cuatro trastornos mentales. La feature "
            "dominante es <i>schizophrenia</i>, con una relación no lineal respecto "
            "al target. <i>Anxiety</i> y <i>bipolar</i> aportan un nivel intermedio. "
            "El <i>año</i> es prácticamente irrelevante: la prevalencia "
            "estandarizada por edad cambia muy poco en 30 años. Lo importante "
            "epidemiológicamente es <i>qué tan comórbidos</i> son los trastornos "
            "en un país, no en qué año estamos.",
            body,
        )
    )

    story.append(Paragraph("¿Qué conclusiones útiles deja el proyecto?", h3))
    story.append(
        Paragraph(
            "1) Las prevalencias de otros trastornos mentales <b>sí contienen "
            "información predictiva</b> sobre la prevalencia de depresión a nivel "
            "país, pero la relación es no lineal y la familia lineal no la captura. "
            "2) La generalización a <b>países nuevos</b> es limitada (R² ≈ 0.43), "
            "pero la generalización a <b>años nuevos del mismo país</b> es alta "
            "(R² ≈ 0.83). Son problemas distintos. "
            "3) Hay un núcleo persistente de países donde el error es alto; con las "
            "5 features actuales no se explican.",
            body,
        )
    )

    story.append(Paragraph("¿Qué haría falta para mejorar o desplegar?", h3))
    story.append(
        Paragraph(
            "<b>Features:</b> añadir indicadores socioeconómicos (GDP per cápita, "
            "gasto en salud, urbanización, índices de equidad) probablemente mueva "
            "el R² de ~0.4 a 0.6+. "
            "<b>Modelado:</b> probar enfoques jerárquicos que respeten la estructura "
            "país/región/grupo de ingreso. "
            "<b>Validación:</b> reportar promedio sobre varias semillas y validación "
            "cruzada por región. "
            "<b>Despliegue:</b> intervalos de predicción explícitos (cuantiles de "
            "RF), alertas si los inputs de un país nuevo caen fuera del rango "
            "aprendido, y monitoreo de drift al actualizar el GBD anualmente.",
            body,
        )
    )

    # ---------- 12. Limitaciones ----------
    story.append(Paragraph("12. Limitaciones explícitas", h2))
    story.append(
        Paragraph(
            "1) El target son <b>estimaciones del IHME</b>, no observaciones "
            "directas; ya están suavizadas por sus modelos epidemiológicos. "
            "Cualquier patrón aprendido por mi modelo es relativo a esa fuente.<br/>"
            "2) Las 5 features capturan solo la <b>comorbilidad entre trastornos</b>; "
            "no incluyen variables sociales, económicas ni de sistema de salud.<br/>"
            "3) El resultado <b>depende del split</b> de países (R² entre 0.16 y "
            "0.58 según la semilla). No es la limitación del modelo, sino del "
            "tamaño del dataset (205 países).<br/>"
            "4) <b>No estoy haciendo inferencia causal.</b> Decir que "
            "<i>schizophrenia</i> es la feature más importante <b>no implica</b> "
            "que cause depresión; es la variable que más reduce el error en este "
            "modelo concreto sobre estos datos concretos.<br/>"
            "5) <b>No hay validación geográfica fuera del IHME</b>: si las "
            "prevalencias estimadas por otra fuente (encuestas nacionales, p.ej.) "
            "fueran sistemáticamente distintas, el modelo no se transferiría sin "
            "calibración.",
            body,
        )
    )

    # ---------- 13. Reproducibilidad ----------
    story.append(Paragraph("13. Reproducibilidad", h2))
    story.append(
        Paragraph(
            "Todo el proyecto es reproducible end-to-end. Semilla fija "
            f"<b>SEED={m['seed']}</b>. Para regenerar la Entrega 3:<br/><br/>"
            "<font face='Courier'>"
            "cd project<br/>"
            "pip install -r requirements.txt<br/>"
            "python src/build_entrega3.py        # experimentos + figuras + JSON<br/>"
            "python src/build_notebook_entrega3.py  # regenera el notebook<br/>"
            "python src/build_pdf_entrega3.py    # regenera este PDF"
            "</font><br/><br/>"
            "El detalle del checklist de reproducibilidad está en "
            "<i>report/reproducibility_checklist.md</i>.",
            body,
        )
    )

    SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Entrega 3 - Modelo final",
        author="Valentina Giraldo",
    ).build(story)
    print(f"PDF generado en {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    build()
