"""Genera el reporte PDF de la Entrega 2.

Lee report/entrega2_metrics.json y las figuras en figures/ para
producir report/entrega2_reporte.pdf con la metodologia, los
resultados comparativos y la conclusion provisional.
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
METRICS = PROJECT / "report" / "entrega2_metrics.json"
FIG = PROJECT / "figures"
OUT = PROJECT / "report" / "entrega2_reporte.pdf"


def fmt(x, decimals=4):
    return f"{x:.{decimals}f}"


def build():
    m = json.loads(METRICS.read_text(encoding="utf-8"))

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=18,
        spaceAfter=10,
        textColor=colors.HexColor("#1f3a5f"),
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#2a5a8a"),
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10.5,
        leading=14,
        spaceAfter=6,
        alignment=4,  # justify
    )
    small = ParagraphStyle(
        "Small", parent=styles["BodyText"], fontSize=9, leading=11, textColor=colors.grey
    )

    story = []

    # ---------- Title ----------
    story.append(Paragraph("Entrega 2 — Comparación de familias y validación", h1))
    story.append(
        Paragraph(
            "Proyecto aplicado · Aprendizaje de Máquina Aplicado · EAFIT · 2026-04-30",
            small,
        )
    )
    story.append(Paragraph("Valentina Giraldo", small))
    story.append(Spacer(1, 0.4 * cm))

    # ---------- Resumen ----------
    story.append(Paragraph("Resumen ejecutivo", h2))
    story.append(
        Paragraph(
            "En esta entrega comparo tres familias de modelos para predecir la prevalencia "
            "de depresión a nivel país-año a partir de la prevalencia de los otros cuatro "
            "trastornos mentales del dataset IHME/GBD. Mantengo el problema, las features y "
            "el target definidos en la Entrega 1, y uso una validación sin fuga entre países "
            "(GroupShuffleSplit externo + GroupKFold interno). Los resultados muestran que "
            "los modelos basados en árboles (Random Forest y, sobre todo, "
            "<b>Histogram Gradient Boosting</b>) sí logran superar al dummy en países no vistos, "
            "mientras que las familias lineales se quedan por debajo. El mejor candidato "
            f"obtiene RMSE = {fmt(m['test_metrics']['HistGB']['RMSE'], 3)} y "
            f"R² = {fmt(m['test_metrics']['HistGB']['R2'], 3)} en test.",
            body,
        )
    )

    # ---------- Problema y datos ----------
    story.append(Paragraph("1. Problema y datos (recap)", h2))
    story.append(
        Paragraph(
            "Tarea de regresión supervisada: predecir <i>depression</i> (share de la población "
            "con trastorno depresivo, estandarizado por edad) a partir de "
            "<i>schizophrenia, anxiety, bipolar, eating</i> y <i>Year</i>. Tras excluir los "
            "9 agregados regionales del CSV, trabajo con "
            f"{m['n_train']+m['n_test']} filas y "
            f"{m['n_countries_train']+m['n_countries_test']} países entre 1990 y 2019. La "
            "métrica principal es <b>RMSE</b> (en puntos porcentuales del target), "
            "complementada por MAE y R².",
            body,
        )
    )

    # ---------- Metodología ----------
    story.append(Paragraph("2. Metodología y protocolo de validación", h2))
    story.append(
        Paragraph(
            "<b>Split externo:</b> <i>GroupShuffleSplit</i> 80/20 por <i>Entity</i> "
            f"(SEED={m['seed']}). Entreno con "
            f"{m['n_train']} filas / {m['n_countries_train']} países; reservo "
            f"{m['n_test']} filas / {m['n_countries_test']} países para una única "
            "evaluación final.<br/>"
            "<b>Tuning interno:</b> <i>GridSearchCV</i> con "
            "<i>GroupKFold(5)</i> por país sobre el train, scoring=neg_RMSE.<br/>"
            "<b>Evitar fuga:</b> el escalado se hace dentro del <i>Pipeline</i>, así que cada "
            "fold ajusta el <i>StandardScaler</i> solo con su propio train. El test no se "
            "toca hasta el final.<br/>"
            "<b>Familias comparadas:</b> (i) lineal regularizado — ElasticNet "
            "(α, l1_ratio); (ii) Random Forest (n_estimators, max_depth, "
            "min_samples_leaf); (iii) Histogram Gradient Boosting (learning_rate, "
            "max_iter, max_depth). Se incluye un <i>DummyRegressor</i> (media) y la "
            "<i>LinearRegression</i> de la Entrega 1 como referencias.",
            body,
        )
    )

    # ---------- Tabla CV ----------
    story.append(Paragraph("3. Resultados de validación cruzada", h2))
    cv_data = [["Modelo", "RMSE CV (media)", "± std"]]
    for row in m["cv_table"]:
        cv_data.append(
            [
                row["model"],
                fmt(row["cv_rmse_mean"], 4),
                fmt(row["cv_rmse_std"], 4),
            ]
        )
    t1 = Table(cv_data, colWidths=[6 * cm, 4 * cm, 3 * cm])
    t1.setStyle(
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
    story.append(t1)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Las tres familias entrenadas mejoran claramente al dummy en CV; entre ellas "
            "<b>HistGB</b> es la mejor (RMSE ≈ 0.76), seguida de cerca por ElasticNet y "
            "del LinReg de Entrega 1. La desviación estándar entre folds es alta "
            "(0.13–0.20), lo que indica que algunos países son más difíciles de predecir "
            "que otros y el promedio depende mucho de la composición del fold.",
            body,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Image(str(FIG / "08_cv_comparison.png"), width=15 * cm, height=8.4 * cm)
    )

    story.append(PageBreak())

    # ---------- Tabla Test ----------
    story.append(Paragraph("4. Evaluación final en test (países no vistos)", h2))
    tm_data = [["Modelo", "RMSE", "MAE", "R²"]]
    for name, mm in m["test_metrics"].items():
        tm_data.append(
            [name, fmt(mm["RMSE"], 4), fmt(mm["MAE"], 4), fmt(mm["R2"], 4)]
        )
    t2 = Table(tm_data, colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
    t2.setStyle(
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
    story.append(t2)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "<b>HistGB</b> obtiene RMSE ≈ "
            f"{fmt(m['test_metrics']['HistGB']['RMSE'], 3)} y R² ≈ "
            f"{fmt(m['test_metrics']['HistGB']['R2'], 3)} en países que no vio durante el "
            "entrenamiento. Random Forest también supera al dummy (R² ≈ "
            f"{fmt(m['test_metrics']['RandomForest']['R2'], 3)}). Las familias lineales, "
            "incluso con regularización (ElasticNet), quedan por debajo del dummy "
            "(R² &lt; 0): el sesgo del modelo lineal no logra absorber la heterogeneidad "
            "entre países.",
            body,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Image(str(FIG / "09_test_metrics.png"), width=15.5 * cm, height=5.5 * cm)
    )

    # ---------- Best params ----------
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("5. Hiperparámetros seleccionados", h2))
    bp = m["tuned_best_params"]
    bp_data = [["Familia", "Mejores hiperparámetros"]]
    for name, params in bp.items():
        bp_data.append(
            [name, ", ".join(f"{k.replace('model__','')}={v}" for k, v in params.items())]
        )
    t3 = Table(bp_data, colWidths=[3.5 * cm, 12 * cm])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(t3)

    # ---------- Análisis de errores ----------
    story.append(PageBreak())
    story.append(Paragraph("6. Análisis de errores", h2))
    story.append(
        Paragraph(
            "El gráfico de residuos muestra que los modelos lineales mantienen una banda "
            "inclinada característica (subestiman valores altos y sobrestiman bajos), "
            "mientras que la nube de residuos de los modelos de árboles se centra mejor "
            "alrededor de cero, aunque todavía con outliers. El listado de países con "
            "mayor MAE coincide en buena parte con casos extremos del EDA (perfiles "
            "atípicos en una o más prevalencias) y sugiere que añadir información regional "
            "o de grupo de ingreso podría mejorar el modelo en la Entrega 3.",
            body,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Image(str(FIG / "10_residuos_familias.png"), width=16 * cm, height=4.4 * cm)
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Image(str(FIG / "13_top_errores.png"), width=13 * cm, height=8.5 * cm)
    )

    # ---------- Interpretabilidad ----------
    story.append(PageBreak())
    story.append(Paragraph("7. Interpretabilidad (permutation importance)", h2))
    pi = m["permutation_importance"]
    pi_data = [["Feature", "Aumento RMSE", "± std"]]
    for f, im, isd in zip(
        pi["features"], pi["importance_mean"], pi["importance_std"]
    ):
        pi_data.append([f, fmt(im, 4), fmt(isd, 4)])
    t4 = Table(pi_data, colWidths=[5 * cm, 4 * cm, 3 * cm])
    t4.setStyle(
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
    story.append(t4)
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            f"Sobre el modelo ganador ({pi['model']}), la feature con mayor impacto al "
            "permutarse es la <b>prevalencia de esquizofrenia</b> "
            f"(+{fmt(pi['importance_mean'][0], 3)} de RMSE), seguida de la <b>ansiedad</b> "
            f"(+{fmt(pi['importance_mean'][1], 3)}). Las demás aportan menos y el <b>año</b> "
            "es prácticamente irrelevante. Este resultado coincide con el orden de magnitud "
            "de los coeficientes estandarizados del baseline lineal de la Entrega 1, aunque "
            "no con la correlación marginal observada en el EDA (donde la ansiedad parecía "
            "el predictor más fuerte): es un buen recordatorio de que la importancia "
            "marginal y la importancia condicional pueden no coincidir.",
            body,
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Image(str(FIG / "12_permutation_importance.png"), width=15 * cm, height=8.8 * cm)
    )

    # ---------- Desbalance / umbral ----------
    story.append(PageBreak())
    story.append(Paragraph("8. ¿Desbalance o ajuste de umbral?", h2))
    story.append(
        Paragraph(
            "Estos análisis pertenecen al universo de la clasificación. Como el problema "
            "aquí es de regresión sobre un share continuo, no aplica desbalance de clases "
            "ni ajuste de umbral. El equivalente relevante para regresión es el "
            "<b>sesgo del target</b>: el target está sesgado a la derecha y un puñado de "
            "países concentra los valores más altos. Por eso reporto tanto RMSE (más "
            "sensible a esos extremos) como MAE (más robusto). La diferencia entre RMSE y "
            f"MAE del modelo ganador ({fmt(m['test_metrics']['HistGB']['RMSE'], 3)} vs "
            f"{fmt(m['test_metrics']['HistGB']['MAE'], 3)}) confirma que hay pocos países "
            "con errores grandes que elevan el RMSE; el listado del top-15 los identifica.",
            body,
        )
    )

    # ---------- Conclusión provisional ----------
    story.append(Paragraph("9. Conclusión provisional y limitaciones", h2))
    story.append(
        Paragraph(
            "<b>Decisión provisional:</b> el modelo más prometedor es "
            "<i>HistGradientBoostingRegressor</i>. Es el único que supera de forma clara al "
            "dummy en países no vistos (con un margen significativo) y mantiene una buena "
            "interpretabilidad vía permutation importance. La regularización lineal "
            "no fue suficiente para cerrar la brecha de Entrega 1.<br/><br/>"
            "<b>Limitaciones abiertas:</b> (i) un subconjunto de países concentra el error y "
            "podría beneficiarse de features regionales o de grupo de ingreso; (ii) no se "
            "ha probado una validación temporal estricta (train hasta año <i>t</i>, test "
            "después), que sería el escenario más realista para predicción prospectiva; "
            "(iii) el grid de hiperparámetros es modesto: tuneos más finos o stacking "
            "podrían mover el resultado. Estos puntos quedan como agenda para la "
            "Entrega 3.",
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
        title="Entrega 2 - Comparacion de familias",
        author="Valentina Giraldo",
    ).build(story)
    print(f"PDF generado en {OUT.relative_to(PROJECT)}")


if __name__ == "__main__":
    build()
