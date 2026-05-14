"""
build_entrega2.py
Script reproducible de la Entrega 2 del proyecto de Aprendizaje de Maquina.

Lo que hace este script:
  1. Carga el CSV de mental-illnesses-prevalence y aplica la misma limpieza
     que en la Entrega 1 (excluye agregados regionales).
  2. Construye un split por pais sin fuga (mismo SEED=42 que Entrega 1)
     y reserva ~20% de paises como test final.
  3. Entrena y tuned tres familias de modelos con GroupKFold por pais sobre
     el train:
       - Lineal regularizado (ElasticNet, dentro de Pipeline con StandardScaler)
       - Random Forest
       - Histogram Gradient Boosting
  4. Compara las familias por CV (RMSE) y evalua la mejor en el test set.
  5. Guarda figuras PNG en figures/ y metricas en report/entrega2_metrics.json.

Como correrlo:
    cd project
    python src/build_entrega2.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    GroupShuffleSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Config ----------
SEED = 42
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_PATH = PROJECT / "data" / "mental-illnesses-prevalence.csv"
FIG_DIR = PROJECT / "figures"
REPORT_DIR = PROJECT / "report"
FIG_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = ["schizophrenia", "anxiety", "bipolar", "eating", "Year"]
TARGET = "depression"

COL_MAP = {
    "Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized": "schizophrenia",
    "Depressive disorders (share of population) - Sex: Both - Age: Age-standardized": "depression",
    "Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized": "anxiety",
    "Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized": "bipolar",
    "Eating disorders (share of population) - Sex: Both - Age: Age-standardized": "eating",
}

sns.set_theme(style="whitegrid", context="notebook")


# ---------- Helpers ----------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metric_dict(y_true, y_pred):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def load_data():
    raw = pd.read_csv(DATA_PATH)
    df = raw.rename(columns=COL_MAP)
    countries = df[df["Code"].notna()].copy().reset_index(drop=True)
    return countries


def build_pipelines():
    pipes = {
        "ElasticNet": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(random_state=SEED, max_iter=20000)),
            ]
        ),
        "RandomForest": Pipeline(
            [("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))]
        ),
        "HistGB": Pipeline(
            [("model", HistGradientBoostingRegressor(random_state=SEED))]
        ),
    }
    grids = {
        "ElasticNet": {
            "model__alpha": [0.001, 0.01, 0.1, 1.0],
            "model__l1_ratio": [0.1, 0.5, 0.9],
        },
        "RandomForest": {
            "model__n_estimators": [300, 600],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 5],
        },
        "HistGB": {
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [300, 600],
            "model__max_depth": [None, 6],
        },
    }
    return pipes, grids


def run_tuning(pipes, grids, X_tr, y_tr, g_tr):
    gkf = GroupKFold(n_splits=5)
    results = {}
    for name, pipe in pipes.items():
        gs = GridSearchCV(
            pipe,
            grids[name],
            scoring="neg_root_mean_squared_error",
            cv=gkf.split(X_tr, y_tr, g_tr),
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X_tr, y_tr)
        cv_rmse_mean = float(-gs.best_score_)
        # also compute the std of the best params across folds from cv_results_
        idx = gs.best_index_
        cv_rmse_std = float(gs.cv_results_["std_test_score"][idx])
        results[name] = {
            "best_params": gs.best_params_,
            "cv_rmse_mean": cv_rmse_mean,
            "cv_rmse_std": cv_rmse_std,
            "estimator": gs.best_estimator_,
        }
        print(
            f"  - {name:13s} best CV RMSE = {cv_rmse_mean:.4f} +/- {cv_rmse_std:.4f}"
        )
        print(f"      params = {gs.best_params_}")
    return results


def baseline_linear_cv(X_tr, y_tr, g_tr):
    """Regresion lineal sin regularizacion, mismo Pipeline, para comparar."""
    pipe = Pipeline(
        [("scaler", StandardScaler()), ("model", LinearRegression())]
    )
    gkf = GroupKFold(n_splits=5)
    scores = -cross_val_score(
        pipe,
        X_tr,
        y_tr,
        groups=g_tr,
        cv=gkf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    pipe.fit(X_tr, y_tr)
    return {
        "cv_rmse_mean": float(scores.mean()),
        "cv_rmse_std": float(scores.std()),
        "estimator": pipe,
    }


def dummy_cv(X_tr, y_tr, g_tr):
    pipe = DummyRegressor(strategy="mean")
    gkf = GroupKFold(n_splits=5)
    scores = -cross_val_score(
        pipe,
        X_tr,
        y_tr,
        groups=g_tr,
        cv=gkf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    pipe.fit(X_tr, y_tr)
    return {
        "cv_rmse_mean": float(scores.mean()),
        "cv_rmse_std": float(scores.std()),
        "estimator": pipe,
    }


def per_country_errors(test_df, y_true, y_pred):
    tmp = test_df.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    tmp["abs_err"] = (tmp["y_true"] - tmp["y_pred"]).abs()
    by_country = (
        tmp.groupby("Entity")["abs_err"].mean().sort_values(ascending=False)
    )
    return tmp, by_country


# ---------- Figures ----------
def fig_cv_comparison(cv_table: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = ["#999999", "#7f7f7f", "#3b6aa0", "#1f7a3f", "#a14b2a"]
    ax.bar(
        cv_table["model"],
        cv_table["cv_rmse_mean"],
        yerr=cv_table["cv_rmse_std"],
        capsize=4,
        color=palette[: len(cv_table)],
    )
    ax.set_ylabel("RMSE (CV por pais)")
    ax.set_title("Comparacion de familias - GroupKFold por pais (5 folds)")
    for i, row in cv_table.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["cv_rmse_mean"] + row["cv_rmse_std"] + 0.01,
            f"{row['cv_rmse_mean']:.3f}",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_test_metrics(test_metrics: dict, out: Path):
    df = pd.DataFrame(test_metrics).T.reset_index().rename(columns={"index": "model"})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, m, color in zip(axes, ["RMSE", "MAE", "R2"], ["#3b6aa0", "#1f7a3f", "#a14b2a"]):
        ax.bar(df["model"], df[m], color=color)
        ax.set_title(f"{m} en test (paises no vistos)")
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(df[m]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        if m == "R2":
            ax.axhline(0, color="red", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_residuals(y_true, y_preds: dict, out: Path):
    n = len(y_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, yhat) in zip(axes, y_preds.items()):
        resid = np.asarray(y_true) - np.asarray(yhat)
        ax.scatter(yhat, resid, s=10, alpha=0.45, color="#3b6aa0")
        ax.axhline(0, color="red", ls="--", lw=1)
        ax.set_title(f"Residuos - {name}")
        ax.set_xlabel("y_pred")
    axes[0].set_ylabel("y - y_pred")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_pred_vs_true(y_true, y_preds: dict, out: Path):
    n = len(y_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, yhat) in zip(axes, y_preds.items()):
        ax.scatter(y_true, yhat, s=10, alpha=0.45, color="#1f7a3f")
        lo = float(min(np.min(y_true), np.min(yhat)))
        hi = float(max(np.max(y_true), np.max(yhat)))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_title(f"y vs y_pred - {name}")
        ax.set_xlabel("y real")
    axes[0].set_ylabel("y predicho")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_permutation_importance(imp_mean, imp_std, features, model_name, out: Path):
    order = np.argsort(imp_mean)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(
        np.array(features)[order],
        imp_mean[order],
        xerr=imp_std[order],
        color="#3b6aa0",
    )
    ax.set_xlabel("Aumento del RMSE al permutar la feature")
    ax.set_title(f"Permutation importance - {model_name}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_top_errors(by_country: pd.Series, out: Path, k: int = 15):
    top = by_country.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.index, top.values, color="#a14b2a")
    ax.set_xlabel("MAE promedio del pais")
    ax.set_title(f"Top {k} paises con mas error (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------- Main ----------
def main():
    print(f"Cargando datos desde {DATA_PATH.relative_to(PROJECT)} ...")
    df = load_data()
    print(f"Filas: {len(df)} | Paises: {df['Entity'].nunique()}")

    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df["Entity"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]
    test_df = df.iloc[te_idx].reset_index(drop=True)

    print(
        f"Train: {len(X_tr)} filas ({pd.Series(g_tr).nunique()} paises) | "
        f"Test: {len(X_te)} filas ({pd.Series(g_te).nunique()} paises)"
    )

    # ---- Baselines ----
    print("\n>> Baselines (CV GroupKFold sobre train)")
    dum = dummy_cv(X_tr, y_tr, g_tr)
    print(
        f"  - Dummy (media)   CV RMSE = {dum['cv_rmse_mean']:.4f} +/- {dum['cv_rmse_std']:.4f}"
    )
    lin = baseline_linear_cv(X_tr, y_tr, g_tr)
    print(
        f"  - LinReg (Entr.1) CV RMSE = {lin['cv_rmse_mean']:.4f} +/- {lin['cv_rmse_std']:.4f}"
    )

    # ---- Tuning de familias ----
    print("\n>> Tuning con GridSearchCV (GroupKFold por pais)")
    pipes, grids = build_pipelines()
    tuned = run_tuning(pipes, grids, X_tr, y_tr, g_tr)

    # ---- Tabla resumen CV ----
    cv_rows = [
        {
            "model": "Dummy (media)",
            "cv_rmse_mean": dum["cv_rmse_mean"],
            "cv_rmse_std": dum["cv_rmse_std"],
        },
        {
            "model": "LinReg (baseline)",
            "cv_rmse_mean": lin["cv_rmse_mean"],
            "cv_rmse_std": lin["cv_rmse_std"],
        },
    ]
    for name, r in tuned.items():
        cv_rows.append(
            {
                "model": name,
                "cv_rmse_mean": r["cv_rmse_mean"],
                "cv_rmse_std": r["cv_rmse_std"],
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    print("\nTabla CV:")
    print(cv_table.to_string(index=False))

    # ---- Evaluacion en test ----
    print("\n>> Evaluacion en test (paises no vistos)")
    models_for_test = {
        "Dummy": dum["estimator"],
        "LinReg": lin["estimator"],
        **{k: v["estimator"] for k, v in tuned.items()},
    }
    test_metrics = {}
    test_preds = {}
    for name, est in models_for_test.items():
        # already fit on whole train above? cross_val_score does not refit
        # GridSearchCV with refit=True did fit the best estimator on all train.
        # dummy_cv and baseline_linear_cv explicitly fit.
        y_hat = est.predict(X_te)
        m = metric_dict(y_te, y_hat)
        test_metrics[name] = m
        test_preds[name] = y_hat
        print(
            f"  - {name:13s} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R2={m['R2']:.4f}"
        )

    # ---- Mejor familia segun CV ----
    best_name = min(
        tuned.keys(), key=lambda k: tuned[k]["cv_rmse_mean"]
    )
    best_est = tuned[best_name]["estimator"]
    print(f"\nMejor familia segun CV: {best_name}")

    # ---- Permutation importance ----
    print(f">> Permutation importance sobre test con {best_name} ...")
    pi = permutation_importance(
        best_est,
        X_te,
        y_te,
        scoring="neg_root_mean_squared_error",
        n_repeats=20,
        random_state=SEED,
        n_jobs=-1,
    )
    # pi.importances_mean = drop in score when feature is permuted.
    # With scoring='neg_root_mean_squared_error' that drop equals
    # (RMSE_permuted - RMSE_baseline). Positive = feature was important.
    imp_mean = pi.importances_mean
    imp_std = pi.importances_std

    # ---- Errores por pais con el mejor modelo ----
    test_with_pred, by_country = per_country_errors(
        test_df, y_te, test_preds[best_name]
    )

    # ---- Figuras ----
    print(">> Guardando figuras ...")
    fig_cv_comparison(cv_table, FIG_DIR / "08_cv_comparison.png")
    fig_test_metrics(test_metrics, FIG_DIR / "09_test_metrics.png")
    fig_residuals(
        y_te,
        {
            "LinReg": test_preds["LinReg"],
            "ElasticNet": test_preds["ElasticNet"],
            "RandomForest": test_preds["RandomForest"],
            "HistGB": test_preds["HistGB"],
        },
        FIG_DIR / "10_residuos_familias.png",
    )
    fig_pred_vs_true(
        y_te,
        {
            "ElasticNet": test_preds["ElasticNet"],
            "RandomForest": test_preds["RandomForest"],
            "HistGB": test_preds["HistGB"],
        },
        FIG_DIR / "11_pvt_familias.png",
    )
    fig_permutation_importance(
        imp_mean, imp_std, FEATURES, best_name, FIG_DIR / "12_permutation_importance.png"
    )
    fig_top_errors(by_country, FIG_DIR / "13_top_errores.png", k=15)

    # ---- Guardar metricas ----
    metrics_out = {
        "seed": SEED,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_countries_train": int(pd.Series(g_tr).nunique()),
        "n_countries_test": int(pd.Series(g_te).nunique()),
        "features": FEATURES,
        "target": TARGET,
        "cv_table": cv_table.to_dict(orient="records"),
        "tuned_best_params": {k: v["best_params"] for k, v in tuned.items()},
        "test_metrics": test_metrics,
        "best_family_cv": best_name,
        "permutation_importance": {
            "model": best_name,
            "features": FEATURES,
            "importance_mean": imp_mean.tolist(),
            "importance_std": imp_std.tolist(),
        },
        "top_error_countries": by_country.head(15).round(4).to_dict(),
        "test_y_stats": {
            "mean": float(np.mean(y_te)),
            "std": float(np.std(y_te)),
            "min": float(np.min(y_te)),
            "max": float(np.max(y_te)),
        },
    }
    out_json = REPORT_DIR / "entrega2_metrics.json"
    out_json.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    print(f"\nMetricas escritas en {out_json.relative_to(PROJECT)}")
    print("Listo.")


if __name__ == "__main__":
    main()
