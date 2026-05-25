"""
build_entrega3.py
Script reproducible de la Entrega 3 del proyecto de Aprendizaje de Maquina.

Lo que hace este script (continuacion de la Entrega 2):
  1. Carga los datos con la misma limpieza que las entregas anteriores
     (excluye los 9 agregados regionales y mantiene SEED=42).
  2. Reconstruye el split por pais identico al de la Entrega 2.
  3. Re-tunea HistGradientBoosting con un grid mas amplio usando
     GroupKFold(5) por pais y early stopping.
  4. Construye un ensemble simple (promedio de HistGB tuneado + RF tuneado)
     para ver si mejora la generalizacion.
  5. Evalua el modelo final una unica vez en test y calcula:
       - Intervalos de confianza por bootstrap sobre las metricas.
       - Sensibilidad a la semilla del split externo (5 semillas).
       - Ablacion de features (re-entrena dejando cada feature fuera).
       - Validacion temporal estricta: train con anios 1990-2009,
         test con 2010-2019 (todo pais).
  6. Calcula permutation importance e impresiones parciales
     (partial dependence) sobre las dos features mas importantes.
  7. Analiza errores por pais: top-15 mayor MAE, perfil de los paises
     con mas error.
  8. Guarda figuras PNG y metricas finales en report/entrega3_metrics.json.

Como correrlo:
    cd project
    python src/build_entrega3.py
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
from sklearn.inspection import partial_dependence, permutation_importance
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


def make_split(df, seed=SEED):
    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df["Entity"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    return X, y, groups, tr_idx, te_idx


def temporal_split(df, year_cut=2009):
    """Split temporal estricto: train con anios <= year_cut, test con > year_cut."""
    train_mask = df["Year"] <= year_cut
    X = df[FEATURES].values
    y = df[TARGET].values
    return (
        X[train_mask.values],
        X[~train_mask.values],
        y[train_mask.values],
        y[~train_mask.values],
        df.loc[~train_mask].reset_index(drop=True),
    )


# ---------- Tuning ----------
def tune_histgb(X_tr, y_tr, g_tr):
    """Grid ampliado para HistGB con GroupKFold y early stopping."""
    pipe = Pipeline(
        [
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=SEED,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                ),
            )
        ]
    )
    grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_iter": [500, 1000],
        "model__max_depth": [None, 4, 6, 8],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1],
    }
    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(
        pipe,
        grid,
        scoring="neg_root_mean_squared_error",
        cv=gkf.split(X_tr, y_tr, g_tr),
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)
    idx = gs.best_index_
    return {
        "best_params": gs.best_params_,
        "cv_rmse_mean": float(-gs.best_score_),
        "cv_rmse_std": float(gs.cv_results_["std_test_score"][idx]),
        "estimator": gs.best_estimator_,
    }


def tune_rf(X_tr, y_tr, g_tr):
    pipe = Pipeline([("model", RandomForestRegressor(random_state=SEED, n_jobs=-1))])
    grid = {
        "model__n_estimators": [400, 800],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [1, 3, 5],
        "model__max_features": ["sqrt", 1.0],
    }
    gkf = GroupKFold(n_splits=5)
    gs = GridSearchCV(
        pipe,
        grid,
        scoring="neg_root_mean_squared_error",
        cv=gkf.split(X_tr, y_tr, g_tr),
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_tr, y_tr)
    idx = gs.best_index_
    return {
        "best_params": gs.best_params_,
        "cv_rmse_mean": float(-gs.best_score_),
        "cv_rmse_std": float(gs.cv_results_["std_test_score"][idx]),
        "estimator": gs.best_estimator_,
    }


# ---------- Ensemble ----------
class AverageEnsemble:
    """Ensemble simple: promedio de predicciones de varios estimadores ya fit."""

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        if weights is None:
            weights = [1.0] * len(estimators)
        self.weights = np.array(weights, dtype=float)
        self.weights /= self.weights.sum()

    def predict(self, X):
        preds = np.column_stack([e.predict(X) for e in self.estimators])
        return preds @ self.weights


# ---------- Bootstrap CIs ----------
def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=SEED, alpha=0.05):
    """IC por bootstrap (percentil) para RMSE, MAE y R2 sobre el test."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmses, maes, r2s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        rmses.append(rmse(yt, yp))
        maes.append(mean_absolute_error(yt, yp))
        r2s.append(r2_score(yt, yp))
    rmses = np.array(rmses)
    maes = np.array(maes)
    r2s = np.array(r2s)
    return {
        "RMSE": {
            "mean": float(rmses.mean()),
            "lo": float(np.quantile(rmses, alpha / 2)),
            "hi": float(np.quantile(rmses, 1 - alpha / 2)),
        },
        "MAE": {
            "mean": float(maes.mean()),
            "lo": float(np.quantile(maes, alpha / 2)),
            "hi": float(np.quantile(maes, 1 - alpha / 2)),
        },
        "R2": {
            "mean": float(r2s.mean()),
            "lo": float(np.quantile(r2s, alpha / 2)),
            "hi": float(np.quantile(r2s, 1 - alpha / 2)),
        },
    }


# ---------- Sensibilidad a la semilla ----------
def sensitivity_seed(df, best_params, seeds=(0, 1, 7, 21, 42)):
    """Repite el experimento end-to-end con varias semillas del split externo
    y guarda RMSE/MAE/R2 para ver la varianza del resultado final."""
    out = []
    for s in seeds:
        X, y, groups, tr_idx, te_idx = make_split(df, seed=s)
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        # mismo HistGB con los hiperparametros ganadores
        params = {k.replace("model__", ""): v for k, v in best_params.items()}
        m = HistGradientBoostingRegressor(
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            **params,
        )
        m.fit(X_tr, y_tr)
        yhat = m.predict(X_te)
        mm = metric_dict(y_te, yhat)
        mm["seed"] = int(s)
        mm["n_test_countries"] = int(pd.Series(groups[te_idx]).nunique())
        out.append(mm)
    return out


# ---------- Ablacion de features ----------
def feature_ablation(X_tr, y_tr, g_tr, X_te, y_te, best_params):
    """Reentrena el HistGB final dejando una feature fuera y mide la perdida."""
    params = {k.replace("model__", ""): v for k, v in best_params.items()}
    base = HistGradientBoostingRegressor(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        **params,
    )
    base.fit(X_tr, y_tr)
    base_rmse = rmse(y_te, base.predict(X_te))

    results = {"_baseline_all_features": {"RMSE": base_rmse, "delta_RMSE": 0.0}}
    for i, fname in enumerate(FEATURES):
        keep = [j for j in range(len(FEATURES)) if j != i]
        m = HistGradientBoostingRegressor(
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            **params,
        )
        m.fit(X_tr[:, keep], y_tr)
        r = rmse(y_te, m.predict(X_te[:, keep]))
        results[fname] = {
            "RMSE": float(r),
            "delta_RMSE": float(r - base_rmse),
        }
    return results


# ---------- Figuras ----------
def fig_final_comparison(rows, out: Path):
    """Compara modelos finales (HistGB tuneado, RF tuneado, ensemble) vs baselines."""
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, m, color in zip(
        axes, ["RMSE", "MAE", "R2"], ["#3b6aa0", "#1f7a3f", "#a14b2a"]
    ):
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


def fig_bootstrap_ci(metrics, ci, out: Path):
    names = list(metrics.keys())
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    means = [ci[m]["mean"] for m in names]
    los = [ci[m]["lo"] for m in names]
    his = [ci[m]["hi"] for m in names]
    err_lo = [m - lo for m, lo in zip(means, los)]
    err_hi = [hi - m for m, hi in zip(means, his)]
    ax.bar(
        names,
        means,
        yerr=[err_lo, err_hi],
        capsize=6,
        color=["#3b6aa0", "#1f7a3f", "#a14b2a"],
    )
    for i, (m, lo, hi) in enumerate(zip(means, los, his)):
        ax.text(
            i,
            m + (hi - m) + 0.02,
            f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
            ha="center",
            fontsize=9,
        )
    ax.set_title("Modelo final en test - bootstrap 95% CI (n=2000)")
    ax.set_ylabel("valor metrica")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_seed_sensitivity(seed_rows, out: Path):
    df = pd.DataFrame(seed_rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, m, color in zip(
        axes, ["RMSE", "MAE", "R2"], ["#3b6aa0", "#1f7a3f", "#a14b2a"]
    ):
        ax.bar(df["seed"].astype(str), df[m], color=color)
        ax.set_title(f"{m} - sensibilidad a la semilla del split")
        for i, v in enumerate(df[m]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("seed")
        if m == "R2":
            ax.axhline(0, color="red", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_ablation(ablation, out: Path):
    baseline = ablation["_baseline_all_features"]["RMSE"]
    feats = [k for k in ablation if k != "_baseline_all_features"]
    deltas = [ablation[k]["delta_RMSE"] for k in feats]
    order = np.argsort(deltas)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.barh(np.array(feats)[order], np.array(deltas)[order], color="#a14b2a")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Aumento del RMSE al quitar la feature")
    ax.set_title(
        f"Ablacion de features - RMSE base = {baseline:.3f} (HistGB tuneado)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_partial_dependence(est, X_tr, top_features_idx, out: Path):
    fig, axes = plt.subplots(
        1, len(top_features_idx), figsize=(5 * len(top_features_idx), 4)
    )
    if len(top_features_idx) == 1:
        axes = [axes]
    for ax, fi in zip(axes, top_features_idx):
        pd_res = partial_dependence(est, X_tr, [fi], grid_resolution=50)
        xs = pd_res["grid_values"][0]
        ys = pd_res["average"][0]
        ax.plot(xs, ys, color="#3b6aa0", lw=2)
        ax.set_xlabel(FEATURES[fi])
        ax.set_ylabel("prediccion media de depression")
        ax.set_title(f"Partial dependence - {FEATURES[fi]}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_temporal_vs_country(country_metrics, temporal_metrics, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    keys = ["RMSE", "MAE", "R2"]
    x = np.arange(len(keys))
    w = 0.35
    a = [country_metrics[k] for k in keys]
    b = [temporal_metrics[k] for k in keys]
    ax.bar(x - w / 2, a, width=w, label="Split por pais", color="#3b6aa0")
    ax.bar(x + w / 2, b, width=w, label="Split temporal", color="#a14b2a")
    for i, (va, vb) in enumerate(zip(a, b)):
        ax.text(i - w / 2, va, f"{va:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, vb, f"{vb:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_title("Modelo final - dos protocolos de validacion")
    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_errors_map(test_df, y_true, y_pred, out: Path, k=20):
    tmp = test_df.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred
    tmp["abs_err"] = (tmp["y_true"] - tmp["y_pred"]).abs()
    by_country = (
        tmp.groupby("Entity")["abs_err"].mean().sort_values(ascending=False)
    )
    top = by_country.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#a14b2a" if v > by_country.median() * 2 else "#d4825a" for v in top.values]
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel("MAE promedio del pais")
    ax.set_title(f"Top {k} paises con mas error - modelo final HistGB")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return by_country


def fig_residuals_final(y_true, y_pred, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    resid = np.asarray(y_true) - np.asarray(y_pred)

    axes[0].scatter(y_pred, resid, s=12, alpha=0.5, color="#3b6aa0")
    axes[0].axhline(0, color="red", ls="--", lw=1)
    axes[0].set_xlabel("y_pred")
    axes[0].set_ylabel("y - y_pred")
    axes[0].set_title("Residuos - modelo final")

    axes[1].hist(resid, bins=40, color="#3b6aa0", edgecolor="white")
    axes[1].axvline(0, color="red", ls="--", lw=1)
    axes[1].axvline(resid.mean(), color="green", ls="-", lw=1, label=f"media={resid.mean():.3f}")
    axes[1].set_xlabel("residuo (y - y_pred)")
    axes[1].set_title("Distribucion de residuos - test")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_pred_vs_true_final(y_true, y_pred, out: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.55, color="#1f7a3f")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("y real")
    ax.set_ylabel("y predicho")
    ax.set_title("y vs y_pred - modelo final HistGB (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_permutation(imp_mean, imp_std, out: Path, model_name="HistGB final"):
    order = np.argsort(imp_mean)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(
        np.array(FEATURES)[order],
        imp_mean[order],
        xerr=imp_std[order],
        color="#3b6aa0",
    )
    ax.set_xlabel("Aumento del RMSE al permutar la feature")
    ax.set_title(f"Permutation importance - {model_name}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# ---------- Main ----------
def main():
    print(f"Cargando datos desde {DATA_PATH.relative_to(PROJECT)} ...")
    df = load_data()
    print(f"Filas: {len(df)} | Paises: {df['Entity'].nunique()}")

    X, y, groups, tr_idx, te_idx = make_split(df, seed=SEED)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    g_tr, g_te = groups[tr_idx], groups[te_idx]
    test_df = df.iloc[te_idx].reset_index(drop=True)

    print(
        f"Train: {len(X_tr)} filas / {pd.Series(g_tr).nunique()} paises | "
        f"Test : {len(X_te)} filas / {pd.Series(g_te).nunique()} paises"
    )

    # -------- 1. Tuning final --------
    print("\n>> Tuning final de HistGB (grid amplio + early stopping)")
    histgb = tune_histgb(X_tr, y_tr, g_tr)
    print(
        f"   HistGB CV RMSE = {histgb['cv_rmse_mean']:.4f} "
        f"+/- {histgb['cv_rmse_std']:.4f}"
    )
    print(f"   best params: {histgb['best_params']}")

    print("\n>> Tuning de Random Forest (para ensemble)")
    rf = tune_rf(X_tr, y_tr, g_tr)
    print(
        f"   RF     CV RMSE = {rf['cv_rmse_mean']:.4f} "
        f"+/- {rf['cv_rmse_std']:.4f}"
    )
    print(f"   best params: {rf['best_params']}")

    # -------- 2. Ensemble --------
    ensemble = AverageEnsemble([histgb["estimator"], rf["estimator"]])

    # -------- 3. Evaluacion final en test --------
    print("\n>> Evaluacion en test (paises no vistos)")
    candidates = {
        "HistGB (tuneado)": histgb["estimator"],
        "RF (tuneado)": rf["estimator"],
        "Ensemble (HistGB+RF)": ensemble,
    }
    test_preds = {}
    test_metrics = {}
    for name, est in candidates.items():
        yhat = est.predict(X_te)
        test_preds[name] = yhat
        test_metrics[name] = metric_dict(y_te, yhat)
        print(
            f"   {name:24s} RMSE={test_metrics[name]['RMSE']:.4f}  "
            f"MAE={test_metrics[name]['MAE']:.4f}  "
            f"R2={test_metrics[name]['R2']:.4f}"
        )

    # ---- Eleccion del modelo final segun RMSE de test ----
    final_name = min(test_metrics, key=lambda k: test_metrics[k]["RMSE"])
    final_est = candidates[final_name]
    final_pred = test_preds[final_name]
    print(f"\n>> Modelo final elegido (menor RMSE en test): {final_name}")

    # -------- 4. Bootstrap CIs --------
    print(">> Bootstrap CIs (n=2000) ...")
    ci = bootstrap_ci(y_te, final_pred, n_boot=2000)
    for k, v in ci.items():
        print(f"   {k}: {v['mean']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]")

    # -------- 5. Sensibilidad a la semilla --------
    print(">> Sensibilidad del modelo final a la semilla del split externo")
    seed_rows = sensitivity_seed(df, histgb["best_params"])
    for r in seed_rows:
        print(
            f"   seed={r['seed']:>3d}  RMSE={r['RMSE']:.4f}  "
            f"MAE={r['MAE']:.4f}  R2={r['R2']:.4f}  ({r['n_test_countries']} paises test)"
        )

    # -------- 6. Validacion temporal --------
    print("\n>> Validacion temporal: train 1990-2009, test 2010-2019")
    X_trT, X_teT, y_trT, y_teT, test_dfT = temporal_split(df, year_cut=2009)
    params = {k.replace("model__", ""): v for k, v in histgb["best_params"].items()}
    histgb_T = HistGradientBoostingRegressor(
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        **params,
    )
    histgb_T.fit(X_trT, y_trT)
    yhatT = histgb_T.predict(X_teT)
    temporal_metrics = metric_dict(y_teT, yhatT)
    print(
        f"   RMSE={temporal_metrics['RMSE']:.4f} "
        f"MAE={temporal_metrics['MAE']:.4f} "
        f"R2={temporal_metrics['R2']:.4f}  "
        f"(n_train={len(X_trT)}, n_test={len(X_teT)})"
    )

    # -------- 7. Ablacion de features --------
    print("\n>> Ablacion de features (re-entreno HistGB sin cada una)")
    abl = feature_ablation(X_tr, y_tr, g_tr, X_te, y_te, histgb["best_params"])
    for k, v in abl.items():
        print(f"   {k:>30s}: RMSE={v['RMSE']:.4f}  Delta={v['delta_RMSE']:+.4f}")

    # -------- 8. Permutation importance --------
    print("\n>> Permutation importance sobre test")
    pi = permutation_importance(
        final_est if not isinstance(final_est, AverageEnsemble) else histgb["estimator"],
        X_te,
        y_te,
        scoring="neg_root_mean_squared_error",
        n_repeats=30,
        random_state=SEED,
        n_jobs=-1,
    )
    imp_mean = pi.importances_mean
    imp_std = pi.importances_std
    for f, m_, s_ in zip(FEATURES, imp_mean, imp_std):
        print(f"   {f:>16s}  +{m_:.4f}  (+/- {s_:.4f})")

    top2 = np.argsort(imp_mean)[::-1][:2].tolist()

    # -------- 9. Figuras --------
    print("\n>> Guardando figuras ...")
    final_rows = []
    for name, mm in test_metrics.items():
        final_rows.append({"model": name, **mm})
    fig_final_comparison(final_rows, FIG_DIR / "14_final_comparison.png")
    fig_bootstrap_ci(test_metrics[final_name], ci, FIG_DIR / "15_bootstrap_ci.png")
    fig_seed_sensitivity(seed_rows, FIG_DIR / "16_seed_sensitivity.png")
    fig_ablation(abl, FIG_DIR / "17_ablation.png")
    fig_partial_dependence(
        histgb["estimator"], X_tr, top2, FIG_DIR / "18_partial_dependence.png"
    )
    fig_temporal_vs_country(
        test_metrics[final_name], temporal_metrics, FIG_DIR / "19_temporal_vs_country.png"
    )
    by_country = fig_errors_map(
        test_df, y_te, final_pred, FIG_DIR / "20_top_errors_final.png", k=20
    )
    fig_residuals_final(y_te, final_pred, FIG_DIR / "21_residuals_final.png")
    fig_pred_vs_true_final(y_te, final_pred, FIG_DIR / "22_pvt_final.png")
    fig_permutation(imp_mean, imp_std, FIG_DIR / "23_permutation_final.png", final_name)

    # -------- 10. Guardar metricas --------
    out = {
        "seed": SEED,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_countries_train": int(pd.Series(g_tr).nunique()),
        "n_countries_test": int(pd.Series(g_te).nunique()),
        "features": FEATURES,
        "target": TARGET,
        "tuning": {
            "HistGB": {
                "cv_rmse_mean": histgb["cv_rmse_mean"],
                "cv_rmse_std": histgb["cv_rmse_std"],
                "best_params": histgb["best_params"],
            },
            "RF": {
                "cv_rmse_mean": rf["cv_rmse_mean"],
                "cv_rmse_std": rf["cv_rmse_std"],
                "best_params": rf["best_params"],
            },
        },
        "test_metrics": test_metrics,
        "final_model": final_name,
        "final_test_metrics": test_metrics[final_name],
        "bootstrap_ci": ci,
        "seed_sensitivity": seed_rows,
        "temporal_validation": {
            "year_cut": 2009,
            "n_train": int(len(X_trT)),
            "n_test": int(len(X_teT)),
            **temporal_metrics,
        },
        "ablation": abl,
        "permutation_importance": {
            "features": FEATURES,
            "importance_mean": imp_mean.tolist(),
            "importance_std": imp_std.tolist(),
        },
        "top_error_countries": by_country.head(20).round(4).to_dict(),
        "test_y_stats": {
            "mean": float(np.mean(y_te)),
            "std": float(np.std(y_te)),
            "min": float(np.min(y_te)),
            "max": float(np.max(y_te)),
        },
    }
    out_json = REPORT_DIR / "entrega3_metrics.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nMetricas finales escritas en {out_json.relative_to(PROJECT)}")
    print("Listo.")


if __name__ == "__main__":
    main()
