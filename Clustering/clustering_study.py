import os
import sys
from dotenv import load_dotenv
import math
import random
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from ETL_from_garmin.src import fsql

# ──────────────────────────────────────────────────────────────────────────────
# Núcleo de evaluación (debe ser top-level para joblib)
# ──────────────────────────────────────────────────────────────────────────────

def _eval_combo(df_values: np.ndarray, col_indices: tuple, k: int, random_state: int = 123) -> float:
    """
    Evalúa silhouette para una combinación de columnas (por índice) y k.
    Trabaja sobre numpy array pre-imputado y pre-escalado para máxima velocidad.
    Retorna -1.0 ante cualquier fallo.
    """
    X = df_values[:, col_indices]
    if X.shape[0] < k + 1:
        return -1.0
    try:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=3, max_iter=100).fit_predict(X)
        if len(np.unique(labels)) < 2:
            return -1.0
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0


def _eval_combo_all_k(args):
    """Wrapper para joblib: evalúa una combo sobre todos los k del rango."""
    df_values, col_indices, k_list, random_state = args
    best_sil, best_k = -1.0, k_list[0]
    for k in k_list:
        sil = _eval_combo(df_values, col_indices, k, random_state)
        if sil > best_sil:
            best_sil, best_k = sil, k
    return col_indices, best_k, best_sil


# ──────────────────────────────────────────────────────────────────────────────
# Warm-start: ranking individual de features por silhouette medio
# ──────────────────────────────────────────────────────────────────────────────

def _rank_features_individually(
    df_values: np.ndarray,
    valid_features: list[str],
    k_list: list[int],
    random_state: int = 123
) -> list[str]:
    """
    Puntúa cada feature por su silhouette medio sobre los k del rango
    (con las otras features como contexto mínimo: usa todas juntas y mide
    la contribución marginal por permutation importance simplificado).

    En la práctica: silhouette de cada feature sola × todos los k → ranking.
    Esto reordena las candidatas para que las combinaciones prometedoras
    aparezcan antes en itertools.combinations → early termination más eficaz.
    """
    scores = {}
    n = df_values.shape[0]
    for i, feat in enumerate(valid_features):
        X = df_values[:, [i]]
        sil_sum = 0.0
        count = 0
        for k in k_list:
            if n < k + 1:
                continue
            try:
                labels = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit_predict(X)
                if len(np.unique(labels)) >= 2:
                    sil_sum += silhouette_score(X, labels)
                    count += 1
            except Exception:
                pass
        scores[feat] = sil_sum / count if count > 0 else -1.0

    ranked = sorted(scores, key=lambda f: -scores[f])
    return ranked


# ──────────────────────────────────────────────────────────────────────────────
# Búsqueda exhaustiva paralelizada sobre exactamente N_FEATURES
# ──────────────────────────────────────────────────────────────────────────────

def _exhaustive_search_parallel(
    df: pd.DataFrame,
    valid_features: list[str],
    k_list: list[int],
    n_features: int,
    max_combos: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 123,
    verbose: bool = True
) -> dict:
    """
    Evalúa todas (o una muestra aleatoria) las combinaciones de exactamente
    `n_features` features sobre todos los k en k_list, en paralelo.

    Args:
        max_combos: Si se especifica, muestrea aleatoriamente ese número de
                    combinaciones en lugar de evaluar las 125.970. Útil para
                    exploración rápida. None = exhaustivo completo.

    Returns: dict con optimal_features, optimal_k, best_silhouette, top_results.
    """
    # Pre-procesar datos una sola vez (impute + scale sobre todas las features)
    X_raw = df[valid_features].copy()
    imputer = SimpleImputer(strategy="mean")
    scaler  = MinMaxScaler()
    X_proc  = scaler.fit_transform(imputer.fit_transform(X_raw))

    # Warm-start: reordenar features por relevancia individual
    if verbose:
        print("  [Warm-start] Ranking individual de features...")
    ranked_features = _rank_features_individually(X_proc, valid_features, k_list, random_state)
    ranked_indices  = [valid_features.index(f) for f in ranked_features]

    if verbose:
        print(f"  Features ordenadas por relevancia: {ranked_features}")

    # Generar todas las combinaciones (sobre índices reordenados)
    all_combos = list(itertools.combinations(ranked_indices, n_features))
    total_combos = len(all_combos)

    if max_combos and max_combos < total_combos:
        # Muestra estratificada: primero las combinaciones con features top-ranked
        # (las primeras de itertools.combinations ya tienen mejor ranking por warm-start)
        # Tomamos las primeras max_combos/2 + aleatorias del resto
        n_top  = max_combos // 2
        n_rand = max_combos - n_top
        sampled = all_combos[:n_top] + random.sample(all_combos[n_top:], min(n_rand, len(all_combos) - n_top))
        combos_to_eval = sampled
        if verbose:
            print(f"  Sampling: {max_combos:,} / {total_combos:,} combinaciones "
                  f"({max_combos/total_combos*100:.1f}%) | {n_top} top + {n_rand} aleatorias")
    else:
        combos_to_eval = all_combos
        if verbose:
            print(f"  Evaluando {total_combos:,} combinaciones × {len(k_list)} ks "
                  f"= {total_combos * len(k_list):,} evaluaciones")

    # Evaluar en paralelo con barra de progreso
    args_list = [(X_proc, combo, k_list, random_state) for combo in combos_to_eval]
    n_total = len(args_list)

    results_raw = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_eval_combo_all_k)(args)
        for args in tqdm(
            args_list,
            total=n_total,
            desc=f"  Evaluando combos",
            unit="combo",
            ncols=80,
            bar_format="{l_bar}{bar}| {n:,}/{total:,} [{elapsed}<{remaining}, {rate_fmt}]"
        )
    )

    # Procesar resultados
    results = [
        {
            "features": [valid_features[i] for i in col_indices],
            "k": best_k,
            "silhouette": best_sil
        }
        for col_indices, best_k, best_sil in results_raw
        if best_sil > -1.0
    ]

    results.sort(key=lambda x: -x["silhouette"])

    if not results:
        return {"optimal_features": [], "optimal_k": k_list[0], "best_silhouette": -1.0, "top_results": []}

    best = results[0]
    return {
        "optimal_features": best["features"],
        "optimal_k":        best["k"],
        "best_silhouette":  best["silhouette"],
        "top_results":      results[:50]   # top-50 para análisis posterior
    }


# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento del modelo final
# ──────────────────────────────────────────────────────────────────────────────

def _fit_optimal_model(df: pd.DataFrame, features: list[str], k: int, random_state: int = 123) -> dict:
    X = df[features].copy()
    valid_idx = X.dropna(thresh=len(features) // 2 + 1).index
    X = X.loc[valid_idx]

    imputer = SimpleImputer(strategy="mean")
    scaler  = MinMaxScaler()
    X_proc  = scaler.fit_transform(imputer.fit_transform(X))

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_proc)

    df_clustered = df.loc[valid_idx].copy()
    df_clustered["cluster"] = labels

    cluster_summary = df_clustered.groupby("cluster")[features].mean().round(3)

    return {
        "k": k,
        "features": features,
        "silhouette": silhouette_score(X_proc, labels),
        "imputer": imputer,
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_summary": cluster_summary,
        "df_clustered": df_clustered
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ──────────────────────────────────────────────────────────────────────────────

def _plot_top_results(top_results: list[dict], activity: str, output_path: str, top_n: int = 20):
    """Barplot de las top-N combinaciones por silhouette."""
    df_top = pd.DataFrame(top_results[:top_n])
    df_top["label"] = df_top.apply(
        lambda r: f"k={r['k']} | " + ", ".join(r["features"][:3]) + "...", axis=1
    )

    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.4)))
    bars = ax.barh(df_top["label"][::-1], df_top["silhouette"][::-1],
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)))
    ax.set_xlabel("Silhouette Score", fontsize=12)
    ax.set_title(f"Top {top_n} combinaciones (8 features, k∈[4-8]) - {activity}",
                 fontsize=13, weight="bold")
    ax.axvline(df_top["silhouette"].max(), color="red", linestyle="--", alpha=0.5, label="Best")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_path}/top_combos_{activity}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_feature_frequency(top_results: list[dict], activity: str, output_path: str, top_n: int = 50):
    """Frecuencia de aparición de cada feature en el top-N de combinaciones."""
    from collections import Counter
    counter = Counter(f for r in top_results[:top_n] for f in r["features"])
    df_freq = pd.DataFrame(counter.most_common(), columns=["feature", "count"])

    fig, ax = plt.subplots(figsize=(10, max(5, len(df_freq) * 0.4)))
    ax.barh(df_freq["feature"][::-1], df_freq["count"][::-1],
            color=plt.cm.plasma(np.linspace(0.2, 0.85, len(df_freq))))
    ax.set_xlabel(f"Apariciones en top-{top_n} combinaciones", fontsize=12)
    ax.set_title(f"Feature Importance por Frecuencia - {activity}", fontsize=13, weight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_frequency_{activity}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cluster_heatmap(cluster_summary: pd.DataFrame, activity: str, k: int, output_path: str):
    summary_norm = (cluster_summary - cluster_summary.mean()) / (cluster_summary.std() + 1e-9)

    plt.figure(figsize=(max(10, len(cluster_summary.columns) * 0.85), 5))
    sns.heatmap(summary_norm, annot=cluster_summary.values, fmt=".2f",
                cmap="RdYlGn", linewidths=0.5,
                annot_kws={"size": 9},
                cbar_kws={"label": "Z-score (valor real en celda)"})
    plt.title(f"Cluster Summary - {activity} (k={k})", fontsize=14, weight="bold")
    plt.xlabel("Feature", fontsize=11)
    plt.ylabel("Cluster", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_path}/heatmap_optimal_{activity}.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_scatter_grid(df_clustered: pd.DataFrame, features: list[str], k: int,
                       activity: str, output_path: str, max_pairs: int = 6):
    pairs = list(itertools.combinations(features[:5], 2))[:max_pairs]
    n_cols = min(3, len(pairs))
    n_rows = (len(pairs) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten() if len(pairs) > 1 else [axes]
    colors = plt.cm.viridis(np.linspace(0, 1, k))

    for idx, (fx, fy) in enumerate(pairs):
        ax = axes_flat[idx]
        for cid in range(k):
            mask = df_clustered["cluster"] == cid
            ax.scatter(df_clustered.loc[mask, fx], df_clustered.loc[mask, fy],
                       color=colors[cid], alpha=0.6, s=40,
                       edgecolors="black", linewidth=0.4, label=f"C{cid}")
        ax.set_xlabel(fx, fontsize=9, weight="bold")
        ax.set_ylabel(fy, fontsize=9, weight="bold")
        ax.set_title(f"{fx} vs {fy}", fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    for idx in range(len(pairs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Scatter Optimal Features - {activity} (k={k})", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_path}/scatter_optimal_{activity}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────────────────────────────────────

def study_optimal_clusters(
    ddbb_sqlite: Path,
    summary_table_sqlite: str,
    cluster_study_path: str,
    n_features: int,          # exactamente 8 features
    k_min: int,                    # k mínimo (4)
    k_max: int,   
    candidate_features: Optional[list[str]] = None,                 # k máximo (8)
    max_combos: Optional[int] = None,      # None = exhaustivo; int = sampling
    n_jobs: int = -1,                      # -1 = todos los cores disponibles
    min_silhouette_threshold: float = 0.20,
    activities: list[str] = ["Road", "Trail"],
    random_state: int = 123,
    verbose: bool = True
) -> tuple[dict, dict]:
    """
    Busca exhaustivamente la combinación óptima de exactamente `n_features`
    features y k en [k_min..k_max] para KMeans, por tipo de actividad.

    Args:
        ddbb_sqlite:               Path a la BD SQLite.
        summary_table_sqlite:      Nombre de la tabla.
        cluster_study_path:        Directorio de salida (gráficos + CSVs).
        candidate_features:        Candidatas. None → usa CANDIDATE_FEATURES (20 features).
        n_features:                Tamaño exacto de cada combinación (default: 8).
        k_min / k_max:             Rango de k evaluado (default: 4-8).
        max_combos:                Límite de combinaciones evaluadas. None = todas.
                                   Recomendado para exploración rápida: 10_000.
                                   Para exhaustivo completo: None (puede tardar 10-30 min).
        n_jobs:                    Paralelismo joblib. -1 = todos los cores.
        min_silhouette_threshold:  Warning si el mejor silhouette está por debajo.
        activities:                Tipos de actividad a analizar.
        random_state:              Semilla de reproducibilidad.
        verbose:                   Logging de progreso.

    Returns:
        optimal_results: {
            activity: {
                'optimal_k':       int,
                'optimal_features': list[str],   # exactamente n_features
                'silhouette':      float,
                'n_samples':       int
            }
        }
        cluster_models: {
            activity: {
                'k', 'features', 'silhouette',
                'imputer', 'scaler', 'kmeans',   # pipeline reutilizable
                'cluster_summary': pd.DataFrame,
                'df_clustered':    pd.DataFrame
            }
        }
    """
    os.makedirs(cluster_study_path, exist_ok=True)
    k_list = list(range(k_min, k_max + 1))
    features_pool = candidate_features or CANDIDATE_FEATURES

    total_combos = math.comb(len(features_pool), n_features)
    total_evals  = total_combos * len(k_list)
    effective    = min(max_combos, total_combos) if max_combos else total_combos

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Configuración de búsqueda")
        print(f"{'='*60}")
        print(f"  Features candidatas : {len(features_pool)}")
        print(f"  Features por combo  : {n_features}")
        print(f"  Rango k             : [{k_min}, {k_max}]")
        print(f"  Combinaciones total : {total_combos:,}")
        print(f"  Evaluaciones total  : {total_evals:,}")
        if max_combos:
            print(f"  Sampling activo     : {effective:,} combos ({effective/total_combos*100:.1f}%)")
        else:
            print(f"  Modo exhaustivo     : completo")
        print(f"  Paralelismo (jobs)  : {n_jobs}")

    # Carga de datos
    df_summary = fsql.sqlite_to_df(
        ddbb_sqlite=ddbb_sqlite,
        table_sqlite=summary_table_sqlite,
        columns=["file_id", "activity_type"] + features_pool,
        filters={"activity_type": activities}
    )

    optimal_results = {}
    cluster_models  = {}

    for activity in activities:
        df_activity = df_summary[df_summary["activity_type"] == activity].copy()

        if len(df_activity) < k_max + 1:
            print(f"⚠️  {activity}: insuficientes datos (n={len(df_activity)}, necesario ≥{k_max+1})")
            optimal_results[activity] = None
            cluster_models[activity]  = None
            continue

        # Features válidas para esta actividad (suficientes valores no-NaN)
        valid_features = [
            f for f in features_pool
            if f in df_activity.columns and df_activity[f].notna().sum() > 10
        ]

        if len(valid_features) < n_features:
            print(f"⚠️  {activity}: solo {len(valid_features)} features válidas, se necesitan {n_features}")
            optimal_results[activity] = None
            cluster_models[activity]  = None
            continue

        print(f"\n{'='*60}")
        print(f"  {activity}  (n={len(df_activity)}, features válidas={len(valid_features)})")
        print(f"{'='*60}")

        # ── Búsqueda ──
        search_result = _exhaustive_search_parallel(
            df=df_activity,
            valid_features=valid_features,
            k_list=k_list,
            n_features=n_features,
            max_combos=max_combos,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )

        optimal_features = search_result["optimal_features"]
        optimal_k        = search_result["optimal_k"]
        best_sil         = search_result["best_silhouette"]
        top_results      = search_result.get("top_results", [])

        print(f"\n  ✓ Resultado óptimo:")
        print(f"    k          = {optimal_k}")
        print(f"    Silhouette = {best_sil:.4f}", end="")
        if best_sil < min_silhouette_threshold:
            print(f"  ⚠️  [BAJO UMBRAL {min_silhouette_threshold}]")
        else:
            print()
        print(f"    Features   = {optimal_features}")

        # ── Modelo final ──
        model_info = _fit_optimal_model(df_activity, optimal_features, optimal_k, random_state)

        # CSVs
        model_info["cluster_summary"].to_csv(
            f"{cluster_study_path}/cluster_summary_{activity}.csv", index=True
        )
        model_info["df_clustered"].to_csv(
            f"{cluster_study_path}/df_clustered_{activity}.csv", index=False
        )
        if top_results:
            pd.DataFrame(top_results).to_csv(
                f"{cluster_study_path}/top_combos_{activity}.csv", index=False
            )

        # Gráficos
        if top_results:
            _plot_top_results(top_results, activity, cluster_study_path)
            _plot_feature_frequency(top_results, activity, cluster_study_path)
        _plot_cluster_heatmap(model_info["cluster_summary"], activity, optimal_k, cluster_study_path)
        _plot_scatter_grid(model_info["df_clustered"], optimal_features, optimal_k, activity, cluster_study_path)

        optimal_results[activity] = {
            "optimal_k":        optimal_k,
            "optimal_features": optimal_features,
            "silhouette":       best_sil,
            "n_samples":        len(model_info["df_clustered"])
        }
        cluster_models[activity] = model_info

    # Resumen global
    print(f"\n{'='*60}")
    print("  RESUMEN GLOBAL")
    print(f"{'='*60}")
    for activity, res in optimal_results.items():
        if res:
            print(f"  {activity}:")
            print(f"    k={res['optimal_k']} | sil={res['silhouette']:.4f} | n={res['n_samples']}")
            print(f"    features={res['optimal_features']}")

    return optimal_results, cluster_models







### MAIN
CANDIDATE_FEATURES = [
    "total_distance", "total_time", "total_accumulated_positive_level",
    "moving_average_pace", "average_rap", "deviation_pace",
    "average_bpm", "max_bpm", "deviation_bpm", "average_beatsxkm",
    "average_cadence", "average_stride", "total_energy",
    "average_power", "standardized_power",
    "training_load", "PI", "vdot", "intensity_factor"
]
env = sys.argv[1] if len(sys.argv) > 1 else "dev"
load_dotenv(f".env.{env}")
optimal_results, cluster_models = study_optimal_clusters(
    ddbb_sqlite = Path(os.getenv("ddbb_sqlite")),
    summary_table_sqlite = str(os.getenv("summary_table_sqlite")),
    cluster_study_path = str(os.getenv("cluster_study_path")),
    candidate_features = CANDIDATE_FEATURES,
    n_features = 8, k_min = 4, k_max = 8)
