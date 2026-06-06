import os
import sys
import numpy as np
from dotenv import load_dotenv
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from ETL_from_garmin.src import fsql


def clustering_type_activity(
    ddbb_sqlite: Path, 
    summary_table_sqlite: str, 
    road_k_clusters: int, 
    trail_k_clusters: int, 
    road_features: list,      # Features específicos para Road
    trail_features: list      # Features específicos para Trail
) -> tuple:
    """
    Clustering independiente para Road y Trail con features diferenciados.
    
    Args:
        ddbb_sqlite: Path a la base de datos SQLite
        summary_table_sqlite: Nombre de la tabla con los datos resumidos
        road_k_clusters: Número de clusters para Road
        trail_k_clusters: Número de clusters para Trail
        road_features: Lista de features para clustering de Road
        trail_features: Lista de features para clustering de Trail
    
    Returns:
        - df_global_with_clusters: todos los datos con columna 'cluster'
        - road_summary: resumen de clusters Road
        - trail_summary: resumen de clusters Trail
        - models: dict con los modelos persistidos (imputer, scaler, kmeans) para cada actividad
    """
    # --- Construir columns dinámicamente: file_id + activity_type + features (sin duplicados) ---
    all_features = list(set(road_features) | set(trail_features))
    columns_to_extract = ["file_id", "activity_type"] + all_features
    
    df_global = fsql.sqlite_to_df(
        ddbb_sqlite=ddbb_sqlite,
        table_sqlite=summary_table_sqlite,
        columns=columns_to_extract,
        filters={"activity_type": ["Road", "Trail"]}
    )
    
    # Datos de training para entrenar los modelos
    df_training = df_global.copy()
    # Eliminar filas donde fallen AL MENAS ONE de las features necesarias (para cada actividad)
    df_training = df_training.dropna(subset=all_features).copy()
    
    # --- Entrenar clustering independiente para Road y Trail con sus features ---
    def train_clustering(data, optimal_clusters, activity_name, features):
        if len(data) <= optimal_clusters:
            print(f"Not enough data for {optimal_clusters} clusters in {activity_name} (n={len(data)})")
            return None, None, None
        
        X = data[features]
        
        # Imputación + Escalado (MinMax)
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # KMeans
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=123, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        data_local = data.copy()
        data_local['Cluster'] = clusters
        
        cluster_summary = data_local.groupby('Cluster')[features].mean().T
        
        return data_local, cluster_summary, (imputer, scaler, kmeans)
    
    df_training_road = df_training[df_training['activity_type'] == 'Road'].copy()
    df_training_trail = df_training[df_training['activity_type'] == 'Trail'].copy()
    
    # Entrenar con features ESPECÍFICOS de cada actividad
    road_clusters, road_summary, road_models = train_clustering(
        df_training_road, road_k_clusters, "Road", road_features
    )
    trail_clusters, trail_summary, trail_models = train_clustering(
        df_training_trail, trail_k_clusters, "Trail", trail_features
    )
    
    # --- Asignar clusters a TODO los datos (training + test) ---
    def assign_clusters_to_all_data(df_all, road_models, trail_models, road_features, trail_features):
        if df_all.empty:
            return df_all
        
        df_with_clusters = df_all.copy()
        df_with_clusters['cluster'] = -1  # Inicializa
        
        # Asignación Road con SUS features
        if road_models is not None:
            df_road = df_with_clusters[df_with_clusters["activity_type"] == "Road"].copy()
            if not df_road.empty:
                road_imputer, road_scaler, road_kmeans = road_models
                expected_features = road_imputer.feature_names_in_
                
                # Asegurar que todas las features de Road están presentes
                for col in expected_features:
                    if col not in df_road.columns:
                        df_road[col] = np.nan
                
                df_road_features = df_road[expected_features]
                df_road_imputed = road_imputer.transform(df_road_features)
                df_road_scaled = road_scaler.transform(df_road_imputed)
                
                road_clusters_pred = road_kmeans.predict(df_road_scaled)
                df_with_clusters.loc[df_road.index, 'cluster'] = road_clusters_pred
        
        # Asignación Trail con SUS features
        if trail_models is not None:
            df_trail = df_with_clusters[df_with_clusters["activity_type"] == "Trail"].copy()
            if not df_trail.empty:
                trail_imputer, trail_scaler, trail_kmeans = trail_models
                expected_features = trail_imputer.feature_names_in_
                
                for col in expected_features:
                    if col not in df_trail.columns:
                        df_trail[col] = np.nan
                
                df_trail_features = df_trail[expected_features]
                df_trail_imputed = trail_imputer.transform(df_trail_features)
                df_trail_scaled = trail_scaler.transform(df_trail_imputed)
                
                trail_clusters_pred = trail_kmeans.predict(df_trail_scaled)
                df_with_clusters.loc[df_trail.index, 'cluster'] = trail_clusters_pred
        
        return df_with_clusters
    
    global_with_clusters = assign_clusters_to_all_data(
        df_global, road_models, trail_models, road_features, trail_features
    )
    
    # Calcular silhouette score en training para evaluar
    if road_models is not None and len(df_training_road) > 1:
        road_imputer, road_scaler, road_kmeans = road_models
        X_road = df_training_road[road_features]
        X_road_imputed = road_imputer.transform(X_road)
        X_road_scaled = road_scaler.transform(X_road_imputed)
        road_sil = silhouette_score(X_road_scaled, road_clusters['Cluster'])
        print(f"Road silhouette (training): {road_sil:.3f}")
    
    if trail_models is not None and len(df_training_trail) > 1:
        trail_imputer, trail_scaler, trail_kmeans = trail_models
        X_trail = df_training_trail[trail_features]
        X_trail_imputed = trail_imputer.transform(X_trail)
        X_trail_scaled = trail_scaler.transform(X_trail_imputed)
        trail_sil = silhouette_score(X_trail_scaled, trail_clusters['Cluster'])
        print(f"Trail silhouette (training): {trail_sil:.3f}")
    
    return global_with_clusters, road_summary, trail_summary, {"Road": road_models, "Trail": trail_models}


# --- MAIN ---
env = sys.argv[1] if len(sys.argv) > 1 else "dev"
load_dotenv(f".env.{env}")

global_with_clusters, road_summary, trail_summary, models = clustering_type_activity(
    ddbb_sqlite=Path(os.getenv("ddbb_sqlite")),
    summary_table_sqlite=str(os.getenv("summary_table_sqlite")),
    road_k_clusters=8,
    trail_k_clusters=5,
    road_features=["total_time", "total_distance", "moving_average_pace", "average_bpm", "deviation_pace"],
    trail_features=["total_time", "total_distance", "total_accumulated_positive_level", "average_rap", "average_bpm"]
)