
import sys
from pathlib import Path
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


from ETL_from_garmin.src import fsql


env = sys.argv[1] if len(sys.argv) > 1 else "dev"
load_dotenv(Path(__file__).parent / f".env.{env}")

def main_analysis(ddbb_sqlite: Path, processed_table_sqlite: str, summary_table_sqlite: str):

    df_summary_for_clustering = fsql.sqlite_to_df(
        ddbb_sqlite = ddbb_sqlite, table_sqlite = summary_table_sqlite, 
        columns = [
            "file_id", "activity_type",
            "total_distance", "total_time", "total_accumulated_positive_level", 
            "moving_average_pace", "average_bpm",
            "average_power", "standardized_power",
            "training_load", "PI"
        ],
        filters = {"activity_type": ["Road", "Trail"], "total_distance": "> 3"}
    )

    CLUSTER_FEATURES = [
        "total_distance", "total_time", "total_accumulated_positive_level", 
        "moving_average_pace", "average_bpm", 
        "average_power", "standardized_power", "training_load", "PI"]
    
    df_summary_for_clustering = df_summary_for_clustering.dropna(subset=CLUSTER_FEATURES).copy()
    X = df_summary_for_clustering[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    df_summary_for_clustering["cluster"] = kmeans.fit_predict(X_scaled)
    df_clustered = df_summary_for_clustering.copy()

    sil = silhouette_score(X_scaled, df_summary_for_clustering["cluster"])
    print(f"Silhouette score (k): {sil:.3f}")
    print(df_clustered.groupby("cluster")[CLUSTER_FEATURES].mean().round(2))

    return df_clustered
