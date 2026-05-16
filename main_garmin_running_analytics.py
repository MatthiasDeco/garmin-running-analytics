
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from src import fgarmin, fsql, garmin_transformation, garmin_analysis

env = sys.argv[1] if len(sys.argv) > 1 else "dev"
load_dotenv(Path(__file__).parent / f".env.{env}")

try:
    fgarmin.fit_activitys_download(
        garmin_mail = str(os.getenv("garmin_mail")),
        garmin_password = str(os.getenv("garmin_password")),
        fit_carp_path = Path(os.getenv("fit_path")))    
except Exception as e:
    print(f"Error while dowloading activitys: {e}")
    pass

garmin_transformation.fit_to_sqlite(
    fit_carp_path = Path(os.getenv("fit_path")),
    FTP_bpm= int(os.getenv("FTP_bpm")),
    FTP_pace= float(os.getenv("FTP_pace")),
    FTP_rap= int(os.getenv("FTP_rap")),
    weight= int(os.getenv("weight")),
    ddbb_sqlite = Path(os.getenv("ddbb_sqlite")),
    processed_table= str(os.getenv("processed_table_sqlite")),
    summary_table= str(os.getenv("summary_table_sqlite")))

df_summary = fsql.sqlite_to_df(
    ddbb_sqlite = Path(os.getenv("ddbb_sqlite")),
    table_sqlite = str(os.getenv("summary_table_sqlite")),
    order_by = "activity_date", order_how = "DESC")
df_summary.to_csv(Path(os.getenv("output_summary_csv_path")), index=False)

df_processed = fsql.sqlite_to_df(
    ddbb_sqlite = Path(os.getenv("ddbb_sqlite")),
    table_sqlite = str(os.getenv("processed_table_sqlite")),
    order_by = "file_id", order_how = "DESC",
    limit = 100000)
df_processed.to_csv(Path(os.getenv("output_processed_csv_path")), index=False)

garmin_analysis.main_analysis(
    ddbb_sqlite = Path(os.getenv("ddbb_sqlite")),
    processed_table_sqlite = str(os.getenv("processed_table_sqlite")),
    summary_table_sqlite = str(os.getenv("summary_table_sqlite")))
