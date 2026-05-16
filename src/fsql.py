import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional

def get_existing_ids(ddbb_sqlite: Path, table: str) -> set:
    if not ddbb_sqlite.exists():
        return set()
    with sqlite3.connect(ddbb_sqlite) as conn:
        cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if not cursor.fetchone():
            return set()
        rows = conn.execute(f"SELECT file_id FROM {table}").fetchall()
    return {row[0] for row in rows}


def upload_df(df: pd.DataFrame, ddbb_sqlite: Path, table: str):
    ddbb_sqlite.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ddbb_sqlite) as conn:
        cursor = conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            df.to_sql(table, conn, if_exists="replace", index=False)
            return

        existing_ids = {row[0] for row in conn.execute(f"SELECT file_id FROM {table}").fetchall()}
        df_new = df[~df["file_id"].isin(existing_ids)]
        if df_new.empty:
            return

        existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        df_new = df_new[[c for c in df_new.columns if c in existing_cols]]
        df_new.to_sql(table, conn, if_exists="append", index=False)


def sqlite_to_df(ddbb_sqlite: Path, table_sqlite: str,
                columns: list[str] | None = None,
                filters: dict | None = None,
                order_by: str | None = None,
                order_how: str = "ASC",
                limit: int | None = None) -> pd.DataFrame:
    
    cols = ", ".join(columns) if columns else "*"
    query = f"SELECT {cols} FROM {table_sqlite}"
    params = []

    if filters:
        conditions = []
        for col, val in filters.items():
            if isinstance(val, list):
                placeholders = ", ".join("?" * len(val))
                conditions.append(f"{col} IN ({placeholders})")
                params.extend(val)
            elif isinstance(val, str) and val.strip()[0] in (">", "<", "!", "="):
                op, *rest = val.strip().split(maxsplit=1)
                conditions.append(f"{col} {op} ?")
                params.append(rest[0] if rest else val)
            else:
                conditions.append(f"{col} = ?")
                params.append(val)
        query += " WHERE " + " AND ".join(conditions)

    if order_by:
        query += f" ORDER BY {order_by} {order_how}"
    if limit:
        query += f" LIMIT {limit}"

    with sqlite3.connect(ddbb_sqlite) as conn:
        df = pd.read_sql(query, conn, params=params if params else None)
        print(f"===== {table_sqlite} exported to df =====")
    return df