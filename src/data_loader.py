from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text


@dataclass
class DBConfig:
    connection_string: str
    table: str = "robot_stream"


def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def add_robot_sources(df: pd.DataFrame, robot_count: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    The workshop requires >= 3 robots. Your CSV doesn't include robot_id.
    We replicate the dataset into N robots and add slight noise to axis signals per robot.
    """
    rng = np.random.default_rng(seed)

    axis_cols = [c for c in df.columns if c.startswith("Axis #")]
    out = []

    for r in range(1, robot_count + 1):
        copy_df = df.copy()
        copy_df["robot_id"] = f"robot_{r}"

        # Add gentle noise so robots aren't identical (keeps realism without destroying patterns)
        for col in axis_cols:
            if pd.api.types.is_numeric_dtype(copy_df[col]):
                noise = rng.normal(loc=0.0, scale=0.02, size=len(copy_df))
                copy_df[col] = copy_df[col].astype(float) + noise

        out.append(copy_df)

    return pd.concat(out, ignore_index=True)


def get_engine(db_config: DBConfig):
    return create_engine(db_config.connection_string, future=True)


def init_db_table(engine, table: str, df_sample: pd.DataFrame) -> None:
    """
    Creates a simple table schema using pandas to_sql (replace if your professor wants explicit DDL).
    """
    with engine.begin() as conn:
        # Create a fresh table if not exists by writing empty DF with same columns
        df_sample.head(0).to_sql(table, conn, if_exists="replace", index=False)


def load_df_to_db(engine, table: str, df: pd.DataFrame, chunk_size: int = 5000) -> None:
    df.to_sql(table, engine, if_exists="append", index=False, chunksize=chunk_size)


def load_from_db(engine, table: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(f"SELECT * FROM {table}"), conn)
