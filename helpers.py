import json
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import equinox as eqx

RUN_NAME = "final_run"


def default_col_cond(v):
    if isinstance(v, bool | int | float):
        return False
    if eqx.is_array(v) and len(v.shape) != 0:
        return True
    return False


def df_to(df: pd.DataFrame, filename: str, array_cols: Optional[Sequence[str]] = None) -> None:
    if array_cols is None:
        array_cols = [c for c in df.columns if default_col_cond(df.iloc[0][c])]

    df_tmp = df.copy()
    for col in array_cols:
        df_tmp[col] = df_tmp[col].apply(lambda a: json.dumps(a.tolist()))

    df_tmp.to_csv(filename, index=False)


def df_from(filename: str, array_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if array_cols is None:
        preview = pd.read_csv(filename, nrows=1)
        guessed = []
        for col in preview.columns:
            try:
                val = json.loads(preview.at[0, col])
                if isinstance(val, list):
                    guessed.append(col)
            except Exception:
                pass
        array_cols = guessed

    converters = {col: (lambda s: np.array(json.loads(s))) for col in array_cols}
    return pd.read_csv(filename, converters=converters)
