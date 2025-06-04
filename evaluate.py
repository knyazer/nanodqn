from pathlib import Path
import pandas as pd
from scipy.stats import binomtest
import warnings

root_dir = Path("results")
csv_files = sorted(root_dir.rglob("*.csv"))

CONF = 0.95  # confidence level
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    n_all = len(df)
    if n_all == 0:
        warnings.warn(f"{csv_file.name} is empty – skipping.")
        continue

    n_weak = df["weak_convergence"].sum()
    p_hat = n_weak / n_all

    ci = binomtest(k=n_weak, n=n_all).proportion_ci(
        confidence_level=CONF,
        method="exact",  # since we have k ~= n, Clopper-Pearson is better
    )

    print(
        f"{csv_file.name}: \tp_weak = {p_hat:.3f}\t"
        f"({int(CONF * 100)}% CI [{ci.low:.3f}, {ci.high:.3f}])"
    )
