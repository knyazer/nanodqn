from pathlib import Path
import pandas as pd

root_dir = Path("results")
csv_files = list(root_dir.rglob("*.csv"))
csv_files.sort()

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    n_all = len(df)
    n_weak_converged = len(df[df["weak_convergence"]])
    print(f"For {csv_file} p_weak={n_weak_converged / n_all}")
