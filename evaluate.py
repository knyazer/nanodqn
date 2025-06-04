from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import binomtest
import matplotlib.pyplot as plt

root = Path("results/01")
files = sorted(root.glob("*.csv"))

K, p, lo, hi = [], [], [], []
p_dqn = None

for f in files:
    df = pd.read_csv(f)
    k = df["weak_convergence"].sum()
    n = len(df)
    phat = k / n
    ci = binomtest(k, n).proportion_ci(method="exact")
    if "dqn" in f.stem:
        p_dqn = phat  # baseline for theory curve
    else:
        K.append(int(f.stem.split("boot")[1]))
        p.append(phat)
        lo.append(ci.low)
        hi.append(ci.high)

if p_dqn is None:
    raise RuntimeError("no dqn file found")

K = np.array(K)
p = np.array(p)
lo = np.array(lo)
hi = np.array(hi)
pred = 1 - (1.0 - p_dqn) ** K

plt.figure()
plt.fill_between(K, lo, hi, alpha=0.3)
plt.plot(K, p, "o-", label="empirical $p_{weak}$")
plt.plot(K, pred, "s--", label=r"$1-p_{weak,dqn}^K$")
plt.xlabel("$K$ (# bootstraps)")
plt.ylabel("$p_{weak}$")
plt.legend()
plt.tight_layout()

Path("plots/png").mkdir(parents=True, exist_ok=True)
plt.savefig("plots/01.svg")
plt.savefig("plots/png/01.png", dpi=300)
plt.close()
