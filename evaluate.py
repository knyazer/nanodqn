from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from scipy.stats import binomtest
import yaml


def plot04():
    postfix = "04"
    root = Path(f"results/{postfix}")
    files = sorted(root.rglob("*.csv"))

    K, p, lo, hi = [], [], [], []
    base_p = base_lo = base_hi = None
    means = {}

    for f in files:
        df = pd.read_csv(f)
        phat = df["weak_convergence"].mean()
        ci = binomtest(df["weak_convergence"].sum(), len(df)).proportion_ci(method="exact")
        if "_dqn" in str(f) and base_p is None:
            base_p, base_lo, base_hi = phat, ci.low, ci.high
            means[1] = df["time_to_weak"].mean()
        elif "_boot" in str(f):
            k = int(str(f).split("boot")[1].split("_")[0][1:-1])
            K.append(k)
            p.append(phat)
            lo.append(ci.low)
            hi.append(ci.high)
            means[k] = df["time_to_weak"].mean()
        else:
            print(f"{f} is not a good path for plot04")

    if base_p is None:
        raise RuntimeError("no dqn baseline found")

    K.append(1)
    p.append(base_p)
    lo.append(base_lo)
    hi.append(base_hi)

    K = np.array(K)
    p = np.array(p)
    lo = np.array(lo)
    hi = np.array(hi)
    idx = np.argsort(K)
    K, p, lo, hi = K[idx], p[idx], lo[idx], hi[idx]

    pred = 1 - (1.0 - 0.007) ** K

    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # linear plot
    axes[0].fill_between(K, lo, hi, alpha=0.3)
    axes[0].plot(K, p, "o-", label="empirical $p_{weak}$")
    axes[0].plot(K, pred, "s--", label=r"$1 - (1 - p_{weak,dqn})^K$")
    axes[0].set_xlabel("$K$ (ensemble size)")
    axes[0].set_ylabel("Weak convergence probability")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].fill_between(K, lo, hi, alpha=0.3)
    axes[1].plot(K, p, "o-", label="empirical $1 - p_{weak}$")
    axes[1].plot(K, pred, "s--", label=r"$1 - (1 - p_{weak,dqn})^K$")

    axes[1].set_xlabel("$K$ (ensemble size)")
    axes[1].set_ylabel("Weak convergence log probability")
    axes[1].legend()
    axes[1].set_yscale("log")
    axes[1].grid(True)

    means_arr = []
    for key in sorted(means.keys()):
        means_arr.append(float(means[key]))
    axes[2].plot(sorted(means.keys()), means_arr)
    axes[2].grid(True)

    plt.tight_layout()

    Path("plots/png").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{postfix}.svg")
    plt.savefig(f"plots/png/{postfix}.png", dpi=300)
    plt.close()


def load(path: str | Path):
    folder = Path(path)
    with open(folder / ".version") as f:
        v = f.read().strip()
    if v != "1":
        raise ValueError(f"unsupported result log version {v}")
    df = pd.read_csv(folder / "results.csv")
    with open(folder / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    fields = [(key, type(value), dataclasses.field(default=value)) for key, value in cfg.items()]
    DynamicConfig = dataclasses.make_dataclass("DynamicConfig", fields)
    cfg_instance = DynamicConfig(**cfg)
    return df, cfg_instance


def make_agg(version):
    root = Path(f"results/{version}")
    paths = sorted(root.glob("*"))
    loaded = [load(path) for path in paths]
    agg = []
    for df, cfg in loaded:
        agg.append(
            {
                "mean_time_to_weak": df["time_to_weak"].mean(),
                "max_time_to_weak": df["time_to_weak"].max(),
                "mean_time_to_strong": df["time_to_strong"].mean(),
                "weak_convergence": (df["time_to_weak"] != df["time_to_weak"].max()).sum()
                / len(df),
                "ensemble_size": cfg.ensemble_size,
                "hardness": cfg.hardness,
                "kind": cfg.kind,
            }
        )
    return pd.DataFrame(agg)


def plot02():
    agg = make_agg("02")
    agg = pd.DataFrame(agg)
    ensemble_sizes = sorted(agg["ensemble_size"].unique())
    hardnesses = sorted(agg["hardness"].unique())

    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(ensemble_sizes)))

    for i, ensemble_size in enumerate(ensemble_sizes):
        v = agg.query(f"ensemble_size == {ensemble_size} and kind == 'boot'")
        if len(v) > 0:
            sorted_by_hardness = v.set_index("hardness", drop=True).sort_index()
            weak_arr = sorted_by_hardness["mean_time_to_weak"].array
            plt.plot(
                hardnesses,
                weak_arr,
                "o-",
                color=colors[i],
                label=f"Bootstrap K={ensemble_size}",
                linewidth=2,
            )

    v = agg.query("kind == 'dqn'")
    if len(v) > 0:
        dqn_weak = v.set_index("hardness", drop=True).sort_index()["mean_time_to_weak"].array
        plt.plot(
            hardnesses,
            dqn_weak,
            "s--",
            color="red",
            label="DQN baseline",
            linewidth=2,
            markersize=8,
        )

    limit = agg["max_time_to_weak"].max()
    plt.axhline(y=limit, color="gray", linestyle=":", alpha=0.7, label=f"Max time limit ({limit})")

    plt.xlabel("Environment Hardness")
    plt.ylabel("Mean Time to Weak Convergence")
    plt.title("Convergence Time vs Environment Hardness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    Path("plots/png").mkdir(parents=True, exist_ok=True)
    plt.savefig("plots/02.svg")
    plt.savefig("plots/png/02.png", dpi=300)
    plt.close()


def plot14(agg: pd.DataFrame):
    # Rename for clarity if needed
    if "weak_convergence" in agg.columns and "weak_probability" not in agg.columns:
        agg = agg.rename(columns={"weak_convergence": "weak_probability"})

    # Unique sorted values for K and n
    Ks = sorted(agg["ensemble_size"].unique())
    ns = sorted(agg["hardness"].unique())

    # 1. Fixed K, plot weak_probability vs. n
    fig1, axes1 = plt.subplots(nrows=(len(Ks) + 2) // 3, ncols=3, figsize=(15, 10))
    axes1 = axes1.flatten()
    for i, K in enumerate(Ks):
        ax = axes1[i]
        subdf = agg[agg["ensemble_size"] == K]
        for kind in subdf["kind"].unique():
            plot_df = subdf[subdf["kind"] == kind].sort_values(by="hardness")
            ax.plot(plot_df["hardness"], plot_df["weak_probability"], label=kind, marker="o")
        ax.set_title(f"K={K}")
        ax.set_xlabel("hardness")
        ax.set_ylabel("weak_probability")
        ax.grid(True)
        ax.legend()
    fig1.suptitle("Weak Probability vs. n (fixed K)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("plots/14_a.svg")
    plt.savefig("plots/png/14_a.png", dpi=300)
    plt.close()
    # 2. Fixed n, plot weak_probability vs. K
    fig2, axes2 = plt.subplots(nrows=(len(ns) + 2) // 3, ncols=3, figsize=(15, 10))
    axes2 = axes2.flatten()
    for i, n_val in enumerate(ns):
        ax = axes2[i]
        subdf = agg[agg["hardness"] == n_val]
        for kind in subdf["kind"].unique():
            plot_df = subdf[subdf["kind"] == kind].sort_values(by="ensemble_size")
            ax.plot(plot_df["ensemble_size"], plot_df["weak_probability"], label=kind, marker="o")
        ax.set_title(f"hardness={n_val}")
        ax.set_xlabel("K")
        ax.set_ylabel("weak_probability")
        ax.grid(True)
        ax.legend()
    fig2.suptitle("Weak Probability vs. K (fixed n)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig("plots/14_b.svg")
    plt.savefig("plots/png/14_b.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    agg = make_agg("14")
    plot14(agg)

    kinds = ["boot", "bootrp"]

    print(kinds)
    for hardness in sorted(agg["hardness"].unique()):
        for ens_size in sorted(agg["ensemble_size"].unique()):
            df = agg.query(f"hardness == {hardness} and ensemble_size == {ens_size}")
            print(f"hardness={hardness},K={ens_size}: \t", end="")
            for kind in kinds:
                fdf = df.loc[df["kind"] == kind, "weak_convergence"]
                v = fdf.iloc[0] if not fdf.empty else None
                if v is not None:
                    print(f"{f'{v:.2f}':<10}", end="\t")
                else:
                    print(f"{'undefined':<10}", end="\t")
            print()
