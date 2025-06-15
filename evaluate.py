from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from scipy.stats import binomtest
import seaborn as sns
import yaml
import functools as ft
from typing import Literal


@ft.lru_cache
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


@ft.lru_cache
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
                "weak_convergence": df["weak_convergence"].sum() / len(df),
                "ensemble_size": cfg.ensemble_size,
                "hardness": cfg.hardness,
                "kind": cfg.kind,
                "prior_scale": cfg.prior_scale,
            }
        )
    return pd.DataFrame(agg)


def plot_save(name):
    Path("plots/png").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{name}.svg")
    plt.savefig(f"plots/png/{name}.png", dpi=600)
    plt.close()


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


def make_theoretical(ax, x_values, kind: Literal["slow", "fast"], param: float, K=None, n=None):
    assert K is not None or n is not None
    if kind == "slow":
        if K is not None:
            y = [1 - (1 - param**n_val) ** K for n_val in x_values]
        else:
            y = [1 - (1 - param**n) ** K_val for K_val in x_values]
        label = f"$1 - (1 - {param:.3f}^n)^K$"
    else:
        if K is not None:
            y = [1 - (1 - param**K) ** n_val for n_val in x_values]
        else:
            y = [1 - (1 - param**K_val) ** n for K_val in x_values]

        label = f"$(1 - {beta:.2f})^K)^n$"
    return y, label


def ax_set_log_scale(ax, m1=False):
    def forward_log_1m(x):
        return np.where(x == 1, -1_000, np.log(1 - x))

    def inverse_log_1m(x):
        return 1 - np.exp(x)

    ax.set_ylim([0.01, 0.99])
    if m1:
        # ax.set_yscale("log")
        ax.set_yscale("function", functions=(forward_log_1m, inverse_log_1m))
    else:
        ax.set_yscale("log")


def plot14(agg: pd.DataFrame):
    # Rename if needed
    if "weak_convergence" in agg.columns and "weak_probability" not in agg.columns:
        agg = agg.rename(columns={"weak_convergence": "weak_probability"})

    # Sorted unique K and n
    Ks = sorted(agg["ensemble_size"].unique())
    ns = sorted(agg["hardness"].unique())

    # 1. Fixed K: vs n
    fig1, axes1 = plt.subplots((len(Ks) + 2) // 3, 3, figsize=(15, 15))
    axes1 = axes1.flatten()
    for i, K in enumerate(Ks):
        ax = axes1[i]
        set_scale(ax)
        subdf = agg[agg["ensemble_size"] == K]
        # Empirical
        for kind in subdf["kind"].unique():
            pdf = subdf[subdf["kind"] == kind].sort_values(by="hardness")
            ax.plot(
                pdf["hardness"], pdf["weak_probability"], marker="o", label=f"empirical ({kind})"
            )
        # Theoretical
        plot_theoretical(ax, ns, K, True)
        ax.set_title(f"K={K}")
        ax.set_xlabel("hardness (n)")
        ax.set_ylabel("weak_probability")
        ax.grid(True)
        ax.legend(fontsize="small", ncol=2)
    fig1.suptitle("Weak Probability vs. n (fixed K)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig1.savefig("plots/14_a.svg")
    fig1.savefig("plots/png/14_a.png", dpi=300)
    plt.close(fig1)

    # 2. Fixed n: vs K
    fig2, axes2 = plt.subplots((len(ns) + 2) // 3, 3, figsize=(15, 15))
    axes2 = axes2.flatten()
    for i, n_val in enumerate(ns):
        ax = axes2[i]
        set_scale(ax, True)
        subdf = agg[agg["hardness"] == n_val]
        for kind in subdf["kind"].unique():
            pdf = subdf[subdf["kind"] == kind].sort_values(by="ensemble_size")
            ax.plot(
                pdf["ensemble_size"],
                pdf["weak_probability"],
                marker="o",
                label=f"empirical ({kind})",
            )
        plot_theoretical(ax, Ks, n_val, False)
        ax.set_title(f"hardness={n_val}")
        ax.set_xlabel("ensemble size (K)")
        ax.set_ylabel("weak_probability")
        ax.grid(True)
        ax.legend(fontsize="small", ncol=2)
    fig2.suptitle("Weak Probability vs. K (fixed n)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig2.savefig("plots/14_b.svg")
    fig2.savefig("plots/png/14_b.png", dpi=300)
    plt.close(fig2)


def plot_heatmaps():
    agg = make_agg("24")
    cmap = sns.color_palette("Blues", as_cmap=True)
    kinds = ["boot"]  # , "bootrp"] # For simplicity, let's just run one for the example

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, kind in enumerate(kinds):
        ax = axes[i]
        df_kind = agg.query(f"kind == '{kind}'")

        pivot_data = df_kind.pivot(
            index="ensemble_size", columns="hardness", values="weak_convergence"
        )

        uniform_index = np.arange(pivot_data.index.min(), pivot_data.index.max() + 1)
        uniform_columns = np.arange(pivot_data.columns.min(), pivot_data.columns.max() + 1)
        uniform_df = pivot_data.reindex(index=uniform_index, columns=uniform_columns)

        interpolated_data = uniform_df.interpolate(method="nearest", limit_direction="both", axis=0)
        interpolated_data = interpolated_data.interpolate(
            method="nearest", limit_direction="both", axis=1
        )

        sns.heatmap(
            interpolated_data,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1.0,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
        )

        ax.set_xticks(
            np.interp(pivot_data.columns, uniform_columns, np.arange(len(uniform_columns))) + 0.5
        )
        ax.set_xticklabels([f"{x}" for x in pivot_data.columns])

        if i == 0:
            ax.set_yticks(
                np.interp(pivot_data.index, uniform_index, np.arange(len(uniform_index))) + 0.5
            )
            ax.set_yticklabels(pivot_data.index.astype(int))
            ax.set_ylabel("Ensemble Size (K)")

        ax.set_title(f"Kind = '{kind}'")
        ax.set_xlabel("Hardness (n)")

    mappable = ax.collections[0]
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.75, pad=0.03)
    cbar.set_label("Probability of Convergence", rotation=270, labelpad=20)

    plot_save("heatmaps")
    plt.show()


if __name__ == "__main__":
    plot_heatmaps()

    agg = make_agg("24")
    # plot14(agg)

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
                    print(f"{f'{v:.3f}':<10}", end="\t")
                else:
                    print(f"{'undefined':<10}", end="\t")
            print()
