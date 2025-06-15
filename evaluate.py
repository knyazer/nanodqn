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
from helpers import df_from


@ft.lru_cache
def load(path: str | Path):
    folder = Path(path)
    with open(folder / ".version") as f:
        v = f.read().strip()
    if not (v == "1" or v == "2" or v == "3"):
        raise ValueError(f"unsupported result log version {v}")
    df = df_from(folder / "results.csv")
    with open(folder / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    fields = [(key, type(value), dataclasses.field(default=value)) for key, value in cfg.items()]
    DynamicConfig = dataclasses.make_dataclass("DynamicConfig", fields)
    cfg_instance = DynamicConfig(**cfg)
    return df, cfg_instance, int(v)


@ft.lru_cache
def make_agg(version):
    root = Path(f"results/{version}")
    paths = sorted(root.glob("*"))
    loaded = [load(path) for path in paths]
    agg = []
    for df, cfg, ver in loaded:
        row = None
        if ver >= 1:
            row = {
                "mean_time_to_weak": df["time_to_weak"].mean(),
                "max_time_to_weak": df["time_to_weak"].max(),
                "mean_time_to_strong": df["time_to_strong"].mean(),
                "weak_convergence": df["weak_convergence"].sum() / len(df),
                "ensemble_size": cfg.ensemble_size,
                "hardness": cfg.hardness,
                "kind": cfg.kind,
                "prior_scale": cfg.prior_scale,
            }
        if ver == 3:
            conv = df[df["weak_convergence"] == True]["collapse_metric"]
            unconv = df[df["weak_convergence"] == False]["collapse_metric"]
            row = {
                **row,
                "collapse_metric_mean_converged": conv.mean(),
                "collapse_metric_mean_not_converged": unconv.mean(),
                "collapse_metric_std_converged": conv.std(),
                "collapse_metric_std_not_converged": unconv.std(),
            }
        if row is None:
            raise RuntimeError(f"version was {ver} which does not seem to match any loading rules")
        agg.append(row)
    return pd.DataFrame(agg)


def plot_save(name):
    Path("plots/png").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{name}.svg")
    plt.savefig(f"plots/png/{name}.png", dpi=600)
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


def plot_heatmap():
    agg = make_agg("heatmap")
    cmap = sns.color_palette("Blues", as_cmap=True)
    kinds = ["boot"]  # , "bootrp"] # For simplicity, let's just run one for the example

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, kind in enumerate(kinds):
        ax = axes[i]
        df_kind = agg.query(f"kind == '{kind}'")
        duplicates = df_kind[df_kind.duplicated(subset=["ensemble_size", "hardness"], keep=False)]
        if len(duplicates) != 0:
            print(duplicates.head(), duplicates["ensemble_size"], duplicates["hardness"])
            breakpoint()
        pivot_data = df_kind.pivot(
            index="ensemble_size", columns="hardness", values="weak_convergence"
        )

        uniform_index = np.arange(pivot_data.index.min(), pivot_data.index.max() + 1)
        uniform_columns = np.arange(pivot_data.columns.min(), pivot_data.columns.max() + 1)
        uniform_df = pivot_data.reindex(index=uniform_index, columns=uniform_columns)

        interpolated_data = (
            uniform_df.interpolate(method="linear", limit_direction="both", axis=0).ffill().bfill()
        )
        interpolated_data = (
            interpolated_data.interpolate(method="linear", limit_direction="both", axis=1)
            .ffill()
            .bfill()
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

    plot_save("heatmap")
    plt.show()


def log(name):
    agg = make_agg(name)

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


if __name__ == "__main__":
    plot_heatmap()

    log("heatmap")
