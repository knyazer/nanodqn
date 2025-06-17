from math import sqrt
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
from scipy.optimize import minimize_scalar
from helpers import df_from

plt.rcParams.update({"font.size": 12})


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
            if df["weak_convergence"].sum() != 0:
                conv = np.vstack(df[df["weak_convergence"] == True]["collapse_metric"].to_numpy())
                row = {
                    **row,
                    "collapse_metric_mean_converged": conv.mean(axis=0),
                    "collapse_metric_std_converged": conv.std(axis=0) / sqrt(conv.shape[0]),
                }
            if df["weak_convergence"].prod() == 0:
                unconv = np.vstack(
                    df[df["weak_convergence"] == False]["collapse_metric"].to_numpy()
                )
                row = {
                    **row,
                    "collapse_metric_mean_not_converged": unconv.mean(axis=0),
                    "collapse_metric_std_not_converged": unconv.std(axis=0) / sqrt(unconv.shape[0]),
                }
        if row is None:
            raise RuntimeError(f"version was {ver} which does not seem to match any loading rules")
        agg.append(row)
    return pd.DataFrame(agg)


def plot_save(name):
    plt.tight_layout()
    Path("plots/png").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{name}.svg")
    plt.savefig(f"plots/png/{name}.png", dpi=160)
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

        label = f"$(1 - {param:.2f})^K)^n$"
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
    agg = make_agg("heatmap24-home")
    cmap = sns.color_palette("Blues", as_cmap=True)
    kinds = ["boot", "bootrp"]  # , "bootrp"] # For simplicity, let's just run one for the example

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    max_ens_size = agg["ensemble_size"].max()
    min_ens_size = agg["ensemble_size"].min()

    max_hardness = agg["hardness"].max()
    min_hardness = agg["hardness"].min()

    for i, kind in enumerate(kinds):
        ax = axes[i]
        df_kind = agg.query(f"kind == '{kind}'")
        duplicates = df_kind[df_kind.duplicated(subset=["hardness", "ensemble_size"], keep=False)]
        if len(duplicates) != 0:
            print(duplicates.head(), duplicates["ensemble_size"], duplicates["hardness"])
            breakpoint()
        pivot_data = df_kind.pivot(
            index="hardness", columns="ensemble_size", values="weak_convergence"
        )

        uniform_index = np.arange(min_hardness, max_hardness + 1)
        uniform_columns = np.arange(min_ens_size, max_ens_size + 1)
        uniform_df = pivot_data.reindex(index=uniform_index, columns=uniform_columns)

        interpolated_data = (
            uniform_df.interpolate(method="nearest", limit_direction="both", axis=0).ffill().bfill()
        )
        interpolated_data = (
            interpolated_data.interpolate(method="nearest", limit_direction="both", axis=1)
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

        ax.set_yticks(
            np.interp(pivot_data.index, uniform_index, np.arange(len(uniform_index))) + 0.5
        )
        ax.set_yticklabels(pivot_data.index.astype(int))
        ax.invert_yaxis()

        if kind == "boot":
            ax.set_title("Probability of Discovery for BDQN")
        if kind == "bootrp":
            ax.set_title("Probability of Discovery for RP-BDQN")
        ax.set_ylabel("Hardness (n)")
        ax.set_xlabel("Ensemble Size (K)")

    mappable = ax.collections[0]
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.75, pad=0.03)
    cbar.set_label("Probability of Convergence", rotation=270, labelpad=20)

    plot_save("heatmap")
    plt.show()


def _fit_beta(df: pd.DataFrame, kind: str):
    df = df[df["ensemble_size"] > 1]

    def loss(b):
        if b <= 0 or b >= 1:
            return 1e18
        p_hat = (1 - b ** df["ensemble_size"]) ** df["hardness"]
        return np.mean((p_hat - df["weak_convergence"]) ** 2)

    res = minimize_scalar(loss, bounds=(1e-6, 1 - 1e-6), method="bounded")
    beta = res.x

    # compute predicted vs observed
    p_hat = (1 - beta ** df["ensemble_size"]) ** df["hardness"]

    y = df["weak_convergence"].values
    mse = np.mean((p_hat - y) ** 2)
    ss_res = np.sum((y - p_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"Kind={kind} gets r2={r2} and mse={mse} for beta={beta}")

    return beta, mse, r2


def compute_frontier(df: pd.DataFrame, kind: str, p: float, all_hardnesses) -> pd.DataFrame:
    """
    Theoretical frontier K(n) that attains probability p with the β
    estimated by `_fit_beta`.  Works for both 'boot' and 'bootrp'.
    """
    beta, *_ = _fit_beta(df, kind)
    n_vals = all_hardnesses

    k_vals = np.log(1 - p ** (1 / n_vals)) / np.log(beta)

    # np.log(1 - p) / np.log(1 - beta**n_vals)

    keep = (k_vals > 0) & np.isfinite(k_vals)
    return pd.DataFrame({"ensemble_size": k_vals[keep], "hardness": n_vals[keep]})


def plot_scatter_with_frontier(p_levels=np.array([0.05, 0.2, 0.5, 0.8, 0.95])):
    agg = make_agg("heatmap24-home")
    kinds = ["boot", "bootrp"]
    palette = "crest"

    # --- 1. single colour map for BOTH dots and lines ---------------------------
    cmap_points = sns.color_palette(palette, as_cmap=True)  # keeps your dots
    norm = plt.Normalize(0, 1)

    # Pull N *distinct* colours out of the same cmap for the curves.
    #   • skip the very light & very dark ends (harder to see on white/black)
    #   • space them evenly so they’re visually distinct
    n_levels = len(p_levels)
    cmap_line = plt.get_cmap(palette)
    colour_idx = np.linspace(0.25, 0.85, n_levels)  # tweak as you like
    curve_cols = [cmap_line(i) for i in colour_idx]

    all_hardnesses = np.sort(
        np.array(list(agg["hardness"].unique()) + [agg["hardness"].max() * 1.11])
    )

    # --- 2. plotting ------------------------------------------------------------
    fig, axes = plt.subplots(1, len(kinds), figsize=(14, 6), sharey=True)

    x_min, x_max = agg["ensemble_size"].agg(["min", "max"])
    y_min, y_max = agg["hardness"].agg(["min", "max"])

    for ax, kind in zip(axes, kinds):
        df = agg.query(f"kind == '{kind}'")

        # empirical points
        ax.scatter(
            df["ensemble_size"],
            df["hardness"],
            c=df["weak_convergence"],
            cmap=cmap_points,
            norm=norm,
            edgecolor="none",
            s=45,
            alpha=0.7,
            rasterized=True,
        )

        # theoretical frontiers
        beta, *_ = _fit_beta(df, kind)
        for p, col in zip(p_levels, curve_cols):
            fr = compute_frontier(df, kind, p, all_hardnesses)
            if not fr.empty:
                ax.plot(
                    fr["ensemble_size"],
                    fr["hardness"],
                    linestyle="--",
                    linewidth=3,
                    color=col,
                    label=f"p={p:.2f}",
                )

        ax.set_xlabel("Ensemble Size (K)")
        if kind == "bootrp":
            ax.set_title(f"RP-BDQN vs $1 - (1 - {beta:.2f}^n)^" + "K$")
        if kind == "boot":
            ax.set_title(f"BDQN vs $1 - (1 - {beta:.2f}^n)^" + "K$")
        ax.grid(True, ls=":", lw=0.4)
        ax.legend(loc="lower right")

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max * 1.05)

    axes[0].set_ylabel("Hardness (n)")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_points),
        ax=axes,
        shrink=0.75,
        pad=0.03,
    )
    cbar.set_label("Probability of Convergence", rotation=270, labelpad=20)

    plot_save("frontier_theory_linear")
    plt.tight_layout()
    plt.show()


def plot_residuals(data_source="heatmap24-home"):
    """
    Generates a publication-quality plot of the scaling law residuals.
    """
    # Use a clean style suitable for papers
    plt.style.use("seaborn-v0_8-whitegrid")

    agg = make_agg(data_source)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))  # Standard figure size

    kind_map = {"boot": "BDQN", "bootrp": "RP-BDQN"}
    # Use a color-blind friendly and distinct palette
    palette = sns.color_palette("colorblind", n_colors=8)

    for i, kind in enumerate(kind_map.keys()):
        df = agg.query(f"kind == '{kind}'").copy()

        # --- 1. Calculate Predictions and Residuals (including K=1) ---
        beta, *_ = _fit_beta(df, kind)
        df["predicted"] = 1 - (1 - beta ** df["hardness"]) ** df["ensemble_size"]
        df["residual"] = df["predicted"] - df["weak_convergence"]

        # --- 2. Group by K and get statistics ---
        residual_stats = df.groupby("ensemble_size")["residual"].agg(["mean", "std"]).reset_index()

        # --- 3. Plotting ---
        color = palette[i + 2]
        label = kind_map[kind]

        ax.plot(
            residual_stats["ensemble_size"],
            residual_stats["mean"],
            label=label,  # Simple legend label
            color=color,
            linewidth=2,
            zorder=10,
        )

        ax.fill_between(
            residual_stats["ensemble_size"],
            residual_stats["mean"] - residual_stats["std"],
            residual_stats["mean"] + residual_stats["std"],
            color=color,
            alpha=0.2,
            zorder=5,
        )

    # --- 4. Final plot styling ---
    # Add horizontal line for perfect fit
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2, zorder=1)

    # *** NEW: Add vertical line to separate K=1 ***
    ax.axvline(1, color="red", linestyle="--", linewidth=1.2, zorder=1, label="K=1 (Pure DQN)")

    # Formal, clear labels
    ax.set_xlabel("Ensemble Size, $K$", fontsize=12)
    ax.set_ylabel("Residual ($P_{pred} - P_{obs}$)", fontsize=12)

    # Adjust ticks and legend
    ax.tick_params(axis="both", which="major", labelsize=11)

    # Re-order legend handles to be more intuitive
    handles, labels = ax.get_legend_handles_labels()
    # Puts the K=1 label last
    order = [0, 1, 2] if len(handles) == 3 else [i for i in range(len(handles))]  # Make robust
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="upper right",
        fontsize=11,
        title="Method",
    )

    # Set axis limits
    ax.set_xlim(left=0, right=agg["ensemble_size"].max() + 1)

    plt.tight_layout(pad=0.5)

    plot_save("residuals_plot")


def plot_diversity_collapse():
    plt.style.use("seaborn-v0_8-whitegrid")

    df_agg = make_agg("heatmap24-home")
    df_critical = df_agg.query("8 <= hardness <= 12 and 3 <= ensemble_size <= 6").copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    # 2. Setup aesthetics and loop variables
    kind_map = {"boot": "BDQN", "bootrp": "RP-BDQN"}
    palette = sns.color_palette("colorblind", n_colors=8)
    outcomes = {
        "converged": ("solid", "Converged"),
        "not_converged": ("dashed", "Failed"),
    }

    for i, (kind_code, kind_name) in enumerate(kind_map.items()):
        color = palette[i + 2]

        for outcome_code, (linestyle, outcome_name) in outcomes.items():
            mean_col = f"collapse_metric_mean_{outcome_code}"
            std_col = f"collapse_metric_std_{outcome_code}"

            # Filter to rows that contain data for this outcome
            df_plot = df_critical[df_critical["kind"] == kind_code].dropna(subset=[mean_col])

            if df_plot.empty:
                print(f"Skipping {kind_name} - {outcome_name} (no data)")
                continue

            # Aggregate the time-series arrays across all configs in the critical region
            mean_curves = np.vstack(df_plot[mean_col].values)
            std_curves = np.vstack(df_plot[std_col].values)

            # Calculate the final mean and std over the aggregated curves
            final_mean = np.mean(mean_curves, axis=0)
            # For visualization, we average the std dev curves. This represents the "average spread".
            final_std = np.mean(std_curves, axis=0)

            # Generate the x-axis (time)
            # Assumes all collapse metric arrays have the same length
            num_logs = len(final_mean)
            time_steps = np.arange(num_logs) * 5e2  # cuz 50k steps and 100 logs

            # Plotting
            label = f"{kind_name} ({outcome_name})"
            ax.plot(
                time_steps,
                final_mean,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
                zorder=10,
            )
            ax.fill_between(
                time_steps,
                final_mean - final_std,
                final_mean + final_std,
                color=color,
                alpha=0.15,
                zorder=5,
            )

    # 4. Final plot styling for publication
    ax.set_xlabel("Training Episodes", fontsize=12)
    ax.set_ylabel("Q-Diversity (higher is better)", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.set_yscale("log")

    # Use scientific notation for the x-axis if numbers are large
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout(pad=0.5)

    plot_save("collapse")


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
    plot_diversity_collapse()
    plot_residuals()
    plot_scatter_with_frontier()
    plot_heatmap()

    log("heatmap")
