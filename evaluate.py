from math import sqrt
from pathlib import Path
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import dataclasses
from matplotlib.patches import Rectangle
from scipy.stats import binomtest
import seaborn as sns
import yaml
import functools as ft
from typing import Literal
from scipy.optimize import minimize_scalar
from helpers import df_from, RUN_NAME
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.size": 16})


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
    Path("plots/png").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"plots/{name}.svg")
    plt.savefig(f"plots/png/{name}.png", dpi=160)
    plt.close()


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


def plot_frontier_and_heatmaps(p_levels=np.array([0.05, 0.2, 0.5, 0.8, 0.95])):
    # ------------- your data ------------------------------------------------
    agg = make_agg(RUN_NAME).query("ensemble_size <= 40")
    kinds = ["boot", "bootrp"]

    cmap = plt.get_cmap("viridis_r")
    norm = plt.Normalize(0, 1)
    curve_cols = [cmap(t) for t in np.linspace(0.15, 0.85, len(p_levels))]
    all_hardnesses = np.sort(np.r_[agg["hardness"].unique(), agg["hardness"].max() * 1.11])

    ens_min, ens_max = agg["ensemble_size"].agg(["min", "max"])
    hard_min, hard_max = agg["hardness"].agg(["min", "max"])

    # ------------- layout: 3 × 2 gridspec -----------------------------------
    fig = plt.figure(figsize=(14, 10), tight_layout=True)
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[10, 10, 1.5], left=0.05, right=0.95, top=0.95, bottom=0.05
    )

    ax_heat = [fig.add_subplot(gs[0, c]) for c in range(2)]
    ax_front = [fig.add_subplot(gs[1, c]) for c in range(2)]
    ax_legend = fig.add_subplot(gs[2, :])  # spans BOTH columns
    ax_legend.axis("off")  # text only, no frame

    # ------------- plotting loop --------------------------------------------
    for i, kind in enumerate(kinds):
        df = agg.query(f"kind == '{kind}'")

        # ---- heat-map (top row)
        piv = df.pivot(index="hardness", columns="ensemble_size", values="weak_convergence")

        ui = np.arange(hard_min, hard_max + 1)
        uc = np.arange(ens_min, ens_max + 1)
        full = piv.reindex(index=ui, columns=uc)
        interp = (
            full.interpolate("nearest", axis=0, limit_direction="both")
            .ffill()
            .bfill()
            .interpolate("nearest", axis=1, limit_direction="both")
            .ffill()
            .bfill()
        )

        sns.heatmap(interp, ax=ax_heat[i], cmap=cmap, vmin=0, vmax=1, cbar=False, rasterized=True)

        # tidy tick labels
        wanted_x = [5, 10, 15, 20, 25, 30, 35, 40]
        ax_heat[i].set_xticks(wanted_x)
        start_h = (hard_min // 5) * 5
        wanted_y = np.arange(start_h, hard_max + 5, 5)
        have_y = [h for h in wanted_y if h in ui]
        ax_heat[i].set_yticks([np.where(ui == h)[0][0] + 0.5 for h in have_y])
        ax_heat[i].set_yticklabels(have_y)
        ax_heat[i].invert_yaxis()
        ax_heat[i].set_xlabel("")
        if i == 0:
            ax_heat[i].set_ylabel("Hardness, n")
        else:
            ax_heat[i].set_ylabel("")
        ax_heat[i].set_title(
            ("BDQN" if kind == "boot" else "RP-BDQN") + " Probability of Discovery"
        )

        # ---- frontier (middle row)
        sc = ax_front[i].scatter(
            df["ensemble_size"],
            df["hardness"],
            c=df["weak_convergence"],
            cmap=cmap,
            norm=norm,
            s=45,
            edgecolor="none",
            alpha=0.7,
            rasterized=True,
        )

        beta, *_ = _fit_beta(df, kind)
        for p, colr in zip(p_levels, curve_cols):
            fr = compute_frontier(df, kind, p, all_hardnesses)
            if fr.empty:
                continue
            lbl = f"p={p:.2f}" if kind == "bootrp" else None
            ax_front[i].plot(
                fr["ensemble_size"], fr["hardness"], ls="--", lw=3, color=colr, label=lbl
            )

        ax_front[i].set_xlim(ens_min, ens_max)
        ax_front[i].set_ylim(hard_min, hard_max * 1.05)
        ax_front[i].grid(ls=":", lw=0.4)
        ax_front[i].set_xlabel("Ensemble Size, K")
        if i == 0:
            ax_front[i].set_ylabel("Hardness, n")

        name = "BDQN" if kind == "boot" else "RP-BDQN"
        ax_front[i].set_title(f"{name} PoD vs $1-(1-{beta:.2f}^n)^K$")

        if kind == "bootrp":
            ax_front[i].add_patch(
                Rectangle((18, 30), 20, 10, lw=2, ec="red", ls="--", fc="none", zorder=6)
            )

        # ---- strip any per-axes legend safely
        lg = ax_front[i].get_legend()
        if lg is not None:
            lg.remove()

    # ------------- single, centred legend row ------------------------------
    handles, labels = ax_front[1].get_legend_handles_labels()
    if handles:  # only if RP-BDQN drew p-curves
        ax_legend.legend(
            handles, labels, loc="center", frameon=False, ncol=len(labels), handlelength=2.0
        )

    # ------------- shared colour-bar ---------------------------------------
    mappable = ax_heat[0].collections[0]  # first heat-map artist
    cbar = fig.colorbar(
        mappable,
        ax=ax_heat + ax_front,
        orientation="vertical",
        fraction=0.025,
        pad=0.02,
        shrink=0.83,
    )
    cbar.set_label("Probability of Discovery (PoD)", rotation=270, labelpad=18)

    plot_save("frontier_and_heatmaps")


def _fit_beta(df: pd.DataFrame, kind: str):
    df = df[df["ensemble_size"] > 1]

    def loss(b):
        if b <= 0 or b >= 1:
            return 1e18
        p_hat = 1 - (1 - b ** df["hardness"]) ** df["ensemble_size"]
        return np.mean((p_hat - df["weak_convergence"]) ** 2)

    res = minimize_scalar(loss, bounds=(1e-6, 1 - 1e-6), method="bounded")
    beta = res.x

    # compute predicted vs observed
    p_hat = 1 - (1 - beta ** df["hardness"]) ** df["ensemble_size"]

    y = df["weak_convergence"].values
    mse = np.mean((p_hat - y) ** 2)
    ss_res = np.sum((y - p_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"Kind={kind} gets r2={r2} and mse={mse} for beta={beta}")

    return beta, mse, r2


def compute_frontier(df: pd.DataFrame, kind: str, p: float, all_hardnesses) -> pd.DataFrame:
    beta, *_ = _fit_beta(df, kind)
    n_vals = all_hardnesses

    k_vals = np.log(1 - p) / np.log(1 - beta**n_vals) + 1

    keep = (k_vals > 0) & np.isfinite(k_vals)
    return pd.DataFrame({"ensemble_size": k_vals[keep], "hardness": n_vals[keep]})


def plot_residuals():
    # Use a clean style suitable for papers
    plt.style.use("seaborn-v0_8-whitegrid")

    agg = make_agg(RUN_NAME)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))  # Standard figure size

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
    ax.set_xlabel("Ensemble Size, $K$")
    ax.set_ylabel("Residual ($P_{pred} - P_{obs}$)")

    # Adjust ticks and legend
    ax.tick_params(axis="both", which="major")

    # Re-order legend handles to be more intuitive
    handles, labels = ax.get_legend_handles_labels()
    # Puts the K=1 label last
    order = [0, 1, 2] if len(handles) == 3 else [i for i in range(len(handles))]  # Make robust
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="lower right",
        frameon=True,
    )

    # Set axis limits
    ax.set_xlim(left=0, right=agg["ensemble_size"].max() + 1)

    plt.tight_layout(pad=0.5)

    plot_save("residuals_plot")


def plot_diversity_collapse():
    plt.style.use("seaborn-v0_8-whitegrid")

    df_agg = make_agg(RUN_NAME)
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
    ax.set_xlabel("Training Episodes")
    ax.set_ylabel("Q-Diversity (higher is better)")
    ax.tick_params(axis="both", which="major")
    ax.legend(frameon=True)
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
    plot_frontier_and_heatmaps()

    log(RUN_NAME)
