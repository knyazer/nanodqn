from math import sqrt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
import seaborn as sns
import yaml
import functools as ft
from scipy.optimize import minimize_scalar
from helpers import df_from, RUN_NAME
import matplotlib.gridspec as gridspec

plt.rcParams.update({"font.size": 7})
# This is the most important part.
plt.rcParams.update(
    {
        "text.usetex": True,  # Use LaTeX to render all text
        "font.family": "serif",  # Use serif font family
        # Match the preamble of your document
        "text.latex.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage{amsmath}
        \usepackage{amssymb}
    """,
    }
)

plt.style.use("seaborn-v0_8-whitegrid")


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
                "rb_size": cfg.rb_size,
                "lr": cfg.lr,
            }
        if ver == 3:
            if df["weak_convergence"].sum() != 0:
                conv = np.vstack(df[df["weak_convergence"] == True]["collapse_metric"].to_numpy())  # noqa
                row = {
                    **row,
                    "collapse_metric_mean_converged": conv.mean(axis=0),
                    "collapse_metric_std_converged": conv.std(axis=0) / sqrt(conv.shape[0]),
                }
            if df["weak_convergence"].prod() == 0:
                unconv = np.vstack(
                    df[df["weak_convergence"] == False]["collapse_metric"].to_numpy()  # noqa
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


def plot_save(name, prefix=None):
    root = Path("plots")
    if prefix is not None:
        root = root / prefix
    root.mkdir(parents=True, exist_ok=True)
    plt.savefig(root / Path(f"{name}.png"), dpi=360)
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


def plot_frontier_and_heatmaps(
    p_levels=np.array([0.05, 0.2, 0.5, 0.8, 0.95]), agg=None, kinds=None, plot_prefix=None
):
    if agg is None:
        agg = make_agg(RUN_NAME).query("ensemble_size <= 40")
    if kinds is None:
        kinds = ["boot", "bootrp"]

    cmap = sns.color_palette("Blues", as_cmap=True)
    norm = plt.Normalize(0, 1)
    curve_cols = [cmap(t) for t in np.linspace(0.15, 0.85, len(p_levels))]
    all_hardnesses = np.sort(np.r_[agg["hardness"].unique(), agg["hardness"].max() * 1.11])

    ens_min, ens_max = agg["ensemble_size"].agg(["min", "max"])
    hard_min, hard_max = agg["hardness"].agg(["min", "max"])

    fig = plt.figure(figsize=(5.5, 4), tight_layout=True)
    gs = gridspec.GridSpec(
        3, 2, height_ratios=[10, 10, 0.0], left=0.08, right=0.95, top=0.92, bottom=0.05, hspace=0.7
    )

    ax_heat = [fig.add_subplot(gs[0, c]) for c in range(2)]
    ax_front = [fig.add_subplot(gs[1, c]) for c in range(2)]
    ax_legend = fig.add_subplot(gs[2, :])  # spans BOTH columns
    ax_legend.axis("off")  # text only, no frame

    # ------------- plotting loop --------------------------------------------
    fit_results = {}  # Store results for table display

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
        wanted_x = [1, 10, 20, 30, 40]
        ax_heat[i].set_xticks(wanted_x)
        ax_heat[i].set_xticklabels(wanted_x, rotation=0)
        start_h = (hard_min // 10) * 10
        wanted_y = np.arange(start_h, hard_max + 5, 10)
        have_y = [h for h in wanted_y if h in ui]
        ax_heat[i].set_yticks([np.where(ui == h)[0][0] + 0.5 for h in have_y])
        ax_heat[i].set_yticklabels(have_y, rotation=0)

        ax_heat[i].invert_yaxis()
        ax_heat[i].set_xlabel("Ensemble size, K")

        ax_heat[i].set_ylabel("Hardness, n")
        ax_heat[i].set_title(
            ("BDQN" if kind == "boot" else "RP-BDQN") + " Probability of Discovery"
        )

        # ---- frontier (middle row)
        ax_front[i].scatter(
            df["ensemble_size"],
            df["hardness"],
            c=df["weak_convergence"],
            cmap=cmap,
            norm=norm,
            s=10,
            edgecolor="#aaa",
            linewidths=0.3,
        )

        psi, mse, dispersion, r2 = _fit_psi(df, kind)
        fit_results[kind] = {"psi": psi, "mse": mse, "dispersion": dispersion, "r2": r2}
        for p, colr in zip(p_levels, curve_cols):
            fr = compute_frontier(df, kind, p, all_hardnesses)
            if fr.empty:
                continue
            lbl = f"PoD={p:.2f}" if kind == "bootrp" else None
            ax_front[i].plot(
                fr["ensemble_size"], fr["hardness"], ls="--", lw=2, color=colr, label=lbl
            )

        ax_front[i].set_xlim(ens_min * 0.8, ens_max * 1.2)
        ax_front[i].set_ylim(hard_min * 0.5, hard_max * 1.05)
        ax_front[i].grid(ls=":", lw=0.4)
        ax_front[i].set_xlabel("Ensemble Size, K")
        ax_front[i].set_ylabel("Hardness, n")

        ax_front[i].set_xscale("log")
        ax_front[i].set_xticks([1, 2, 4, 8, 16, 32])
        ax_front[i].set_xticklabels([1, 2, 4, 8, 16, 32])

        name = "BDQN" if kind == "boot" else "RP-BDQN"
        ax_front[i].set_title(f"{name} vs $\psi={psi:.2f}$ law")

        # ---- strip any per-axes legend safely
        lg = ax_front[i].get_legend()
        if lg is not None:
            lg.remove()

    # ------------- single, centred legend row ------------------------------
    handles, labels = ax_front[1].get_legend_handles_labels()
    if handles:  # only if RP-BDQN drew p-curves
        ax_legend.legend(
            handles,
            labels,
            loc="center",
            ncol=len(labels),
            handlelength=2.0,
            bbox_to_anchor=(0.5, 0.5),
        )

    # ------------- shared colour-bar ---------------------------------------
    mappable = ax_heat[0].collections[0]  # first heat-map artist
    cbar = fig.colorbar(
        mappable,
        ax=ax_heat + ax_front,
        orientation="vertical",
        fraction=0.025,
        pad=0.04,
        shrink=0.83,
    )
    cbar.set_label("Probability of Discovery (PoD)", rotation=270, labelpad=18)

    # Display table only for main heatmap (not hyperparameter sweeps)
    if plot_prefix is None and fit_results:
        print("\n" + "=" * 72)
        print("PSI FITTING RESULTS")
        print("=" * 72)
        print(f"{'Algorithm':<12} {'Psi (ψ)':<12} {'Dispersion':<12} {'R²':<12} {'MSE':<12}")
        print("-" * 72)

        kind_names = {"boot": "BDQN", "bootrp": "RP-BDQN"}
        for kind in ["boot", "bootrp"]:
            if kind in fit_results:
                result = fit_results[kind]
                print(
                    f"{kind_names[kind]:<12} {result['psi']:<12.6f} {result['dispersion']:<12.3f} {result['r2']:<12.3f} {result['mse']:<12.6f}"
                )
        print("=" * 72)

    plot_save("frontier_and_heatmaps", prefix=plot_prefix)


def _fit_psi(df: pd.DataFrame, kind: str, bootstrap_uncertainty: bool = False):
    df = df[df["ensemble_size"] > 1]

    def negloglike_for_df(data_df, psi):
        if psi <= 0 or psi >= 1:
            return 1e18
        p = 1 - (1 - psi ** data_df["hardness"]) ** data_df["ensemble_size"]
        p = np.clip(p, 1e-15, 1 - 1e-15)  # prevent log(0)
        k = data_df["weak_convergence"] * 32  # successes out of 32 trials
        n = 32
        return -np.sum(k * np.log(p) + (n - k) * np.log1p(-p))

    def fit_psi_for_df(data_df):
        def negloglike(psi):
            return negloglike_for_df(data_df, psi)

        res = minimize_scalar(negloglike, bounds=(1e-6, 1 - 1e-6), method="bounded")
        return res.x

    # Fit psi on original data
    psi = fit_psi_for_df(df)

    # compute predicted vs observed for original data
    p_hat = 1 - (1 - psi ** df["hardness"]) ** df["ensemble_size"]

    y = df["weak_convergence"].values
    mse = np.mean((p_hat - y) ** 2)

    # Compute Pearson chi-squared dispersion factor
    # For binomial data: chi^2 = sum((observed - expected)^2 / (expected * (1 - expected) / n))
    # where n = 32 (number of trials)
    n_trials = 32
    expected_successes = p_hat * n_trials
    observed_successes = y * n_trials

    # Variance for binomial: n * p * (1 - p)
    variance = n_trials * p_hat * (1 - p_hat)
    # Avoid division by zero
    variance = np.maximum(variance, 1e-10)

    chi2 = np.sum((observed_successes - expected_successes) ** 2 / variance)
    degrees_of_freedom = len(df) - 1  # -1 for the fitted parameter (psi)
    dispersion_factor = chi2 / degrees_of_freedom if degrees_of_freedom > 0 else np.inf

    # Compute R²
    ss_res = np.sum((y - p_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    if bootstrap_uncertainty:
        # Bootstrap resampling
        n_bootstrap = 100
        bootstrap_psis = []

        for _ in range(n_bootstrap):
            # Resample data with replacement
            bootstrap_df = df.sample(n=len(df), replace=True)
            bootstrap_psi = fit_psi_for_df(bootstrap_df)
            bootstrap_psis.append(bootstrap_psi)

        bootstrap_psis = np.array(bootstrap_psis)

        # Compute 80% confidence interval (10th and 90th percentiles)
        ci_lower = np.percentile(bootstrap_psis, 2.5)
        ci_upper = np.percentile(bootstrap_psis, 97.5)

        return psi, mse, dispersion_factor, r2, (ci_lower, ci_upper)
    else:
        return psi, mse, dispersion_factor, r2


def compute_frontier(df: pd.DataFrame, kind: str, p: float, all_hardnesses) -> pd.DataFrame:
    psi, *_ = _fit_psi(df, kind)
    n_vals = all_hardnesses

    k_vals = np.log(1 - p) / np.log(1 - psi**n_vals) + 1

    keep = (k_vals > 0) & np.isfinite(k_vals)
    return pd.DataFrame({"ensemble_size": k_vals[keep], "hardness": n_vals[keep]})


def plot_residuals():
    # Use a clean style suitable for papers
    agg = make_agg(RUN_NAME)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3), tight_layout=True)  # Paper-ready figure size

    kind_map = {"boot": "BDQN", "bootrp": "RP-BDQN"}
    # Use a color-blind friendly and distinct palette
    palette = sns.color_palette("colorblind", n_colors=8)

    for i, kind in enumerate(kind_map.keys()):
        df = agg.query(f"kind == '{kind}'").copy()

        # --- 1. Calculate Predictions and Residuals (including K=1) ---
        psi, *_ = _fit_psi(df, kind)
        df["predicted"] = 1 - (1 - psi ** df["hardness"]) ** df["ensemble_size"]
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
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(labels),
    )

    # Set axis limits
    ax.set_xlim(left=0, right=agg["ensemble_size"].max() + 1)

    plt.tight_layout(pad=0.5)

    plot_save("residuals_plot")


def plot_diversity_collapse():
    df_agg = make_agg(RUN_NAME)
    df_critical = df_agg.query("8 <= hardness <= 12 and 3 <= ensemble_size <= 6").copy()

    fig, ax = plt.subplots(figsize=(5.5, 3), tight_layout=True)

    # 2. Setup aesthetics and loop variables
    kind_map = {"boot": "BDQN", "bootrp": "RP-BDQN"}
    palette = sns.color_palette("colorblind", n_colors=8)
    outcomes = {
        "converged": ("solid", "Convergent"),
        "not_converged": ("dashed", "Non-convergent"),
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
    ax.set_ylabel("Q-Diversity")
    ax.tick_params(axis="both", which="major")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6)
    ax.set_yscale("log")

    # Use scientific notation for the x-axis if numbers are large
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout(pad=0.5)

    plot_save("collapse")


hp_and_ranges = [
    ("rb_size", [5_000, 20_000, 40_000], ["boot", "bootrp"]),
    ("lr", [8e-5, 5e-4, 1e-3], ["boot", "bootrp"]),
    ("prior_scale", [1.0, 5.0, 10.0], ["bootrp"]),
]


def plot_hyperparameter_sweep():
    df_agg = make_agg("sweep")

    fig = plt.figure(figsize=(5.5, 2.2))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        bottom=0.2,
        top=0.98,
        left=0.08,
        right=0.95,
        wspace=0.5,
        hspace=0.05,
        height_ratios=[10, 1],
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    palette = sns.color_palette("colorblind", n_colors=8)
    kind_colors = {"boot": palette[2], "bootrp": palette[3]}
    kind_names = {"boot": "BDQN", "bootrp": "RP-BDQN"}

    for i, (hp, values, kinds) in enumerate(hp_and_ranges):
        ax = axes[i]

        x_pos = np.arange(len(values))
        width = 0.35

        for j, kind in enumerate(kinds):
            df_kind = df_agg.query(f"kind == '{kind}'")
            psis = []
            ci_lowers = []
            ci_uppers = []

            for value in values:
                df_subset = df_kind.query(f"{hp} == {value}")
                if not df_subset.empty:
                    result = _fit_psi(df_subset, kind, bootstrap_uncertainty=True)
                    psi, _, _, _, (ci_lower, ci_upper) = result
                    psis.append(psi)
                    ci_lowers.append(ci_lower)
                    ci_uppers.append(ci_upper)
                else:
                    psis.append(0)
                    ci_lowers.append(0)
                    ci_uppers.append(0)

            offset = (j - 0.5) * width if len(kinds) == 2 else 0

            # Calculate error bar values (distance from psi to confidence bounds)
            yerr_lower = [
                max(0, psi - ci_lower) if psi > 0 else 0 for psi, ci_lower in zip(psis, ci_lowers)
            ]
            yerr_upper = [
                ci_upper - psi if psi > 0 else 0 for psi, ci_upper in zip(psis, ci_uppers)
            ]

            bars = ax.bar(
                x_pos + offset,
                psis,
                width,
                label=kind_names[kind],
                color=kind_colors[kind],
                alpha=0.8,
            )

            # Add error bars
            ax.errorbar(
                x_pos + offset,
                psis,
                yerr=[yerr_lower, yerr_upper],
                fmt="none",
                capsize=3,
                capthick=0.5,
                ecolor="black",
                alpha=0.5,
            )

            for bar, psi in zip(bars, psis):
                if psi > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{psi:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )

        if hp == "lr":
            ax.set_xlabel("Learning Rate")
        if hp == "rb_size":
            ax.set_xlabel("Replay Buffer Size")
        if hp == "prior_scale":
            ax.set_xlabel("Prior Scale")
        ax.set_ylabel("Fitted $\psi$")
        ax.set_xticks(x_pos)

        if hp == "rb_size":
            ax.set_xticklabels([f"{int(v / 1000)}K" for v in values])
        elif hp == "lr":
            ax.set_xticklabels([f"{v:.0e}" for v in values])
        else:
            ax.set_xticklabels([f"{v}" for v in values])

        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(ax.get_ylim()[1] + 0.1, 0.1))

    # Create a shared legend below the subplots
    # Get handles and labels from the first axis that has both kinds
    handles, labels = None, None
    for ax in axes:
        h, l = ax.get_legend_handles_labels()  # noqa
        if len(h) >= 2:  # Found axis with both BDQN and RP-BDQN
            handles, labels = h, l
            break

    # Remove individual legends
    for ax in axes:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    # Add shared legend if we found handles
    if handles and labels:
        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis("off")
        legend_ax.legend(
            handles, labels, loc="center", ncol=2, bbox_to_anchor=(0.5, -1.6), frameon=False
        )

    plot_save("hyperparameter_sweep")


if __name__ == "__main__":
    plot_hyperparameter_sweep()
    plot_frontier_and_heatmaps()
    plot_diversity_collapse()
    plot_residuals()
    for hp, values, kinds in hp_and_ranges:
        for v in values:
            plot_frontier_and_heatmaps(
                agg=make_agg("sweep").query(f"{hp} == {v}"), kinds=kinds, plot_prefix=f"{hp}_eq_{v}"
            )
