import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Cauchy Mean Paradox Dashboard", page_icon="📉", layout="wide")

st.title("Cauchy Mean Paradox: Averaging Can Fail")
st.markdown(
    """
Compare how sample means behave under two parent distributions:

- **Normal(0, 1)**: sample means concentrate near 0 as `n` grows.
- **Cauchy(0, 1)**: sample means remain heavy-tailed for every `n`.

This is a concrete example of a stable law: Cauchy averages stay Cauchy.
"""
)

st.info(
    """
### What this dashboard demonstrates
1. **Averages can fail**: For Cauchy data, the sample mean does not become stable as sample size grows.
2. **Tail probability is invariant in n**: `P(|sample mean| > k)` stays roughly constant for Cauchy means.
3. **Robust estimator wins**: For Cauchy data, the sample median becomes much tighter than the sample mean.
"""
)

MAX_MAIN_DRAWS_PER_PARENT = 8_000_000
MAX_GRID_DRAWS_PER_PARENT = 10_000_000

with st.sidebar:
    st.header("Simulation Controls")
    sample_size = st.slider("Sample size per experiment (n)", 1, 5000, 120, step=1)
    n_experiments = st.slider("Repeated experiments", 200, 30000, 6000, step=200)
    display_clip = st.slider("Histogram x-clip (readability)", 2.0, 80.0, 20.0, step=0.5)
    threshold = st.slider("Tail threshold k for P(|estimate| > k)", 0.5, 10.0, 1.0, step=0.1)
    bins = st.slider("Histogram bins", 20, 220, 120, step=5)
    grid_experiments = st.slider("Experiments per n in trend plots", 300, 7000, 2000, step=100)
    selected_n = st.multiselect(
        "n values for trend plots",
        options=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        default=[1, 5, 20, 100, 500, 2000],
    )
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

if not selected_n:
    st.error("Pick at least one n value in the sidebar to draw trend plots.")
    st.stop()

effective_experiments = min(n_experiments, MAX_MAIN_DRAWS_PER_PARENT // sample_size)
if effective_experiments < n_experiments:
    st.warning(
        "Requested main simulation is too large for Streamlit Cloud memory limits. "
        f"Using {effective_experiments:,} experiments instead of {n_experiments:,}."
    )

sum_n = int(sum(selected_n))
effective_grid_experiments = min(grid_experiments, MAX_GRID_DRAWS_PER_PARENT // max(sum_n, 1))
if effective_grid_experiments < grid_experiments:
    st.info(
        "Trend simulation reduced for performance: "
        f"{effective_grid_experiments:,} experiments per n (requested {grid_experiments:,})."
    )


@st.cache_data(show_spinner=False)
def run_simulation(
    seed_value: int,
    n: int,
    main_m: int,
    n_values: tuple[int, ...],
    trend_m: int,
    tail_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed_value)

    normal_main = rng.normal(size=(main_m, n))
    cauchy_main = rng.standard_cauchy(size=(main_m, n))

    normal_means = normal_main.mean(axis=1)
    cauchy_means = cauchy_main.mean(axis=1)
    cauchy_medians = np.median(cauchy_main, axis=1)

    trend_rows: list[dict[str, float | int | str]] = []
    robust_rows: list[dict[str, float | int | str]] = []
    for n_value in n_values:
        normal = rng.normal(size=(trend_m, n_value))
        cauchy = rng.standard_cauchy(size=(trend_m, n_value))

        normal_mean = normal.mean(axis=1)
        cauchy_mean = cauchy.mean(axis=1)
        cauchy_median = np.median(cauchy, axis=1)

        trend_rows.append(
            {
                "n": n_value,
                "distribution": "Normal parent",
                "tail_prob": float(np.mean(np.abs(normal_mean) > tail_threshold)),
            }
        )
        trend_rows.append(
            {
                "n": n_value,
                "distribution": "Cauchy parent",
                "tail_prob": float(np.mean(np.abs(cauchy_mean) > tail_threshold)),
            }
        )
        robust_rows.append(
            {
                "n": n_value,
                "estimator": "Cauchy sample mean",
                "iqr": float(np.quantile(cauchy_mean, 0.75) - np.quantile(cauchy_mean, 0.25)),
            }
        )
        robust_rows.append(
            {
                "n": n_value,
                "estimator": "Cauchy sample median",
                "iqr": float(np.quantile(cauchy_median, 0.75) - np.quantile(cauchy_median, 0.25)),
            }
        )

    return (
        normal_means,
        cauchy_means,
        cauchy_medians,
        pd.DataFrame(trend_rows),
        pd.DataFrame(robust_rows),
    )


normal_means, cauchy_means, cauchy_medians, trend_df, robust_df = run_simulation(
    seed_value=int(seed),
    n=sample_size,
    main_m=effective_experiments,
    n_values=tuple(sorted(set(selected_n))),
    trend_m=effective_grid_experiments,
    tail_threshold=threshold,
)

tab_a, tab_b, tab_c = st.tabs(
    [
        "Mean Distribution",
        "Tail Invariance Across n",
        "Robust Alternative",
    ]
)

with tab_a:
    st.markdown(
        """
**How to read this panel**
- Blue (Normal means) should squeeze toward 0 as `n` increases.
- Red (Cauchy means) keeps wide tails even for large `n`.
- `P(|.| > k)` on the right quantifies this contrast.
"""
    )

    plot_df = pd.DataFrame(
        {
            "estimate": np.concatenate([normal_means, cauchy_means]),
            "distribution": ["Normal parent"] * len(normal_means) + ["Cauchy parent"] * len(cauchy_means),
        }
    )
    plot_df_clipped = plot_df[plot_df["estimate"].between(-display_clip, display_clip)]

    left, right = st.columns([2, 1])
    with left:
        fig = px.histogram(
            plot_df_clipped,
            x="estimate",
            color="distribution",
            barmode="overlay",
            nbins=bins,
            opacity=0.65,
            histnorm="probability density",
            marginal="box",
            color_discrete_map={"Normal parent": "#1f77b4", "Cauchy parent": "#d62728"},
        )
        fig.update_layout(legend_title_text="", xaxis_title="Sample mean", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Quick Stats")
        st.metric("P(|Normal mean| > k)", f"{np.mean(np.abs(normal_means) > threshold):.4f}")
        st.metric("P(|Cauchy mean| > k)", f"{np.mean(np.abs(cauchy_means) > threshold):.4f}")
        st.metric(
            "Cauchy points clipped",
            f"{100 * np.mean(np.abs(cauchy_means) > display_clip):.2f}%",
            help="Share of Cauchy sample means outside the displayed histogram range.",
        )
        st.metric(
            "Normal points clipped",
            f"{100 * np.mean(np.abs(normal_means) > display_clip):.4f}%",
            help="Share of Normal sample means outside the displayed histogram range.",
        )

        summary_df = pd.DataFrame(
            [
                {
                    "Estimator": "Normal sample mean",
                    "Median": float(np.median(normal_means)),
                    "IQR": float(np.quantile(normal_means, 0.75) - np.quantile(normal_means, 0.25)),
                    f"P(|.|>{threshold})": float(np.mean(np.abs(normal_means) > threshold)),
                },
                {
                    "Estimator": "Cauchy sample mean",
                    "Median": float(np.median(cauchy_means)),
                    "IQR": float(np.quantile(cauchy_means, 0.75) - np.quantile(cauchy_means, 0.25)),
                    f"P(|.|>{threshold})": float(np.mean(np.abs(cauchy_means) > threshold)),
                },
                {
                    "Estimator": "Cauchy sample median",
                    "Median": float(np.median(cauchy_medians)),
                    "IQR": float(np.quantile(cauchy_medians, 0.75) - np.quantile(cauchy_medians, 0.25)),
                    f"P(|.|>{threshold})": float(np.mean(np.abs(cauchy_medians) > threshold)),
                },
            ]
        )
        st.dataframe(
            summary_df.style.format(
                {
                    "Median": "{:.4f}",
                    "IQR": "{:.4f}",
                    f"P(|.|>{threshold})": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_b:
    st.markdown(
        """
**How to read this panel**
- If averaging worked for Cauchy, the red line would fall with `n`.
- Instead, the red line stays near a constant level (the dashed theoretical line).
- The blue Normal line drops toward 0.
"""
    )

    theoretical_cauchy_tail = 1.0 - (2.0 / np.pi) * np.arctan(threshold)
    tail_fig = px.line(
        trend_df,
        x="n",
        y="tail_prob",
        color="distribution",
        markers=True,
        log_x=True,
        color_discrete_map={"Normal parent": "#1f77b4", "Cauchy parent": "#d62728"},
    )
    tail_fig.add_hline(
        y=theoretical_cauchy_tail,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="Cauchy theory",
    )
    tail_fig.update_layout(
        legend_title_text="",
        xaxis_title="Sample size n (log scale)",
        yaxis_title=f"P(|sample mean| > {threshold})",
    )
    st.plotly_chart(tail_fig, use_container_width=True)
    st.caption(
        "For Cauchy, the line stays flat near the theoretical value because "
        "the sample mean has the same distribution for every n."
    )

with tab_c:
    st.markdown(
        """
**How to read this panel**
- Red line: spread of Cauchy sample mean (stays large).
- Green line: spread of Cauchy sample median (shrinks with `n`).
- This shows why robust estimators are preferable for heavy tails.
"""
    )

    robust_fig = px.line(
        robust_df,
        x="n",
        y="iqr",
        color="estimator",
        markers=True,
        log_x=True,
        color_discrete_map={"Cauchy sample mean": "#d62728", "Cauchy sample median": "#2ca02c"},
    )
    robust_fig.update_layout(
        legend_title_text="",
        xaxis_title="Sample size n (log scale)",
        yaxis_title="IQR of estimator",
    )
    st.plotly_chart(robust_fig, use_container_width=True)

    max_n = int(robust_df["n"].max())
    iqr_mean_max = float(
        robust_df.loc[(robust_df["n"] == max_n) & (robust_df["estimator"] == "Cauchy sample mean"), "iqr"].iloc[0]
    )
    iqr_median_max = float(
        robust_df.loc[(robust_df["n"] == max_n) & (robust_df["estimator"] == "Cauchy sample median"), "iqr"].iloc[0]
    )
    st.markdown(
        f"At **n={max_n}**, IQR(sample mean) is about **{iqr_mean_max:.3f}** "
        f"vs IQR(sample median) about **{iqr_median_max:.3f}**."
    )
    st.caption(
        "The median is a robust estimator for Cauchy location; "
        "the mean is not."
    )

st.markdown("---")
st.markdown(
    rf"""
### Mathematical fact behind the paradox

- If $X_1,\dots,X_n \sim \mathrm{{Cauchy}}(0,1)$ i.i.d., then
  $\frac{{1}}{{n}}\sum_i X_i \sim \mathrm{{Cauchy}}(0,1)$ for every $n$.
- So $P(|\bar{{X}}| > {threshold}) = 1 - \frac{{2}}{{\pi}}\arctan({threshold:.1f}) \approx {theoretical_cauchy_tail:.4f}$.
- In contrast, Normal means obey concentration and shrink as $n$ grows.
"""
)
