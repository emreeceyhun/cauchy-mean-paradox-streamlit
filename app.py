import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Distribution Presentations", page_icon="📊", layout="wide")

st.title("Statistical Properties by Distribution")
st.markdown(
    """
This app is organized as separate presentations:

1. **Cauchy**: sample mean does not stabilize.
2. **Dirichlet component (centered)**: bounded data with concentration as `n` grows.
3. **Normal**: classic concentration / CLT baseline.
"""
)

MAX_MAIN_DRAWS = 8_000_000
MAX_GRID_DRAWS = 10_000_000

with st.sidebar:
    st.header("Shared Controls")
    sample_size = st.slider("Sample size per experiment (n)", 1, 5000, 150, step=1)
    n_experiments = st.slider("Repeated experiments", 200, 30000, 6000, step=200)
    threshold = st.slider("Tail threshold k for P(|estimate| > k)", 0.01, 10.0, 0.5, step=0.01)
    display_clip = st.slider("Histogram x-clip", 1.0, 80.0, 20.0, step=0.5)
    bins = st.slider("Histogram bins", 20, 220, 110, step=5)

    st.divider()
    st.subheader("Trend Controls")
    grid_experiments = st.slider("Experiments per n in trend plots", 300, 7000, 2000, step=100)
    selected_n = st.multiselect(
        "n values for trend plots",
        options=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        default=[1, 5, 20, 100, 500, 2000],
    )

    st.divider()
    st.subheader("Dirichlet Parameters")
    dirichlet_dim = st.slider("Dirichlet dimension d", 2, 20, 5, step=1)
    dirichlet_alpha = st.slider("Dirichlet alpha (symmetric)", 0.1, 5.0, 0.7, step=0.1)

    st.divider()
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

if not selected_n:
    st.error("Pick at least one n value in the sidebar.")
    st.stop()

n_values = tuple(sorted(set(selected_n)))
effective_experiments = min(n_experiments, MAX_MAIN_DRAWS // sample_size)
if effective_experiments < n_experiments:
    st.warning(
        "Main simulation reduced for memory safety: "
        f"{effective_experiments:,} experiments instead of {n_experiments:,}."
    )

grid_cap = MAX_GRID_DRAWS // max(sum(n_values), 1)
effective_grid_experiments = min(grid_experiments, grid_cap)
if effective_grid_experiments < grid_experiments:
    st.info(
        "Trend simulation reduced for performance: "
        f"{effective_grid_experiments:,} experiments per n (requested {grid_experiments:,})."
    )


@st.cache_data(show_spinner=False)
def simulate_cauchy(
    seed_value: int,
    n: int,
    main_m: int,
    trend_m: int,
    n_grid: tuple[int, ...],
    k: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed_value)

    main = rng.standard_cauchy(size=(main_m, n))
    means = main.mean(axis=1)
    medians = np.median(main, axis=1)

    tail_rows: list[dict[str, float | int | str]] = []
    robust_rows: list[dict[str, float | int | str]] = []
    for n_value in n_grid:
        samples = rng.standard_cauchy(size=(trend_m, n_value))
        sample_means = samples.mean(axis=1)
        sample_medians = np.median(samples, axis=1)

        tail_rows.append({"n": n_value, "series": "Sample mean", "tail_prob": float(np.mean(np.abs(sample_means) > k))})

        robust_rows.append(
            {
                "n": n_value,
                "estimator": "Sample mean",
                "iqr": float(np.quantile(sample_means, 0.75) - np.quantile(sample_means, 0.25)),
            }
        )
        robust_rows.append(
            {
                "n": n_value,
                "estimator": "Sample median",
                "iqr": float(np.quantile(sample_medians, 0.75) - np.quantile(sample_medians, 0.25)),
            }
        )

    return means, medians, pd.DataFrame(tail_rows), pd.DataFrame(robust_rows)


@st.cache_data(show_spinner=False)
def simulate_dirichlet_component(
    seed_value: int,
    n: int,
    main_m: int,
    trend_m: int,
    n_grid: tuple[int, ...],
    k: float,
    d: int,
    alpha: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed_value)
    center = 1.0 / d

    # For symmetric Dirichlet(alpha,...,alpha), one component is Beta(alpha, (d-1)alpha).
    main = rng.beta(alpha, (d - 1) * alpha, size=(main_m, n)) - center
    means = main.mean(axis=1)

    tail_rows: list[dict[str, float | int]] = []
    scale_rows: list[dict[str, float | int | str]] = []
    for n_value in n_grid:
        samples = rng.beta(alpha, (d - 1) * alpha, size=(trend_m, n_value)) - center
        sample_means = samples.mean(axis=1)

        iqr = float(np.quantile(sample_means, 0.75) - np.quantile(sample_means, 0.25))
        tail_rows.append({"n": n_value, "tail_prob": float(np.mean(np.abs(sample_means) > k))})
        scale_rows.append({"n": n_value, "series": "IQR(mean)", "value": iqr})
        scale_rows.append({"n": n_value, "series": "IQR(mean) * sqrt(n)", "value": iqr * np.sqrt(n_value)})

    return means, pd.DataFrame(tail_rows), pd.DataFrame(scale_rows)


@st.cache_data(show_spinner=False)
def simulate_normal(
    seed_value: int,
    n: int,
    main_m: int,
    trend_m: int,
    n_grid: tuple[int, ...],
    k: float,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed_value)

    main = rng.normal(size=(main_m, n))
    means = main.mean(axis=1)

    tail_rows: list[dict[str, float | int]] = []
    scale_rows: list[dict[str, float | int | str]] = []
    for n_value in n_grid:
        samples = rng.normal(size=(trend_m, n_value))
        sample_means = samples.mean(axis=1)

        std = float(np.std(sample_means))
        tail_rows.append({"n": n_value, "tail_prob": float(np.mean(np.abs(sample_means) > k))})
        scale_rows.append({"n": n_value, "series": "Std(mean)", "value": std})
        scale_rows.append({"n": n_value, "series": "Std(mean) * sqrt(n)", "value": std * np.sqrt(n_value)})

    return means, pd.DataFrame(tail_rows), pd.DataFrame(scale_rows)


cauchy_means, cauchy_medians, cauchy_tail_df, cauchy_robust_df = simulate_cauchy(
    seed_value=int(seed),
    n=sample_size,
    main_m=effective_experiments,
    trend_m=effective_grid_experiments,
    n_grid=n_values,
    k=threshold,
)

dirichlet_means, dirichlet_tail_df, dirichlet_scale_df = simulate_dirichlet_component(
    seed_value=int(seed) + 101,
    n=sample_size,
    main_m=effective_experiments,
    trend_m=effective_grid_experiments,
    n_grid=n_values,
    k=threshold,
    d=dirichlet_dim,
    alpha=dirichlet_alpha,
)

normal_means, normal_tail_df, normal_scale_df = simulate_normal(
    seed_value=int(seed) + 202,
    n=sample_size,
    main_m=effective_experiments,
    trend_m=effective_grid_experiments,
    n_grid=n_values,
    k=threshold,
)


def clipped_df(values: np.ndarray, label: str, clip: float) -> pd.DataFrame:
    df = pd.DataFrame({"estimate": values, "series": label})
    return df[df["estimate"].between(-clip, clip)]


tab_cauchy, tab_dirichlet, tab_normal = st.tabs(
    [
        "1) Cauchy: Mean Paradox",
        "2) Dirichlet: Bounded Concentration",
        "3) Normal: CLT Baseline",
    ]
)

with tab_cauchy:
    st.subheader("Presentation 1: Cauchy Distribution")
    st.markdown(
        """
**Key message**
- The sample mean of Cauchy data does not become stable with larger `n`.
- Tail probability for the sample mean stays roughly constant in `n`.
- The sample median is much more stable than the sample mean.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        cauchy_hist_df = clipped_df(cauchy_means, "Cauchy sample mean", display_clip)
        fig = px.histogram(
            cauchy_hist_df,
            x="estimate",
            nbins=bins,
            opacity=0.75,
            histnorm="probability density",
            color_discrete_sequence=["#d62728"],
            marginal="box",
            title="Distribution of Cauchy sample mean",
        )
        fig.update_layout(xaxis_title="Sample mean", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("P(|sample mean| > k)", f"{np.mean(np.abs(cauchy_means) > threshold):.4f}")
        st.metric("P(|sample median| > k)", f"{np.mean(np.abs(cauchy_medians) > threshold):.4f}")
        st.metric("% clipped at x-range", f"{100 * np.mean(np.abs(cauchy_means) > display_clip):.2f}%")

        cauchy_summary = pd.DataFrame(
            [
                {
                    "Estimator": "Cauchy sample mean",
                    "Median": float(np.median(cauchy_means)),
                    "IQR": float(np.quantile(cauchy_means, 0.75) - np.quantile(cauchy_means, 0.25)),
                },
                {
                    "Estimator": "Cauchy sample median",
                    "Median": float(np.median(cauchy_medians)),
                    "IQR": float(np.quantile(cauchy_medians, 0.75) - np.quantile(cauchy_medians, 0.25)),
                },
            ]
        )
        st.dataframe(cauchy_summary.style.format({"Median": "{:.4f}", "IQR": "{:.4f}"}), hide_index=True)

    cauchy_tail_theory = 1.0 - (2.0 / np.pi) * np.arctan(threshold)
    tail_fig = px.line(
        cauchy_tail_df,
        x="n",
        y="tail_prob",
        markers=True,
        log_x=True,
        color="series",
        color_discrete_map={"Sample mean": "#d62728"},
        title=f"Tail probability across n: P(|mean| > {threshold})",
    )
    tail_fig.add_hline(
        y=cauchy_tail_theory,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="Cauchy theory",
    )
    tail_fig.update_layout(legend_title_text="", xaxis_title="n (log scale)", yaxis_title="Tail probability")
    st.plotly_chart(tail_fig, use_container_width=True)

    robust_fig = px.line(
        cauchy_robust_df,
        x="n",
        y="iqr",
        color="estimator",
        markers=True,
        log_x=True,
        color_discrete_map={"Sample mean": "#d62728", "Sample median": "#2ca02c"},
        title="Robustness check: spread of estimators",
    )
    robust_fig.update_layout(legend_title_text="", xaxis_title="n (log scale)", yaxis_title="IQR")
    st.plotly_chart(robust_fig, use_container_width=True)

    st.markdown(
        rf"""
For Cauchy(0,1), if $X_1,\ldots,X_n$ are i.i.d., then
$\bar{{X}} = \frac{{1}}{{n}}\sum_i X_i \sim \text{{Cauchy}}(0,1)$ for every $n$.
Therefore $P(|\bar{{X}}|>{threshold}) = 1 - \frac{{2}}{{\pi}}\arctan({threshold:.2f}) \approx {cauchy_tail_theory:.4f}$.
"""
    )

with tab_dirichlet:
    st.subheader("Presentation 2: Dirichlet Distribution")
    st.markdown(
        rf"""
**Setup used here**
- Symmetric Dirichlet with dimension $d={dirichlet_dim}$ and $\alpha={dirichlet_alpha:.1f}$.
- We present one centered component: $X_1 - 1/d$ (mean is 0).

**Key message**
- This component is bounded, has finite variance, and sample means concentrate with larger `n`.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        dir_hist_df = clipped_df(dirichlet_means, "Dirichlet centered component mean", display_clip)
        fig = px.histogram(
            dir_hist_df,
            x="estimate",
            nbins=bins,
            opacity=0.75,
            histnorm="probability density",
            color_discrete_sequence=["#ff7f0e"],
            marginal="box",
            title="Distribution of centered Dirichlet component sample mean",
        )
        fig.update_layout(xaxis_title="Sample mean", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("P(|sample mean| > k)", f"{np.mean(np.abs(dirichlet_means) > threshold):.4f}")
        st.metric("% clipped at x-range", f"{100 * np.mean(np.abs(dirichlet_means) > display_clip):.4f}%")
        st.metric("IQR(sample mean)", f"{(np.quantile(dirichlet_means, 0.75) - np.quantile(dirichlet_means, 0.25)):.4f}")

    dir_tail_fig = px.line(
        dirichlet_tail_df,
        x="n",
        y="tail_prob",
        markers=True,
        log_x=True,
        color_discrete_sequence=["#ff7f0e"],
        title=f"Tail probability across n: P(|mean| > {threshold})",
    )
    dir_tail_fig.update_layout(xaxis_title="n (log scale)", yaxis_title="Tail probability")
    st.plotly_chart(dir_tail_fig, use_container_width=True)

    dir_scale_fig = px.line(
        dirichlet_scale_df,
        x="n",
        y="value",
        color="series",
        markers=True,
        log_x=True,
        color_discrete_map={"IQR(mean)": "#ff7f0e", "IQR(mean) * sqrt(n)": "#9467bd"},
        title="Concentration rate check",
    )
    dir_scale_fig.update_layout(legend_title_text="", xaxis_title="n (log scale)", yaxis_title="Value")
    st.plotly_chart(dir_scale_fig, use_container_width=True)

    dir_var = (dirichlet_dim - 1) / (dirichlet_dim**2 * (dirichlet_dim * dirichlet_alpha + 1))
    st.markdown(
        rf"""
For symmetric $\mathrm{{Dir}}(\alpha,\dots,\alpha)$, one component satisfies
$X_1 \sim \mathrm{{Beta}}(\alpha, (d-1)\alpha)$.
Its variance is finite:
$\mathrm{{Var}}(X_1) = \frac{{d-1}}{{d^2(d\alpha+1)}} \approx {dir_var:.5f}$.
Finite variance is why sample means concentrate.
"""
    )

with tab_normal:
    st.subheader("Presentation 3: Normal Distribution")
    st.markdown(
        """
**Key message**
- Normal data is the classic case where averaging works.
- Spread of sample means decreases at rate about `1/sqrt(n)`.
"""
    )

    col1, col2 = st.columns(2)
    with col1:
        normal_hist_df = clipped_df(normal_means, "Normal sample mean", display_clip)
        fig = px.histogram(
            normal_hist_df,
            x="estimate",
            nbins=bins,
            opacity=0.75,
            histnorm="probability density",
            color_discrete_sequence=["#1f77b4"],
            marginal="box",
            title="Distribution of Normal sample mean",
        )
        fig.update_layout(xaxis_title="Sample mean", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("P(|sample mean| > k)", f"{np.mean(np.abs(normal_means) > threshold):.4f}")
        st.metric("% clipped at x-range", f"{100 * np.mean(np.abs(normal_means) > display_clip):.4f}%")
        st.metric("Std(sample mean)", f"{np.std(normal_means):.4f}")

    normal_tail_fig = px.line(
        normal_tail_df,
        x="n",
        y="tail_prob",
        markers=True,
        log_x=True,
        color_discrete_sequence=["#1f77b4"],
        title=f"Tail probability across n: P(|mean| > {threshold})",
    )
    normal_tail_fig.update_layout(xaxis_title="n (log scale)", yaxis_title="Tail probability")
    st.plotly_chart(normal_tail_fig, use_container_width=True)

    normal_scale_fig = px.line(
        normal_scale_df,
        x="n",
        y="value",
        color="series",
        markers=True,
        log_x=True,
        color_discrete_map={"Std(mean)": "#1f77b4", "Std(mean) * sqrt(n)": "#17becf"},
        title="CLT scaling check",
    )
    normal_scale_fig.update_layout(legend_title_text="", xaxis_title="n (log scale)", yaxis_title="Value")
    st.plotly_chart(normal_scale_fig, use_container_width=True)

st.markdown("---")
st.caption("If you want, I can add more numbered presentations (e.g., Student-t, Pareto, Lognormal).")
