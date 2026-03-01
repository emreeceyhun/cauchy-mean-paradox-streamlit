import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Cauchy Mean Paradox Dashboard",
    page_icon="📉",
    layout="wide",
)

st.title("Cauchy Mean Paradox: When Averaging Fails")
st.markdown(
    """
This dashboard compares sample means from two distributions:

- **Normal(0, 1)**: sample means concentrate around 0 as sample size grows.
- **Cauchy(0, 1)**: sample means do **not** stabilize; they remain Cauchy-distributed.

For the Cauchy distribution, even averaging many samples does not tame extreme values.
"""
)

with st.sidebar:
    st.header("Simulation Controls")
    sample_size = st.slider("Sample size per experiment (n)", 1, 5000, 50, step=1)
    n_experiments = st.slider("Number of repeated experiments", 100, 20000, 5000, step=100)
    display_clip = st.slider("Display x-range clip (for readability)", 2.0, 100.0, 15.0, step=0.5)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)
    bins = st.slider("Histogram bins", 20, 300, 120, step=5)

rng = np.random.default_rng(int(seed))

normal_samples = rng.normal(loc=0.0, scale=1.0, size=(n_experiments, sample_size))
cauchy_samples = rng.standard_cauchy(size=(n_experiments, sample_size))

normal_means = normal_samples.mean(axis=1)
cauchy_means = cauchy_samples.mean(axis=1)

plot_df = pd.DataFrame(
    {
        "sample_mean": np.concatenate([normal_means, cauchy_means]),
        "distribution": ["Normal parent"] * n_experiments + ["Cauchy parent"] * n_experiments,
    }
)

plot_df_clipped = plot_df[
    (plot_df["sample_mean"] >= -display_clip) & (plot_df["sample_mean"] <= display_clip)
]

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Sample Means")
    fig = px.histogram(
        plot_df_clipped,
        x="sample_mean",
        color="distribution",
        barmode="overlay",
        nbins=bins,
        opacity=0.6,
        marginal="box",
        histnorm="probability density",
    )
    fig.update_layout(legend_title_text="", xaxis_title="Sample mean", yaxis_title="Density")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Robust Summary")

    def summary(values: np.ndarray, name: str) -> dict:
        return {
            "Distribution": name,
            "Median": float(np.median(values)),
            "IQR": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
            "Std. Dev.": float(np.std(values)),
            "P(|mean| > 1)": float(np.mean(np.abs(values) > 1)),
            "P(|mean| > 5)": float(np.mean(np.abs(values) > 5)),
        }

    summary_df = pd.DataFrame(
        [
            summary(normal_means, "Normal parent"),
            summary(cauchy_means, "Cauchy parent"),
        ]
    )

    st.dataframe(summary_df.style.format({k: "{:.4f}" for k in summary_df.columns if k != "Distribution"}), use_container_width=True)

st.markdown("---")
st.markdown(
    rf"""
### Why this happens

- For **Normal(0,1)** data, the sample mean has standard deviation about $1/\sqrt{{n}}$.
- For **Cauchy(0,1)** data, the sample mean has the **same Cauchy law** for every $n$.

Current settings: $n={sample_size}$, experiments = {n_experiments}.  
Despite larger $n$, Cauchy sample means keep showing large outliers.
"""
)

st.caption("Tip: Increase sample size and watch Normal means collapse toward 0 while Cauchy means stay heavy-tailed.")
