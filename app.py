import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Interesting Distribution Properties", page_icon="📊", layout="wide")

st.title("Interesting Statistical Properties by Distribution")
st.markdown(
    """
Each presentation focuses on a **different** property:

1. **Cauchy**: sample mean paradox (mean does not stabilize).
2. **Dirichlet**: simplex tradeoff and negative dependence between components.
3. **Normal**: squared Euclidean norm follows a Chi-square law.
4. **Dirichlet for documents**: alpha controls topic mixing and classification difficulty.
"""
)

MAX_MAIN_DRAWS = 8_000_000
MAX_GRID_DRAWS = 10_000_000
MAX_DOC_WORD_CELLS = 12_000_000
ALPHA_GRID = np.array([0.15, 0.3, 0.5, 1.0, 2.0, 4.0])
DOC_ALPHA_GRID = np.array([0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 3.0])

with st.sidebar:
    st.header("Shared Controls")
    n_experiments = st.slider("Repeated experiments", 500, 30000, 7000, step=250)
    seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

    st.divider()
    st.subheader("Cauchy Controls")
    cauchy_n = st.slider("Cauchy sample size (n)", 1, 5000, 150, step=1)
    cauchy_threshold = st.slider("Cauchy tail threshold k", 0.05, 10.0, 0.5, step=0.05)
    cauchy_clip = st.slider("Cauchy histogram x-clip", 1.0, 80.0, 20.0, step=0.5)
    cauchy_bins = st.slider("Cauchy histogram bins", 20, 240, 120, step=5)
    cauchy_grid_experiments = st.slider("Cauchy trend experiments per n", 300, 8000, 2200, step=100)
    cauchy_n_grid = st.multiselect(
        "Cauchy n values for trend",
        options=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000],
        default=[1, 5, 20, 100, 500, 2000],
    )

    st.divider()
    st.subheader("Dirichlet Controls")
    dirichlet_d = st.slider("Dirichlet dimension d", 3, 20, 5, step=1)
    dirichlet_alpha = st.slider("Dirichlet alpha (symmetric)", 0.1, 5.0, 0.7, step=0.1)
    dirichlet_points = st.slider("Dirichlet points shown", 400, 6000, 2500, step=100)

    st.divider()
    st.subheader("Normal Controls")
    normal_k = st.slider("Normal dimension k (for sum of squares)", 1, 40, 6, step=1)
    normal_bins = st.slider("Normal histogram bins", 20, 220, 100, step=5)

    st.divider()
    st.subheader("Dirichlet Document Demo")
    doc_classes = st.slider("Number of classes", 2, 8, 4, step=1)
    doc_topics = st.slider("Number of topics", max(3, doc_classes), 12, max(6, doc_classes), step=1)
    doc_vocab_size = st.slider("Vocabulary size", 200, 2000, 800, step=50)
    doc_length = st.slider("Words per document", 30, 400, 120, step=10)
    doc_train_docs = st.slider("Training documents", 200, 5000, 1200, step=100)
    doc_test_docs = st.slider("Test documents", 200, 5000, 1000, step=100)
    doc_alpha = st.slider("Base alpha (topic mixing noise)", 0.02, 3.0, 0.25, step=0.02)
    doc_class_boost = st.slider("Class-topic boost", 0.2, 8.0, 2.5, step=0.1)
    doc_topic_eta = st.slider("Topic-word eta", 0.02, 2.0, 0.25, step=0.02)

if not cauchy_n_grid:
    st.error("Pick at least one value in 'Cauchy n values for trend'.")
    st.stop()

cauchy_n_values = tuple(sorted(set(cauchy_n_grid)))

cauchy_main_m = min(n_experiments, MAX_MAIN_DRAWS // max(cauchy_n, 1))
normal_main_m = min(n_experiments, MAX_MAIN_DRAWS // max(normal_k, 1))
dirichlet_main_m = min(n_experiments, MAX_MAIN_DRAWS // max(dirichlet_d, 1))
doc_train_eff = min(doc_train_docs, MAX_DOC_WORD_CELLS // max(doc_vocab_size, 1))
doc_test_eff = min(doc_test_docs, MAX_DOC_WORD_CELLS // max(doc_vocab_size, 1))

cauchy_grid_m = min(cauchy_grid_experiments, MAX_GRID_DRAWS // max(sum(cauchy_n_values), 1))

if cauchy_main_m < n_experiments:
    st.warning(
        f"Cauchy main simulation reduced to {cauchy_main_m:,} experiments for memory safety "
        f"(requested {n_experiments:,})."
    )
if normal_main_m < n_experiments:
    st.info(
        f"Normal main simulation reduced to {normal_main_m:,} experiments for memory safety "
        f"(requested {n_experiments:,})."
    )
if dirichlet_main_m < n_experiments:
    st.info(
        f"Dirichlet main simulation reduced to {dirichlet_main_m:,} experiments for memory safety "
        f"(requested {n_experiments:,})."
    )
if cauchy_grid_m < cauchy_grid_experiments:
    st.info(
        f"Cauchy trend simulation reduced to {cauchy_grid_m:,} experiments per n "
        f"(requested {cauchy_grid_experiments:,})."
    )
if doc_train_eff < doc_train_docs or doc_test_eff < doc_test_docs:
    st.info(
        "Dirichlet document demo reduced for memory safety: "
        f"train {doc_train_eff:,}/{doc_train_docs:,}, test {doc_test_eff:,}/{doc_test_docs:,}."
    )


@st.cache_data(show_spinner=False)
def simulate_cauchy(
    seed_value: int,
    n: int,
    main_m: int,
    trend_m: int,
    n_grid: tuple[int, ...],
    threshold_k: float,
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

        tail_rows.append(
            {
                "n": n_value,
                "series": "Sample mean",
                "tail_prob": float(np.mean(np.abs(sample_means) > threshold_k)),
            }
        )

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
def simulate_dirichlet(
    seed_value: int,
    d: int,
    alpha: float,
    draws: int,
    alpha_grid: tuple[float, ...],
) -> tuple[pd.DataFrame, float, float, float, pd.DataFrame]:
    rng = np.random.default_rng(seed_value)
    samples = rng.dirichlet(np.full(d, alpha), size=draws)

    x1 = samples[:, 0]
    x2 = samples[:, 1]
    x3 = samples[:, 2]
    sum_error = float(np.max(np.abs(samples.sum(axis=1) - 1.0)))

    empirical_corr = float(np.corrcoef(x1, x2)[0, 1])
    theoretical_corr = -1.0 / (d - 1)

    plot_df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "max_component": np.max(samples, axis=1),
        }
    )

    profile_rows: list[dict[str, float]] = []
    profile_draws = min(5000, max(1500, draws // 2))
    for a in alpha_grid:
        profile_samples = rng.dirichlet(np.full(d, a), size=profile_draws)
        profile_rows.append(
            {
                "alpha": float(a),
                "mean_max_component": float(np.mean(np.max(profile_samples, axis=1))),
                "median_entropy": float(
                    np.median(-np.sum(profile_samples * np.log(np.clip(profile_samples, 1e-12, 1.0)), axis=1))
                ),
            }
        )

    return plot_df, empirical_corr, theoretical_corr, sum_error, pd.DataFrame(profile_rows)


@st.cache_data(show_spinner=False)
def simulate_normal_chisq(seed_value: int, k: int, draws: int) -> np.ndarray:
    rng = np.random.default_rng(seed_value)
    z = rng.normal(size=(draws, k))
    return np.sum(z**2, axis=1)


@st.cache_data(show_spinner=False)
def simulate_dirichlet_doc_classification(
    seed_value: int,
    n_topics: int,
    n_classes: int,
    vocab_size: int,
    doc_len: int,
    train_docs: int,
    test_docs: int,
    base_alpha: float,
    class_boost: float,
    topic_word_eta: float,
    alpha_grid: tuple[float, ...],
) -> tuple[float, pd.DataFrame, pd.DataFrame, float, float]:
    base_topic_rng = np.random.default_rng(seed_value)
    topic_word = base_topic_rng.dirichlet(np.full(vocab_size, topic_word_eta), size=n_topics)

    def sample_docs(local_rng: np.random.Generator, n_docs: int, alpha_noise: float) -> tuple[np.ndarray, np.ndarray, float, float]:
        y = local_rng.integers(0, n_classes, size=n_docs)
        x = np.zeros((n_docs, vocab_size), dtype=np.int32)
        dominant_total = 0.0
        entropy_total = 0.0

        for i, cls in enumerate(y):
            concentration = np.full(n_topics, alpha_noise)
            concentration[cls % n_topics] += class_boost
            theta = local_rng.dirichlet(concentration)

            word_prob = theta @ topic_word
            x[i] = local_rng.multinomial(doc_len, word_prob)
            dominant_total += float(np.max(theta))
            entropy_total += float(-np.sum(theta * np.log(np.clip(theta, 1e-12, 1.0))))

        return x, y, dominant_total / n_docs, entropy_total / n_docs

    def train_nb(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        class_word = np.zeros((n_classes, vocab_size), dtype=np.float64)
        class_count = np.bincount(y, minlength=n_classes).astype(np.float64)
        for cls in range(n_classes):
            if class_count[cls] > 0:
                class_word[cls] = x[y == cls].sum(axis=0)

        smoothing = 1.0
        phi = (class_word + smoothing) / (class_word.sum(axis=1, keepdims=True) + smoothing * vocab_size)
        log_phi = np.log(phi)
        log_prior = np.log(np.clip(class_count / max(len(y), 1), 1e-12, 1.0))
        return log_phi, log_prior

    def predict_nb(log_phi: np.ndarray, log_prior: np.ndarray, x: np.ndarray) -> np.ndarray:
        scores = x @ log_phi.T + log_prior
        return np.argmax(scores, axis=1)

    selected_rng = np.random.default_rng(seed_value + 71)
    x_train, y_train, dom_selected, ent_selected = sample_docs(selected_rng, train_docs, base_alpha)
    x_test, y_test, _, _ = sample_docs(selected_rng, test_docs, base_alpha)
    log_phi, log_prior = train_nb(x_train, y_train)
    y_pred = predict_nb(log_phi, log_prior, x_test)

    selected_acc = float(np.mean(y_pred == y_test))
    conf = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(conf, (y_test, y_pred), 1)
    conf_df = pd.DataFrame(
        conf,
        index=[f"true {i}" for i in range(n_classes)],
        columns=[f"pred {i}" for i in range(n_classes)],
    )

    profile_rows: list[dict[str, float]] = []
    for i, alpha_value in enumerate(alpha_grid):
        alpha_rng = np.random.default_rng(seed_value + 1000 + i)
        x_train_a, y_train_a, dom_a, ent_a = sample_docs(alpha_rng, train_docs, alpha_value)
        x_test_a, y_test_a, _, _ = sample_docs(alpha_rng, test_docs, alpha_value)
        log_phi_a, log_prior_a = train_nb(x_train_a, y_train_a)
        y_pred_a = predict_nb(log_phi_a, log_prior_a, x_test_a)
        profile_rows.append(
            {
                "alpha": float(alpha_value),
                "accuracy": float(np.mean(y_pred_a == y_test_a)),
                "mean_dominant_topic_weight": float(dom_a),
                "mean_topic_entropy": float(ent_a),
            }
        )

    return selected_acc, conf_df, pd.DataFrame(profile_rows), float(dom_selected), float(ent_selected)


def chi_square_pdf(x: np.ndarray, k: int) -> np.ndarray:
    coeff = 1.0 / ((2.0 ** (k / 2.0)) * math.gamma(k / 2.0))
    return coeff * np.power(x, (k / 2.0) - 1.0) * np.exp(-x / 2.0)


cauchy_means, cauchy_medians, cauchy_tail_df, cauchy_robust_df = simulate_cauchy(
    seed_value=int(seed),
    n=cauchy_n,
    main_m=cauchy_main_m,
    trend_m=cauchy_grid_m,
    n_grid=cauchy_n_values,
    threshold_k=cauchy_threshold,
)

dirichlet_df, dir_emp_corr, dir_theory_corr, dir_sum_error, dir_profile_df = simulate_dirichlet(
    seed_value=int(seed) + 101,
    d=dirichlet_d,
    alpha=dirichlet_alpha,
    draws=max(dirichlet_main_m, dirichlet_points),
    alpha_grid=tuple(ALPHA_GRID.tolist()),
)

normal_r2 = simulate_normal_chisq(seed_value=int(seed) + 202, k=normal_k, draws=normal_main_m)
doc_alpha_grid_values = np.unique(np.append(DOC_ALPHA_GRID, doc_alpha))
doc_selected_acc, doc_conf_df, doc_profile_df, doc_dom_selected, doc_entropy_selected = (
    simulate_dirichlet_doc_classification(
        seed_value=int(seed) + 303,
        n_topics=doc_topics,
        n_classes=doc_classes,
        vocab_size=doc_vocab_size,
        doc_len=doc_length,
        train_docs=doc_train_eff,
        test_docs=doc_test_eff,
        base_alpha=doc_alpha,
        class_boost=doc_class_boost,
        topic_word_eta=doc_topic_eta,
        alpha_grid=tuple(float(v) for v in doc_alpha_grid_values),
    )
)

tab_cauchy, tab_dirichlet, tab_normal, tab_doc = st.tabs(
    [
        "1) Cauchy: Mean Paradox",
        "2) Dirichlet: Simplex & Dependence",
        "3) Normal: Chi-square Geometry",
        "4) Dirichlet for Document Classification",
    ]
)

with tab_cauchy:
    st.subheader("Presentation 1: Cauchy Distribution")
    st.markdown(
        """
**Interesting property**
- The sample mean of i.i.d. Cauchy data stays Cauchy for every sample size `n`.
- So averaging does not stabilize the mean, unlike finite-variance cases.
"""
    )

    cauchy_hist_df = pd.DataFrame({"estimate": cauchy_means})
    cauchy_hist_df = cauchy_hist_df[cauchy_hist_df["estimate"].between(-cauchy_clip, cauchy_clip)]

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.histogram(
            cauchy_hist_df,
            x="estimate",
            nbins=cauchy_bins,
            opacity=0.75,
            histnorm="probability density",
            marginal="box",
            color_discrete_sequence=["#d62728"],
            title="Distribution of Cauchy sample mean",
        )
        fig.update_layout(xaxis_title="Sample mean", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.metric("P(|sample mean| > k)", f"{np.mean(np.abs(cauchy_means) > cauchy_threshold):.4f}")
        st.metric("P(|sample median| > k)", f"{np.mean(np.abs(cauchy_medians) > cauchy_threshold):.4f}")
        st.metric("IQR(sample mean)", f"{(np.quantile(cauchy_means, 0.75) - np.quantile(cauchy_means, 0.25)):.4f}")
        st.metric("IQR(sample median)", f"{(np.quantile(cauchy_medians, 0.75) - np.quantile(cauchy_medians, 0.25)):.4f}")

    cauchy_theory_tail = 1.0 - (2.0 / np.pi) * np.arctan(cauchy_threshold)
    tail_fig = px.line(
        cauchy_tail_df,
        x="n",
        y="tail_prob",
        color="series",
        markers=True,
        log_x=True,
        color_discrete_map={"Sample mean": "#d62728"},
        title=f"Tail invariance: P(|mean| > {cauchy_threshold}) across n",
    )
    tail_fig.add_hline(
        y=cauchy_theory_tail,
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
        title="Robustness contrast inside Cauchy",
    )
    robust_fig.update_layout(legend_title_text="", xaxis_title="n (log scale)", yaxis_title="IQR")
    st.plotly_chart(robust_fig, use_container_width=True)

with tab_dirichlet:
    st.subheader("Presentation 2: Dirichlet Distribution")
    st.markdown(
        rf"""
**Interesting properties**
- Samples lie on the simplex: components are nonnegative and sum to 1.
- Components are negatively dependent; for symmetric Dirichlet, 
  $\mathrm{{Corr}}(X_1, X_2) = -1/(d-1)$.
- The concentration parameter $\alpha$ controls corner-vs-center behavior.

Current parameters: `d={dirichlet_d}`, `alpha={dirichlet_alpha:.2f}`.
"""
    )

    vis_df = dirichlet_df.sample(n=min(dirichlet_points, len(dirichlet_df)), random_state=int(seed))

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("max |sum - 1|", f"{dir_sum_error:.2e}")
    d2.metric("Empirical Corr(X1, X2)", f"{dir_emp_corr:.4f}")
    d3.metric("Theory Corr(X1, X2)", f"{dir_theory_corr:.4f}")
    d4.metric("E[max component]", f"{float(vis_df['max_component'].mean()):.4f}")

    if dirichlet_d == 3:
        ternary_fig = px.scatter_ternary(
            vis_df,
            a="x1",
            b="x2",
            c="x3",
            color="max_component",
            color_continuous_scale="Viridis",
            opacity=0.7,
            title="Simplex geometry (d=3): points move toward corners when sparse",
        )
        st.plotly_chart(ternary_fig, use_container_width=True)
    else:
        scatter_fig = px.scatter(
            vis_df,
            x="x1",
            y="x2",
            color="max_component",
            color_continuous_scale="Viridis",
            opacity=0.65,
            title="Negative dependence in components: X1 vs X2",
        )
        scatter_fig.update_layout(xaxis_title="X1", yaxis_title="X2")
        st.plotly_chart(scatter_fig, use_container_width=True)

    profile_fig = px.line(
        dir_profile_df,
        x="alpha",
        y="mean_max_component",
        markers=True,
        log_x=True,
        title="How alpha changes sparsity: E[max component]",
        color_discrete_sequence=["#ff7f0e"],
    )
    profile_fig.update_layout(xaxis_title="alpha (log scale)", yaxis_title="E[max component]")
    st.plotly_chart(profile_fig, use_container_width=True)

    entropy_fig = px.line(
        dir_profile_df,
        x="alpha",
        y="median_entropy",
        markers=True,
        log_x=True,
        title="How alpha changes spread: median entropy",
        color_discrete_sequence=["#9467bd"],
    )
    entropy_fig.update_layout(xaxis_title="alpha (log scale)", yaxis_title="Median entropy")
    st.plotly_chart(entropy_fig, use_container_width=True)

with tab_normal:
    st.subheader("Presentation 3: Normal Distribution")
    st.markdown(
        rf"""
**Interesting property**
If $Z_1,\ldots,Z_k$ are i.i.d. $\mathcal{{N}}(0,1)$, then
$\sum_{{i=1}}^k Z_i^2 \sim \chi^2_k$.

Here `k={normal_k}`.
"""
    )

    x_max = float(np.percentile(normal_r2, 99.8))
    x_grid = np.linspace(1e-4, max(x_max, 1.0), 400)
    pdf_grid = chi_square_pdf(x_grid, normal_k)

    hist = go.Histogram(
        x=normal_r2,
        histnorm="probability density",
        nbinsx=normal_bins,
        name="Simulated sum of squares",
        marker_color="#1f77b4",
        opacity=0.65,
    )
    pdf_line = go.Scatter(
        x=x_grid,
        y=pdf_grid,
        mode="lines",
        name="Chi-square pdf",
        line=dict(color="#d62728", width=3),
    )

    fig = go.Figure(data=[hist, pdf_line])
    fig.update_layout(
        title="Normal geometry: squared norm distribution",
        xaxis_title=r"Value of sum(Z_i^2)",
        yaxis_title="Density",
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Empirical mean", f"{np.mean(normal_r2):.4f}")
    n2.metric("Theory mean", f"{float(normal_k):.4f}")
    n3.metric("Empirical variance", f"{np.var(normal_r2):.4f}")
    n4.metric("Theory variance", f"{float(2 * normal_k):.4f}")

with tab_doc:
    st.subheader("Presentation 4: Dirichlet for Document Classification")
    st.markdown(
        rf"""
**Interesting property**
- Document topic proportions are drawn from a Dirichlet distribution.
- As base alpha increases, documents become more mixed, reducing class-separation signal.
- This demo trains a multinomial Naive Bayes classifier on synthetic bag-of-words documents.

Current setup: classes={doc_classes}, topics={doc_topics}, vocab={doc_vocab_size}, doc length={doc_length}.
"""
    )

    doc_random_baseline = 1.0 / doc_classes
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Accuracy at selected alpha", f"{doc_selected_acc:.4f}")
    d2.metric("Random baseline", f"{doc_random_baseline:.4f}")
    d3.metric("Mean dominant topic weight", f"{doc_dom_selected:.4f}")
    d4.metric("Mean topic entropy", f"{doc_entropy_selected:.4f}")

    acc_fig = px.line(
        doc_profile_df,
        x="alpha",
        y="accuracy",
        markers=True,
        log_x=True,
        title="Classification accuracy vs Dirichlet base alpha",
        color_discrete_sequence=["#1f77b4"],
    )
    acc_fig.add_hline(
        y=doc_random_baseline,
        line_dash="dash",
        line_color="#d62728",
        annotation_text="Random baseline",
    )
    acc_fig.add_vline(
        x=doc_alpha,
        line_dash="dot",
        line_color="#2ca02c",
        annotation_text="Selected alpha",
    )
    acc_fig.update_layout(xaxis_title="base alpha (log scale)", yaxis_title="Accuracy")
    st.plotly_chart(acc_fig, use_container_width=True)

    mix_fig = px.line(
        doc_profile_df,
        x="alpha",
        y=["mean_dominant_topic_weight", "mean_topic_entropy"],
        markers=True,
        log_x=True,
        title="How alpha changes topic structure",
    )
    mix_fig.update_layout(
        xaxis_title="base alpha (log scale)",
        yaxis_title="Value",
        legend_title_text="",
    )
    st.plotly_chart(mix_fig, use_container_width=True)

    conf_fig = px.imshow(
        doc_conf_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion matrix at selected alpha = {doc_alpha:.2f}",
        aspect="auto",
    )
    conf_fig.update_layout(xaxis_title="Predicted class", yaxis_title="True class")
    st.plotly_chart(conf_fig, use_container_width=True)

st.markdown("---")
st.caption("If you want, next I can add one more distribution with a distinct property (e.g., Student-t, Pareto, Lognormal).")
