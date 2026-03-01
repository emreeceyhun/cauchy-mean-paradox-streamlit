# Cauchy Mean Paradox Dashboard

A Streamlit dashboard that demonstrates a subtle but important statistical fact:

- For **Normal(0,1)** data, sample means stabilize near 0 as sample size increases.
- For **Cauchy(0,1)** data, sample means do **not** stabilize; the average remains Cauchy-distributed.

## What the app shows

- **Mean Distribution tab**: overlay histograms of sample means (Normal vs Cauchy).
- **Tail Invariance Across n tab**: empirical `P(|sample mean| > k)` across multiple `n` values plus the Cauchy theoretical line.
- **Robust Alternative tab**: compares Cauchy sample mean vs sample median using estimator IQR.
- Built-in simulation caps prevent memory blowups on Streamlit Community Cloud.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this project to a **public GitHub repo**.
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repo/branch and set main file path to `app.py`.
5. Deploy.

## Why this is interesting

Many people assume averaging always reduces noise. That is true for many distributions with finite variance, but false for heavy-tailed distributions like Cauchy.
