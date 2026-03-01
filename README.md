# Cauchy Mean Paradox Dashboard

A Streamlit dashboard that demonstrates a subtle but important statistical fact:

- For **Normal(0,1)** data, sample means stabilize near 0 as sample size increases.
- For **Cauchy(0,1)** data, sample means do **not** stabilize; the average remains Cauchy-distributed.

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
