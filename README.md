# Interesting Distribution Properties Dashboard

A Streamlit dashboard with separate mini-presentations, each focused on a distinct statistical property.

## What the app shows

Numbered, separate presentations:

1. **Cauchy: Mean Paradox**  
   Sample mean remains heavy-tailed and does not stabilize with larger sample size.
2. **Dirichlet: Simplex & Dependence**  
   Shows simplex constraint, negative component correlation, and alpha-controlled sparsity.
3. **Laplace: L1 Geometry**  
   Shows that Laplace location MLE corresponds to minimizing absolute error (sample median), with robustness implications.

The app includes memory-safe simulation caps for Streamlit Community Cloud.

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

It contrasts three different kinds of behavior:

- Heavy tails that break averaging intuition (Cauchy),
- Compositional dependence and sparsity control (Dirichlet),
- L1-driven robust location estimation (Laplace).
