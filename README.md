# Q-Resilience web app

An interactive website that walks judges through the Q-Resilience project. Every number on the page is computed live by the notebook's code (`qres_core.py`), running on the Qiskit simulator.

## What's in this folder

| File | What it is |
| --- | --- |
| `app.py` | The website (Streamlit) |
| `qres_classical.py` | The notebook's classical functions: exact enumeration, Monte Carlo, the v1 oracle compiler |
| `qres_core.py` | The notebook's Qiskit functions: Ry state preparation, v0 oracle, IQAE, resource counts, noise study |
| `worker.py` | Helper process that runs all Qiskit work (avoids a Qiskit crash on macOS) |
| `figures/` | The four figures exported from the notebook |
| `requirements.txt` | Exact package versions the app was tested with |
| `.streamlit/config.toml` | Colors and settings |

## Put it online (free, about 10 minutes)

1. **Put this folder in a GitHub repo** (everything except `.venv`, including the hidden `.streamlit` folder and `figures`).
2. **Go to [share.streamlit.io](https://share.streamlit.io)** and sign in with the GitHub account that owns the repo.
3. Click **Create app → Deploy a public app from GitHub**, then fill in:
   - Repository: your repo
   - Branch: `main`
   - Main file path: `app.py` (or `app/app.py` if you put it in a subfolder)
   - App URL: pick something like `q-resilience`
4. Open **Advanced settings** and set **Python version 3.13**. Click **Deploy**.
5. The first build takes about 5 minutes (Qiskit is large). You'll get a link like `https://q-resilience.streamlit.app`.

## Before judging

- **Open the link 10 minutes before you present.** Free Streamlit apps go to sleep after a few days without visitors and take a minute to wake up.
- **Have a backup on your laptop** in case the Wi-Fi is bad:

```
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`.

## What each tab shows

1. **Problem**: the network, a "roll a random day" simulator, the formula, and why checking every scenario explodes.
2. **Classical**: exact enumeration (the answer key) and a live Monte Carlo convergence chart.
3. **Quantum circuit**: Ry angles for each link, the oracle tested on any failure pattern you pick, and the full circuit diagram.
4. **Amplitude estimation**: how the Grover operator magnifies the probability, and IQAE run live against Monte Carlo with the same budget.
5. **Head-to-head**: error vs number of runs (notebook Figure 1), rerun live.
6. **Noise**: depolarizing noise vs circuit depth (Figure 2), rerun live with Qiskit Aer.
7. **Scaling**: v0 vs v1 oracle cost on grid networks (Figures 3 and 4), plus cross-validation on 10 networks.
8. **Limits & Q&A**: honest limitations and answers to likely judge questions.

Each tab ends with a "What to tell judges" box. Turn on **Presenter notes** in the sidebar to see what to click, what to say and answers to likely questions on every tab.

## One implementation note

For speed, the web app compiles circuit A into one equivalent unitary before running IQAE (`fast=True` in `build_estimation_problem`). It's the same operator, so the estimates match the notebook exactly (for example 0.065423 with 1,900 runs on the default network). The noise study uses the original gates, so its circuit depths match the notebook.
