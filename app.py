"""
Q-Resilience web app.
Every number on the page is computed live by the Q-Resilience notebook code (qres_core.py).
Run locally:  streamlit run app.py
"""
import math
import time
from pathlib import Path

import pickle
import subprocess
import sys
import threading
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import qres_classical as q   # no Qiskit in the web process; Qiskit runs in worker.py

HERE = Path(__file__).parent
FIG_DIR = HERE / "figures"
REPO_URL = "https://github.com/GamingBoyOfficial/Q-Resilience"

# Colors shared by every chart: classical = red, quantum = blue, exact = ink.
C_MC = "#da1e28"
C_Q = "#0f62fe"
C_V1 = "#8a3ffc"
C_OK = "#24a148"
C_EXACT = "#6f6f6f"

st.set_page_config(page_title="Q-Resilience", page_icon=":material/hub:", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stMarkdown, p, li, label, input, textarea, button { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3, h4 { font-family: 'IBM Plex Sans Condensed', 'IBM Plex Sans', sans-serif !important; letter-spacing: -0.01em; text-wrap: balance; }
code, .mono { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }
.block-container { padding-top: 3.6rem; max-width: 1240px; }
.hero-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; color: #0f62fe; margin-bottom: 0.2rem; }
.hero-title { font-family: 'IBM Plex Sans Condensed', sans-serif; font-weight: 700; font-size: clamp(2.4rem, 5vw, 3.6rem); line-height: 1.02; margin: 0; }
.hero-sub { font-size: 1.15rem; opacity: 0.82; max-width: 62ch; margin-top: 0.6rem; }
.ket { font-family: 'IBM Plex Mono', monospace; color: #0f62fe; }
.callout { border-left: 3px solid #0f62fe; padding: 0.6rem 0.9rem; background: rgba(15, 98, 254, 0.06); border-radius: 0 6px 6px 0; margin: 0.4rem 0 0.9rem; }
.callout.red { border-color: #da1e28; background: rgba(218, 30, 40, 0.06); }
.callout.violet { border-color: #8a3ffc; background: rgba(138, 63, 252, 0.07); }
.say { font-size: 0.95rem; border: 1px dashed rgba(128,128,128,0.45); border-radius: 6px; padding: 0.55rem 0.8rem; margin-top: 0.8rem; }
.say b { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.7; display: block; margin-bottom: 0.15rem; }
div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }
.stTabs [data-baseweb="tab-list"] { gap: 0.2rem; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { padding: 0.45rem 0.8rem; }
</style>
""",
    unsafe_allow_html=True,
)


def callout(text, kind=""):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def say(text):
    if not st.session_state.get("presenter"):
        return
    st.markdown(f'<div class="say"><b>What to tell judges</b>{text}</div>', unsafe_allow_html=True)


def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------------------
# Network input (sidebar)
# ---------------------------------------------------------------------------
MAX_LINKS = 12
UPLOAD = "Upload or paste a CSV"

with st.sidebar:
    st.markdown("### Network")
    choice = st.selectbox("Choose a network", list(q.PRESETS) + [UPLOAD], key="preset")
    if choice == UPLOAD:
        up = st.file_uploader("CSV with columns u, v, p_fail", type=["csv", "txt"])
        pasted = st.text_area("…or paste CSV text", "u,v,p_fail\ns,a,0.1\na,t,0.15\ns,b,0.2\nb,t,0.1", height=130)
        text = up.getvalue().decode("utf-8", "replace") if up else pasted
        try:
            import csv, io
            rows = list(csv.DictReader(io.StringIO(text.strip())))
            base_df = pd.DataFrame([{"u": r["u"].strip(), "v": r["v"].strip(), "p_fail": float(r["p_fail"])} for r in rows])
        except Exception:
            st.error("Couldn't read that CSV. It needs a header row: u,v,p_fail")
            st.stop()
        default_s, default_t = base_df["u"].iloc[0], base_df["v"].iloc[-1]
        st.caption("Each row is one link and its chance of failing.")
    else:
        P = q.PRESETS[choice]
        base_df = pd.DataFrame([{"u": str(u), "v": str(v), "p_fail": p} for (u, v), p in P["edges"].items()])
        default_s, default_t = str(P["source"]), str(P["target"])
        st.caption(P["blurb"])

    st.markdown("**Links and failure chances** · edit any cell")
    df = st.data_editor(
        base_df, num_rows="dynamic", width="stretch", hide_index=True, key=f"ed_{choice}",
        column_config={"p_fail": st.column_config.NumberColumn("p_fail", min_value=0.001, max_value=0.999, step=0.01, format="%.3f")},
    )
    df = df.dropna()
    nodes = sorted(set(df["u"].astype(str)) | set(df["v"].astype(str)), key=str)
    if len(nodes) < 2:
        st.error("Add at least one link.")
        st.stop()
    c1, c2 = st.columns(2)
    src = c1.selectbox("Source", nodes, index=nodes.index(default_s) if default_s in nodes else 0)
    tgt = c2.selectbox("Target", nodes, index=nodes.index(default_t) if default_t in nodes else len(nodes) - 1)

    csv_text = "u,v,p_fail\n" + "\n".join(f"{r.u},{r.v},{float(r.p_fail)}" for r in df.itertuples())
    try:
        probs = q.load_network_csv(csv_text, src, tgt)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    if len(probs) > MAX_LINKS:
        st.error(f"This web demo handles up to {MAX_LINKS} links, because exact enumeration and the v0 oracle "
                 f"check 2^m scenarios. You have {len(probs)}.")
        st.stop()
    st.divider()
    PRESENTER = st.toggle("Presenter notes", value=False, key="presenter",
                          help="Shows what to click, what to say and how to explain each tab. Turn off before judges look at the screen.")
    st.caption("Every number in this app is computed live by the Q-Resilience notebook code, "
               "running on the Qiskit simulator.")
    st.markdown(f"[Source code on GitHub]({REPO_URL})")

NET = tuple(sorted((u, v, round(float(p), 6)) for (u, v), p in probs.items()))
G = q.build_graph(probs)
EDGES = q.ordered_edges(G)
M = len(EDGES)
EDGE_P = [q.failure_probability(probs, e) for e in EDGES]


def net_probs(net):
    return {(u, v): p for u, v, p in net}


# ---------------------------------------------------------------------------
# Cached computations (all call the notebook code)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def c_exact(net, s, t):
    p = net_probs(net)
    g = q.build_graph(p)
    edges = q.ordered_edges(g)
    pr = [q.failure_probability(p, e) for e in edges]
    masks = q.disconnected_masks(g, edges, s, t)
    return q.exact_enumeration(g, p, s, t), [(m, q.mask_probability(m, pr)) for m in masks]


@st.cache_data(show_spinner=False)
def c_mc(net, s, t, n, seed):
    p = net_probs(net)
    return q.monte_carlo_simulation(q.build_graph(p), p, s, t, n, seed)


@st.cache_data(show_spinner=False)
def c_mc_running(net, s, t, n, seed):
    p = net_probs(net)
    return q.monte_carlo_running(q.build_graph(p), p, s, t, n, seed)


# All Qiskit work runs in one long-lived helper process (worker.py). Qiskit's compiled
# core can crash on macOS when called from the new threads Streamlit starts on each click;
# in the helper it always runs on the same main thread, and a crash there can't take the site down.
class _QWorker:
    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, str(HERE / "worker.py")], cwd=str(HERE),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.lock = threading.Lock()

    def call(self, fn, *args):
        with self.lock:
            pickle.dump((fn, args), self.proc.stdin)
            self.proc.stdin.flush()
            return pickle.load(self.proc.stdout)

    def stop(self):
        try:
            self.proc.kill()
        except Exception:
            pass


@st.cache_resource(show_spinner=False)
def _qworker():
    return _QWorker()


def run_q(fn, *args):
    """Run worker.<fn>(*args) in the helper process; restart the helper once if it died."""
    for _ in range(2):
        w = _qworker()
        try:
            ok, result = w.call(fn, *args)
        except (EOFError, BrokenPipeError, OSError, pickle.UnpicklingError):
            w.stop()
            _qworker.clear()
            continue
        if ok:
            return result
        raise RuntimeError(result)
    st.error("The quantum simulator stopped unexpectedly. Reload the page to try again.")
    st.stop()


@st.cache_data(show_spinner=False)
def c_iqae(net, s, t, eps, alpha, shots, seed):
    return run_q("iqae", net, s, t, eps, alpha, shots, seed)


@st.cache_data(show_spinner=False)
def c_profile(net, s, t):
    return run_q("profile", net, s, t)


@st.cache_data(show_spinner=False)
def c_v1(net, s, t):
    return run_q("v1", net, s, t)


@st.cache_data(show_spinner=False)
def c_v0_amp(net, s, t):
    return run_q("v0_amp", net, s, t)


@st.cache_data(show_spinner=False)
def c_amplification(net, s, t, kmax):
    return run_q("amplification", net, s, t, kmax)


@st.cache_data(show_spinner=False)
def c_oracle_on(net, s, t, mask):
    return run_q("oracle_on", net, s, t, mask)


@st.cache_data(show_spinner=False)
def c_noise(net, s, t, k, rho):
    return run_q("noise", net, s, t, k, rho)


@st.cache_data(show_spinner=False)
def c_circuit_png(net, s, t):
    return run_q("circuit_png", net, s, t)


@st.cache_data(show_spinner=False)
def c_crossval():
    return run_q("crossval")


@st.cache_data(show_spinner=False)
def c_grid_scaling():
    rows = []
    for r_, c_ in [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4)]:
        g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(r_, c_))
        p = {(u, v): 0.1 for u, v in g.edges()}
        s, t, m = 0, g.number_of_nodes() - 1, g.number_of_edges()
        b, _ = q.build_reachability_state_prep(g, p, s, t)
        stv = b.stats()
        if m <= 12:
            nm, v0t, v0q = q.v0_cost(g, s, t)
        else:
            nm, v0t, v0q = None, None, m + 1
        rows.append({"grid": f"{r_}×{c_}", "links": m, "v0 scenarios marked": nm, "v0 qubits": v0q,
                     "v0 Toffoli cost": v0t, "v1 qubits": stv["qubits"], "v1 Toffoli cost": stv["toffoli_equiv"]})
    return rows


# ---------------------------------------------------------------------------
# Network drawing
# ---------------------------------------------------------------------------
def layout(g):
    """Source on the left, target on the right: x = hops from the source, nodes spread vertically per layer."""
    if choice == "2 x 3 grid" and all(str(n).isdigit() for n in g.nodes()):
        return {n: (int(n) % 3, -(int(n) // 3)) for n in g.nodes()}
    dist = nx.single_source_shortest_path_length(g, src)
    if tgt in dist:
        far = max(dist.values())
        dist[tgt] = max(far, dist[tgt])
        for n in g.nodes():
            if n != tgt and dist.get(n, far) >= dist[tgt]:
                dist[tgt] = dist.get(n, far) + 1
    layers = {}
    for n in g.nodes():
        layers.setdefault(dist.get(n, 0), []).append(n)
    pos = {}
    for x, members in layers.items():
        members = sorted(members, key=str)
        for i, n in enumerate(members):
            pos[n] = (x, (len(members) - 1) / 2 - i)
    return pos


POS = layout(G)


def draw_network(failed_mask=None, height=380):
    fig = go.Figure()
    reach = None
    if failed_mask is not None:
        alive = nx.Graph()
        alive.add_nodes_from(G.nodes())
        alive.add_edges_from(e for i, e in enumerate(EDGES) if not (failed_mask >> i) & 1)
        reach = nx.node_connected_component(alive, src)
    for i, (u, v) in enumerate(EDGES):
        (x0, y0), (x1, y1) = POS[u], POS[v]
        broken = failed_mask is not None and (failed_mask >> i) & 1
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", hoverinfo="skip", showlegend=False,
            line=dict(color=C_MC if broken else "rgba(120,130,150,0.75)", width=2.5 if broken else 3.5,
                      dash="dot" if broken else "solid")))
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, showarrow=False,
                           text=f"<b>{'✕ ' if broken else ''}{pct(EDGE_P[i], 0)}</b>",
                           font=dict(size=13, color=C_MC if broken else "#393939", family="IBM Plex Mono, monospace"),
                           bgcolor="rgba(255,255,255,0.85)", borderpad=2)
    xs, ys, labels, colors, sizes = [], [], [], [], []
    for n in G.nodes():
        xs.append(POS[n][0]); ys.append(POS[n][1]); labels.append(str(n))
        if n == src:
            colors.append(C_Q); sizes.append(30)
        elif n == tgt:
            colors.append(C_V1); sizes.append(30)
        elif reach is not None and n in reach:
            colors.append(C_OK); sizes.append(22)
        else:
            colors.append("#8d8d8d"); sizes.append(22)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=labels, textposition="top center",
                             marker=dict(size=sizes, color=colors, symbol="circle", line=dict(width=2, color="white")),
                             textfont=dict(size=15, family="IBM Plex Sans, sans-serif"), hoverinfo="text", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False, range=[min(p[0] for p in POS.values()) - 0.35, max(p[0] for p in POS.values()) + 0.35]),
                      yaxis=dict(visible=False, range=[min(p[1] for p in POS.values()) - 0.6, max(p[1] for p in POS.values()) + 0.6]),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def mask_links(mask):
    return [f"{u}–{v}" for i, (u, v) in enumerate(EDGES) if (mask >> i) & 1]


def mask_bits(mask):
    """Bitstring in qubit order q0 q1 ... (left = qubit 0)."""
    return "".join(str((mask >> i) & 1) for i in range(M))


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
EXACT, BAD = c_exact(NET, src, tgt)
with st.spinner("Running IQAE on the quantum simulator…"):
    IQ = c_iqae(NET, src, tgt, 0.02, 0.05, 100, 7)
MC20 = c_mc(NET, src, tgt, 20_000, 7)

st.markdown('<div class="hero-eyebrow">IBM Qiskit Fall Fest 2026 · Network reliability × quantum amplitude estimation</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Q-Resilience</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="hero-sub">What is the chance that <b>{tgt}</b> gets cut off from <b>{src}</b> when links fail at random? '
    f"We answer it three ways: exact enumeration, classical Monte Carlo sampling, and Iterative Quantum Amplitude "
    f"Estimation (IQAE) built in Qiskit. Pick or edit a network in the sidebar; everything below recomputes.</p>",
    unsafe_allow_html=True,
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Links → failure scenarios", f"{M} → {2 ** M:,}")
m2.metric("Exact P(cut off)", pct(EXACT, 3))
m3.metric("Monte Carlo · 20,000 samples", pct(MC20, 3), f"error {abs(MC20 - EXACT) * 100:.3f} pts", delta_color="off", delta_arrow="off")
m4.metric(f"IQAE · {IQ['calls']:,} circuit runs", pct(IQ["est"], 3), f"error {abs(IQ['est'] - EXACT) * 100:.3f} pts",
          delta_color="off", delta_arrow="off")


def presenter(key):
    """Speaker notes for one tab, shown only when the sidebar switch is on."""
    if not st.session_state.get("presenter"):
        return
    n = NOTES[key]
    with st.container(border=True):
        st.markdown(f"#### 🎤 Presenter notes · {n['title']}")
        c1, c2 = st.columns([1, 1.35], gap="large")
        with c1:
            st.markdown("**Do this**")
            st.markdown(n["click"])
            st.markdown("**What's on the screen**")
            st.markdown(n["explain"])
        with c2:
            st.markdown("**Say this**")
            st.markdown(n["say"])
            st.markdown("**If a judge asks…**")
            st.markdown(n["ask"])


TWO_K = 2 ** M
NOTES = {
    "problem": {
        "title": "Introduce the problem",
        "click": "1. Click **Roll a random day** 2–3 times.\n"
                 "2. In the sidebar, switch to **Mini power grid** to show a realistic case.\n"
                 "3. Point at the four numbers at the top.",
        "explain": "- The picture is the network. Each label is that link's chance of failing on a given day.\n"
                   "- Red dotted links are broken on this simulated day; green nodes can still be reached from the source.\n"
                   f"- With {M} links there are 2^{M} = {TWO_K:,} possible combinations of broken links.\n"
                   "- The top numbers: the true answer (exact), the normal estimate (Monte Carlo) and our quantum estimate (IQAE).",
        "say": f"\"Networks like power grids lose links all the time: a storm takes out a line, a cable gets cut. Usually there's a backup route. "
               f"But sometimes enough links fail that something important gets cut off. Here, {src} is the source and {tgt} is the target. "
               f"Every extra link doubles the number of scenarios, so for real networks you can't check them all. "
               f"Our tool estimates **how likely** it is that {tgt} gets cut off, and tests whether a quantum computer can do it with fewer runs.\"",
        "ask": "- **Why assume links fail independently?** It's a standard first model. Real failures can be linked, like one storm hitting several lines; that's future work.\n"
               "- **Where do the failure chances come from?** In practice, from historical outage data. Here they're illustrative numbers you can edit in the sidebar.",
    },
    "classical": {
        "title": "The two classical methods",
        "click": "1. Point at the table of cut-off scenarios.\n"
                 "2. Drag the **Number of simulated days** slider left, then right.",
        "explain": "- **Exact enumeration**: the table lists every failure combination that cuts the target off. Adding their probabilities gives the true answer.\n"
                   "- **Monte Carlo**: the red line is the estimate as more random days are simulated. The dashed line is the true value.\n"
                   "- The pink band is where Monte Carlo is expected to land 95% of the time. It narrows slowly.",
        "say": "\"Our first method, exact enumeration, checks every combination. It's perfect, so we use it as our answer key, "
               "but it only works for small networks. The second, Monte Carlo, is what engineers actually use: simulate random days and count. "
               "It works on any size, but it improves slowly: to be 10 times more accurate it needs 100 times more samples.\"",
        "ask": "- **Aren't there smarter classical methods?** Yes, for example cut-set methods and importance sampling. We compare against plain Monte Carlo; comparing against the best classical methods is future work.\n"
               "- **What does 1/√N mean?** Error shrinks with the square root of the samples: 100× the samples gives 10× the accuracy.",
    },
    "circuit": {
        "title": "Turning the network into a quantum circuit",
        "click": "1. Point at the qubit table.\n"
                 "2. In **Failed links**, pick links that cut the target off (oracle shows **1**).\n"
                 "3. Remove one of them (oracle flips to **0**).",
        "explain": "- One qubit per link. The Ry rotation angle θ = 2·arcsin(√p) makes the qubit read 1 (failed) with exactly that link's chance p.\n"
                   "- The **oracle** flips one extra qubit (the objective) to 1 for every combination that cuts the target off.\n"
                   "- The bottom-right number is the exact probability that the objective reads 1. It equals the true answer.",
        "say": "\"Each link becomes a qubit, rotated so it reads 'broken' with that link's real chance. So every scenario is in the circuit at once, "
               "with the right weight. Then the oracle flips one extra qubit for every scenario where the target is cut off. Watch: I break these links, "
               "the oracle says 1, and a normal search agrees. I fix one, it says 0. So the chance that extra qubit reads 1 is exactly our answer.\"",
        "ask": "- **Why the square root?** A qubit's probability is its amplitude squared, so the amplitude must be √p.\n"
               "- **Isn't the oracle cheating if it lists the bad scenarios?** This simple version (v0) does need that list. That's why we built v1, shown in tab ⑦, which works it out inside the circuit.",
    },
    "ae": {
        "title": "Amplitude estimation, the key idea",
        "click": "1. Move the **Apply Q up to k times** slider.\n"
                 "2. Change **Target accuracy** and point at IQAE vs Monte Carlo.",
        "explain": "- We write the answer as P = sin²θ. The Grover operator Q rotates by 2θ each time, so after k applications the chance becomes sin²((2k+1)θ).\n"
                   "- Blue dots: the real simulated circuit. Dotted line: the theory. They match.\n"
                   "- Bottom chart: IQAE vs Monte Carlo **with the same number of runs**. The dashed line is the true value.",
        "say": f"\"Measuring that qubit over and over would just be Monte Carlo again. Instead, the Grover operator magnifies the small probability "
               f"into big, easy-to-measure numbers. It's like measuring the thickness of paper: stack 100 sheets, measure, divide. "
               f"IQAE does this in rounds, narrowing down the answer. On this network it used {IQ['calls']:,} runs and got {pct(IQ['est'], 2)} "
               f"against the true {pct(EXACT, 2)}. Monte Carlo used 20,000 samples and got {pct(MC20, 2)}.\"",
        "ask": "- **Why IQAE instead of the original algorithm?** The original needs a quantum Fourier transform and many extra qubits. IQAE needs only the Grover operator plus classical post-processing.\n"
               "- **How do you count runs?** Strictly: each shot after k Grover steps runs the circuit 2k+1 times.",
    },
    "h2h": {
        "title": "The speed comparison (Figure 1)",
        "click": "1. Click **Run the sweep live** (do it before judges arrive so it's instant).\n"
                 "2. Point at the red and blue lines.",
        "explain": "- Both axes are logarithmic: each grid line is 10×.\n"
                   "- Red: Monte Carlo error vs samples (theory slope −0.5). Blue: IQAE's guaranteed error vs runs (theory slope −1).\n"
                   "- The dotted and dashed grey lines are the ideal 1/√N and 1/N references.",
        "say": "\"As we give each method more runs, the quantum error falls about twice as steeply as Monte Carlo's: slope about −1 versus −0.5. "
               "That means the same accuracy with quadratically fewer runs. That's the theoretical advantage, and we see it in our own results.\"",
        "ask": "- **Is that a proof?** No. It's a trend from a few runs on a small network. The theory says 1/N; our data agrees.\n"
               "- **Does fewer runs mean faster?** Not necessarily. Each quantum run is a deep circuit, much more expensive than one classical sample.",
    },
    "noise": {
        "title": "Being honest about noise",
        "click": "1. Make sure **Two-route demo** is selected.\n"
                 "2. Click **Run noisy simulations** and point at the lines drifting away.",
        "explain": "- Depolarizing noise: after each gate, with chance ρ, a qubit gets scrambled. Two-qubit gates get twice the noise.\n"
                   "- Each colour is a number of Grover steps k. More steps means a deeper circuit (see the depth table).\n"
                   "- Dotted lines: what you'd get if noise scrambled everything. A curve on its dotted line carries no information.",
        "say": "\"Real quantum computers make errors. The deeper circuits that give us the speedup break first: even at a 0.1% error rate per gate, "
               "the k = 2 estimate is ruined. That's why this doesn't beat classical computers on today's hardware.\"",
        "ask": "- **So will it ever beat classical?** Possibly, once we have error-corrected hardware, large enough networks, and an oracle that stays small. It's an open question, and our tool measures it.\n"
               "- **Why does k = 1 look fine at high noise?** Coincidence: fully scrambled, k = 1 happens to land near the true value. The app flags this.",
    },
    "scaling": {
        "title": "Our two oracles and how we validated them",
        "click": "1. Point at the chart.\n"
                 "2. Scroll to the **Cross-validation** table.",
        "explain": "- **v0** (red) builds one gate per bad scenario, which needs a classical list of them first. Its gate count explodes.\n"
                   "- **v1** (purple) runs breadth-first search inside the circuit: a node is reached if it already was, or a neighbour was AND the link works. Fewer gates, more qubits.\n"
                   "- The table checks both oracles against the exact answer on 10 networks. Errors around 10⁻¹⁴ are just rounding.",
        "say": "\"Our first oracle, v0, needs a list of every bad scenario, and making that list is as hard as the original problem. So we built v1, "
               "which works out connectivity inside the circuit, like power spreading through the network one step at a time. It trades gates for qubits. "
               "We tested both on 10 networks, and they match the exact answer perfectly.\"",
        "ask": "- **Why |V| − 1 rounds?** The longest route that doesn't revisit a node has at most that many links.\n"
               "- **Why not clean up the extra qubits?** It would roughly double the gate count. We chose fewer gates.",
    },
    "limits": {
        "title": "Wrap up and questions",
        "click": "Open the judge questions below if someone asks one of them.",
        "explain": "- Left: our limits and what we built.\n- Right: answers to likely questions.",
        "say": "\"To wrap up: we built a working tool that estimates how likely a network is to get cut off, with three methods side by side. "
               "Our quantum circuits match the exact answer on 10 networks, and in simulation the quantum method needed about 10 times fewer runs for better accuracy. "
               "Next, we want to model failures that happen together, like storms, and test on real infrastructure data as hardware improves. Thank you!\"",
        "ask": "- **Does it beat classical today?** Not on real hardware yet; noise is the main barrier. In simulation it needed about 10× fewer runs.\n"
               "- **What's new here?** Amplitude estimation for reliability has been studied before. Ours is a working, cross-checked tool: CSV in, risk out, with two oracles, a noise study and resource estimates.",
    },
}

tabs = st.tabs(["① Problem", "② Classical", "③ Quantum circuit", "④ Amplitude estimation",
                "⑤ Head-to-head", "⑥ Noise", "⑦ Scaling", "⑧ Limits & Q&A"])

# ---------------------------------------------------------------------------
# 1. Problem
# ---------------------------------------------------------------------------
with tabs[0]:
    presenter("problem")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Simulate one day on this network")
        if "day_seed" not in st.session_state:
            st.session_state.day_seed = 0
        b1, b2 = st.columns([1, 1])
        if b1.button("Roll a random day", type="primary", width="stretch"):
            st.session_state.day_seed += 1
        if b2.button("Everything working", width="stretch"):
            st.session_state.day_seed = 0
        if st.session_state.day_seed == 0:
            day_mask = 0
        else:
            rng = np.random.default_rng(1000 + st.session_state.day_seed)
            day_mask = int(sum(1 << i for i, b in enumerate(rng.random(M) < np.array(EDGE_P)) if b))
        st.plotly_chart(draw_network(day_mask), width="stretch", config={"displayModeBar": False})
        cut = q.is_disconnected(G, EDGES, day_mask, src, tgt)
        broken = mask_links(day_mask)
        if cut:
            callout(f"<b>{tgt} is cut off.</b> Failed links: {', '.join(broken)}.", "red")
        else:
            callout(f"<b>{tgt} still reaches {src}.</b> "
                    + (f"Failed links: {', '.join(broken)}." if broken else "No links failed.")
                    + " Green nodes are still reachable.")
        st.caption("Blue = source, violet = target. Each label is that link's chance of failing on a given day.")
    with right:
        st.subheader("The question")
        st.markdown(
            f"Power grids, telecom backbones, water pipes and supply chains are networks whose links fail. "
            f"Engineers need the **disconnection probability** $P_{{disc}}$: how likely it is that a critical "
            f"node ({tgt}) loses its connection to the source ({src}) when each link fails independently."
        )
        st.latex(r"P_{\mathrm{disc}} = \sum_{x \,\in\, \text{cut-off scenarios}} \;\prod_{e} p_e^{x_e}(1-p_e)^{1-x_e}")
        st.markdown(
            f"Each scenario $x$ is a list of which links failed. With {M} links there are $2^{{{M}}} = {2 ** M:,}$ "
            f"scenarios, and **{len(BAD)}** of them cut {tgt} off. Their probabilities add up to "
            f"**{pct(EXACT, 3)}**."
        )
        if choice.startswith("Two-route") and len(probs) == 4:
            callout("<b>Check it by hand.</b> Route s→a→t fails with chance 1 − 0.90×0.85 = 0.235. "
                    "Route s→b→t fails with chance 1 − 0.80×0.90 = 0.28. The town is cut off only if both fail: "
                    "0.235 × 0.28 = <b>0.0658</b>, exactly what the code computes.")
        st.markdown("**Why it gets hard:** every extra link doubles the number of scenarios.")
        ms = np.arange(4, 61, 4)
        fig = go.Figure(go.Scatter(x=ms, y=2.0 ** ms, mode="lines+markers", line=dict(color=C_MC, width=2.5)))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), yaxis_type="log",
                          xaxis_title="links in the network", yaxis_title="scenarios to check")
        fig.add_annotation(x=40, y=2.0 ** 40, text="40 links → 1.1 trillion", showarrow=True, ay=-30, ax=-60)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    say("Real infrastructure networks have hundreds of links, so checking every failure scenario is impossible. "
        "Engineers fall back on random sampling. We ask whether a quantum computer can estimate the same probability with fewer runs.")

# ---------------------------------------------------------------------------
# 2. Classical
# ---------------------------------------------------------------------------
with tabs[1]:
    presenter("classical")
    st.subheader("Baseline 1 · Exact enumeration (the answer key)")
    a, b = st.columns([1, 1.25], gap="large")
    with a:
        st.markdown(
            "Each scenario is stored as a **bitmask**: bit *i* = 1 means link *i* failed. The code loops over all "
            f"$2^{{{M}}}$ masks, runs a **depth-first search** from {src} over the links that still work "
            f"(`_is_disconnected`), and adds up the probability of every mask where the search never reaches {tgt}."
        )
        st.code("for mask in range(1 << m):           # every scenario\n"
                "    if _is_disconnected(G, edges, mask, s, t):\n"
                "        total += prod(p if failed else 1 - p)", language="python")
        st.metric("Exact disconnection probability", f"{EXACT:.6f}")
    with b:
        top = sorted(BAD, key=lambda x: -x[1])[:12]
        st.markdown(f"**The {len(BAD)} cut-off scenarios** (most likely first{', top 12 shown' if len(BAD) > 12 else ''})")
        st.dataframe(pd.DataFrame([{"failed links": ", ".join(mask_links(mk)), "bits (q0…)": mask_bits(mk),
                                    "probability": pr} for mk, pr in top]),
                     hide_index=True, width="stretch",
                     column_config={"probability": st.column_config.NumberColumn(format="%.5f")})
    st.divider()
    st.subheader("Baseline 2 · Monte Carlo sampling (what engineers use today)")
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        n_mc = st.select_slider("Number of simulated days", options=[100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000],
                                value=30_000)
        seed_mc = st.number_input("Random seed", 0, 9999, 7, key="mcseed")
        run = c_mc_running(NET, src, tgt, n_mc, seed_mc)
        st.metric("Monte Carlo estimate", f"{run[-1]:.6f}", f"error {abs(run[-1] - EXACT):.6f}", delta_color="off", delta_arrow="off")
        callout("Error shrinks like <b>1/√N</b>. To get <b>10×</b> more accurate you need <b>100×</b> more samples.", "red")
    with c2:
        idx = np.unique(np.logspace(0, np.log10(n_mc), 400).astype(int)) - 1
        x = idx + 1
        band = 1.96 * np.sqrt(EXACT * (1 - EXACT) / x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[EXACT + band, (EXACT - band)[::-1]], fill="toself",
                                 fillcolor="rgba(218,30,40,0.10)", line=dict(width=0), name="95% range expected", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=run[idx], name="running estimate", line=dict(color=C_MC, width=2)))
        fig.add_hline(y=EXACT, line=dict(color=C_EXACT, dash="dash"), annotation_text=f"exact {EXACT:.4f}")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log",
                          xaxis_title="simulated days (log scale)", yaxis_title="estimated P(cut off)",
                          yaxis_range=[max(0, EXACT - 0.06), min(1, EXACT + 0.06)],
                          legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    say("Exact enumeration is our ground truth for small networks. Monte Carlo scales to big ones, but its error only "
        "shrinks with the square root of the number of samples. That square root is what the quantum method attacks.")

# ---------------------------------------------------------------------------
# 3. Quantum circuit
# ---------------------------------------------------------------------------
with tabs[2]:
    presenter("circuit")
    st.subheader("Step 1 · Load every link's failure chance into a qubit")
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown(
            "One qubit per link. A qubit starts at 0 (*working*). An $R_y(\\theta)$ rotation makes it read 1 "
            "(*failed*) with probability $\\sin^2(\\theta/2)$, so we pick"
        )
        st.latex(r"\theta_e = 2\arcsin\sqrt{p_e} \quad\Rightarrow\quad R_y(\theta_e)|0\rangle=\sqrt{1-p_e}\,|0\rangle+\sqrt{p_e}\,|1\rangle")
        st.markdown(f"After these {M} rotations, measuring the qubits gives every failure scenario with exactly its "
                    f"real-world probability. **All {2 ** M:,} scenarios live in the register at once.**")
    with b:
        st.dataframe(pd.DataFrame([{"qubit": f"q{i}", "link": f"{u}–{v}", "p_fail": p,
                                    "θ (radians)": 2 * math.asin(math.sqrt(p)), "amplitude of |1⟩": math.sqrt(p)}
                                   for i, ((u, v), p) in enumerate(zip(EDGES, EDGE_P))]),
                     hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["p_fail", "θ (radians)", "amplitude of |1⟩"]})
    st.divider()
    st.subheader("Step 2 · The oracle marks every cut-off scenario")
    a, b = st.columns([1, 1.1], gap="large")
    with a:
        st.markdown(
            f"One extra **objective qubit** (q{M}) is flipped to 1 for each of the {len(BAD)} cut-off scenarios. "
            "For each scenario the v0 oracle does three things:"
        )
        st.markdown("1. **X** the link qubits that should be 0, so that exact pattern becomes all 1s.\n"
                    "2. **MCX**: flip the objective only if all link qubits are 1 (a big AND).\n"
                    "3. **X** again to undo step 1.")
        callout(f"After the circuit runs: <b>P(objective = 1) = P<sub>disc</sub></b>. That one sentence is the whole bridge "
                "between the network problem and the quantum computer.")
    with b:
        if M <= 7:
            st.markdown("**Try the oracle on one scenario** · pick which links fail")
            picks = st.multiselect("Failed links", [f"{u}–{v}" for u, v in EDGES], default=[f"{u}–{v}" for u, v in EDGES][:1] + [f"{u}–{v}" for u, v in EDGES][2:3])
            mask = sum(1 << i for i, (u, v) in enumerate(EDGES) if f"{u}–{v}" in picks)
            out = c_oracle_on(NET, src, tgt, mask)
            classical = q.is_disconnected(G, EDGES, mask, src, tgt)
            k1, k2 = st.columns(2)
            k1.metric("Oracle output (objective qubit)", f"{int(round(out))}")
            k2.metric("Depth-first search says", "cut off" if classical else "connected")
            agree = int(round(out)) == int(classical)
            msg = (f"Input bits {mask_bits(mask)} → objective qubit {int(round(out))}. "
                   + ("The quantum oracle agrees with the classical search." if agree else "Mismatch."))
            if agree:
                callout(msg, "violet")
            else:
                st.error(msg)
        else:
            st.info("The scenario explorer is shown for networks with up to 7 links.")
    if M <= 5:
        with st.expander("Show the full circuit A (state preparation + oracle)", expanded=M <= 4):
            st.image(c_circuit_png(NET, src, tgt), width="stretch")
    prof = c_profile(NET, src, tgt)
    v0_amp = c_v0_amp(NET, src, tgt)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Qubits", prof["total_qubits"])
    k2.metric("Oracle terms (MCX gates)", prof["oracle_minterms"])
    k3.metric("Circuit depth", prof["state_prep_depth"])
    k4.metric("Exact P(objective=1) from statevector", f"{v0_amp:.6f}", f"vs exact {EXACT:.6f}", delta_color="off", delta_arrow="off")
    say("Each link is a qubit rotated so it fails with its real probability. The oracle flips one extra qubit for every "
        "disconnecting scenario, so the chance that qubit reads 1 is exactly the disconnection probability. "
        "We verify that with a full statevector.")

# ---------------------------------------------------------------------------
# 4. Amplitude estimation
# ---------------------------------------------------------------------------
with tabs[3]:
    presenter("ae")
    theta = math.asin(math.sqrt(EXACT))
    st.subheader("Why not just measure the objective qubit many times?")
    st.markdown("That would be Monte Carlo again. Amplitude estimation reads the probability more cleverly.")
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        st.markdown(f"Write the probability as an angle: $P_{{disc}} = \\sin^2\\theta$, so here "
                    f"$\\theta = {theta:.4f}$ rad. The **Grover operator Q** (built automatically by Qiskit from circuit A) "
                    "rotates the state by $2\\theta$ each time it is applied:")
        st.latex(r"P\big(\text{objective}=1 \text{ after } Q^k A\big) = \sin^2\!\big((2k+1)\,\theta\big)")
        callout("<b>Analogy:</b> to measure the thickness of one sheet of paper, stack 100 sheets, measure the stack and "
                "divide. Applying Q is stacking: the angle is multiplied by (2k+1), so any measuring error shrinks by the same factor.")
        kmax = st.slider("Apply Q up to k times", 0, 8, 4)
    with b:
        sim = c_amplification(NET, src, tgt, kmax)
        ks = np.arange(kmax + 1)
        kk = np.linspace(0, kmax, 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=kk, y=np.sin((2 * kk + 1) * theta) ** 2, name="theory sin²((2k+1)θ)",
                                 line=dict(color=C_EXACT, dash="dot")))
        fig.add_trace(go.Scatter(x=ks, y=sim, mode="markers+text", name="simulated circuit Qᵏ A",
                                 marker=dict(size=13, color=C_Q), text=[f"{v:.2f}" for v in sim], textposition="top center"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="k (Grover applications)",
                          yaxis_title="P(objective qubit = 1)", yaxis_range=[0, 1.12], legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.divider()
    st.subheader("Run IQAE live")
    st.markdown("**Iterative** amplitude estimation works in rounds: measure at some k, narrow the range for θ, pick a "
                "bigger k that can't be confused, repeat until the range is smaller than the target accuracy.")
    c1, c2, c3, c4 = st.columns(4)
    eps = c1.select_slider("Target accuracy ε", [0.05, 0.02, 0.01, 0.005, 0.0025], value=0.01)
    alpha = c2.select_slider("Allowed failure α", [0.01, 0.05, 0.1], value=0.05)
    shots = c3.select_slider("Shots per round", [50, 100, 200, 400], value=100)
    seed_q = c4.number_input("Seed", 0, 9999, 7, key="qseed")
    with st.spinner("Running IQAE on the Qiskit simulator…"):
        R = c_iqae(NET, src, tgt, eps, alpha, shots, seed_q)
    mc_same = c_mc(NET, src, tgt, R["calls"], seed_q)
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        k1, k2 = st.columns(2)
        k1.metric("IQAE estimate", f"{R['est']:.6f}", f"error {abs(R['est'] - EXACT):.6f}", delta_color="off", delta_arrow="off")
        k2.metric("Monte Carlo, same budget", f"{mc_same:.6f}", f"error {abs(mc_same - EXACT):.6f}", delta_color="off", delta_arrow="off")
        st.markdown(f"**{R['calls']:,} circuit runs** (state-preparation calls) · {100 * (1 - alpha):.0f}% confidence interval "
                    f"[{R['lo']:.5f}, {R['hi']:.5f}] · rounds used k = {', '.join(map(str, R['powers']))}")
        st.caption("Calls counted strictly: each shot of QᵏA runs circuit A (2k+1) times. Qiskit's own "
                   f"num_oracle_queries reports {R['queries']:,}, which undercounts. For speed, this web demo compiles "
                   "circuit A into one equivalent unitary before IQAE; it is the same operator, so the estimates match the notebook exactly.")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[R["lo"], R["hi"]], y=["IQAE"] * 2, mode="lines", line=dict(color=C_Q, width=10),
                                 name="IQAE confidence interval"))
        fig.add_trace(go.Scatter(x=[R["est"]], y=["IQAE"], mode="markers", marker=dict(size=16, color=C_Q, line=dict(color="white", width=2)), name="IQAE estimate"))
        half = 1.96 * math.sqrt(max(mc_same * (1 - mc_same), 1e-9) / R["calls"])
        fig.add_trace(go.Scatter(x=[mc_same - half, mc_same + half], y=["Monte Carlo"] * 2, mode="lines",
                                 line=dict(color=C_MC, width=10), name="MC 95% interval"))
        fig.add_trace(go.Scatter(x=[mc_same], y=["Monte Carlo"], mode="markers", marker=dict(size=16, color=C_MC, line=dict(color="white", width=2)), name="MC estimate"))
        fig.add_vline(x=EXACT, line=dict(color=C_EXACT, dash="dash"), annotation_text="exact")
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                          xaxis_title="P(cut off)", title=dict(text="Same number of circuit runs", font=dict(size=13)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    say("We write the probability as sin² of an angle. The Grover operator multiplies that angle, so measuring after several "
        "applications pins the angle down more precisely per run than random sampling can.")

# ---------------------------------------------------------------------------
# 5. Head-to-head
# ---------------------------------------------------------------------------
with tabs[4]:
    presenter("h2h")
    st.subheader("Error versus number of runs")
    st.markdown("Monte Carlo error falls like $1/\\sqrt{N}$ (slope −½ on a log-log plot). Ideal amplitude estimation falls "
                "like $1/N$ (slope −1). The gap widens the more accuracy you want.")
    if st.button("Run the sweep live on this network", type="primary"):
        st.session_state["sweep_for"] = NET + (src, tgt)
    if st.session_state.get("sweep_for") == NET + (src, tgt):
        prog = st.progress(0.0, "Monte Carlo sweep…")
        mc_N = [1_000, 4_000, 16_000, 64_000, 256_000]
        mc_err = [np.mean([abs(c_mc(NET, src, tgt, n, s) - EXACT) for s in range(10)]) for n in mc_N]
        eps_grid = [0.02, 0.01, 0.005, 0.0025, 0.00125]
        iq_calls, iq_err, iq_hw = [], [], []
        for i, e in enumerate(eps_grid):
            prog.progress((i + 1) / (len(eps_grid) + 1), f"IQAE at ε = {e}…")
            rs = [c_iqae(NET, src, tgt, e, 0.05, 100, 7 + s) for s in range(2)]
            iq_calls.append(np.median([r["calls"] for r in rs]))
            iq_err.append(max(np.median([abs(r["est"] - EXACT) for r in rs]), 1e-6))
            iq_hw.append(np.median([(r["hi"] - r["lo"]) / 2 for r in rs]))
        prog.empty()
        mc_slope = np.polyfit(np.log10(mc_N), np.log10(mc_err), 1)[0]
        iq_slope = np.polyfit(np.log10(iq_calls), np.log10(iq_hw), 1)[0]
        xx = np.logspace(2.5, 6, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mc_N, y=mc_err, mode="lines+markers", name=f"Monte Carlo error (slope {mc_slope:+.2f})",
                                 line=dict(color=C_MC, width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=iq_calls, y=iq_hw, mode="lines+markers", name=f"IQAE guaranteed error, 95% CI half-width (slope {iq_slope:+.2f})",
                                 line=dict(color=C_Q, width=3), marker=dict(size=10, symbol="square")))
        fig.add_trace(go.Scatter(x=iq_calls, y=iq_err, mode="markers", name="IQAE actual error",
                                 marker=dict(size=9, color=C_Q, symbol="square-open")))
        fig.add_trace(go.Scatter(x=xx, y=mc_err[0] * np.sqrt(mc_N[0] / xx), name="1/√N reference",
                                 line=dict(color=C_EXACT, dash="dot", width=1.5)))
        fig.add_trace(go.Scatter(x=xx, y=mc_err[0] * (mc_N[0] / xx), name="1/N reference",
                                 line=dict(color=C_EXACT, dash="dash", width=1.5)))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log", yaxis_type="log",
                          xaxis_title="circuit runs / samples", yaxis_title="absolute error",
                          legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig, width="stretch")
        callout(f"Measured on this network: Monte Carlo slope <b>{mc_slope:+.2f}</b> (theory −0.5), IQAE confidence-interval "
                f"slope <b>{iq_slope:+.2f}</b> (theory −1). Slopes come from a handful of runs, so they show the trend rather than prove it.")
    else:
        st.caption("The live sweep takes about 20–40 seconds. Below is the notebook's figure for the default network.")
        st.image(str(FIG_DIR / "error-vs-runs.png"), caption="Notebook Figure 1 (default two-route network)", width="stretch")
    say("For the same target accuracy, amplitude estimation needs quadratically fewer circuit runs than random sampling. "
        "On our network IQAE's guaranteed error falls with slope about −1, Monte Carlo's about −0.5.")

# ---------------------------------------------------------------------------
# 6. Noise
# ---------------------------------------------------------------------------
with tabs[5]:
    presenter("noise")
    st.subheader("What happens on noisy hardware?")
    a, b = st.columns([1, 1.4], gap="large")
    with a:
        st.markdown("Real quantum computers make errors. We simulate **depolarizing noise**: after every gate, with chance ρ, "
                    "the qubit is scrambled. Two-qubit gates get 2ρ, since they are worse on real hardware.")
        st.markdown("Big **k** gives the speed-up, but each Grover application makes the circuit much deeper, so it collects more errors.")
        st.caption("The circuit is first compiled into basic gates (rz, sx, x, cx), so every gate, including the big MCX gates, picks up noise.")
        ks_sel = st.multiselect("Grover applications k", [0, 1, 2], default=[0, 1, 2])
        go_noise = st.button("Run noisy simulations", type="primary")
    with b:
        if len(BAD) > 20:
            st.info("The noise study runs on networks with up to 20 oracle terms (the circuits get very deep). "
                    "Try the two-route demo or bridge network.")
        elif go_noise or st.session_state.get("noise_for") == NET + (src, tgt, tuple(ks_sel)):
            st.session_state["noise_for"] = NET + (src, tgt, tuple(ks_sel))
            rhos = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
            colors = {0: "#491d8b", 1: "#009d9a", 2: "#6fdc8c"}
            fig = go.Figure()
            rows = []
            with st.spinner("Simulating noisy circuits with Qiskit Aer…"):
                for k in ks_sel:
                    res = [c_noise(NET, src, tgt, k, r) for r in rhos]
                    depth, cx = res[0][1], res[0][2]
                    rows.append({"k": k, "depth": depth, "CX gates": cx})
                    fig.add_trace(go.Scatter(x=[r if r else 3e-5 for r in rhos], y=[x[0] for x in res], mode="lines+markers",
                                             name=f"k = {k} (depth {depth:,})", line=dict(color=colors[k], width=3), marker=dict(size=9)))
                    fig.add_hline(y=q.recover_probability(0.5, k), line=dict(color=colors[k], dash="dot", width=1))
            fig.add_hline(y=EXACT, line=dict(color=C_EXACT, dash="dash", width=2), annotation_text=f"exact {EXACT:.4f}")
            fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), xaxis_type="log",
                              xaxis=dict(tickvals=[3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2], ticktext=["0", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]),
                              xaxis_title="noise rate ρ per gate", yaxis_title="recovered P(cut off)", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Dotted lines: the value the recovery formula gives when noise has completely scrambled the circuit. "
                       "A curve sitting on its dotted line carries no information.")
            close = [k for k in ks_sel if abs(q.recover_probability(0.5, k) - EXACT) < 0.01]
            if close:
                callout(f"Careful: for k = {', '.join(map(str, close))}, total scrambling happens to give a value close to the exact "
                        "answer. A flat line there is a coincidence, not resilience.", "red")
        else:
            st.image(str(FIG_DIR / "noise-vs-depth.png"), caption="Notebook Figure 2 (default network). Press the button to rerun live.",
                     width="stretch")
    say("The advantage needs deep circuits, and deep circuits drown in noise on today's hardware. Even a 0.1% error rate "
        "ruins the k = 2 estimate on our smallest network. That is the honest barrier to a practical speed-up today.")

# ---------------------------------------------------------------------------
# 7. Scaling & v1
# ---------------------------------------------------------------------------
with tabs[6]:
    presenter("scaling")
    st.subheader("The catch with the v0 oracle, and our v1 fix")
    a, b = st.columns(2, gap="large")
    with a:
        callout("<b>v0 · minterm oracle.</b> Lists every cut-off scenario on a classical computer first, then builds one MCX gate per "
                "scenario. Simple and easy to verify, but listing scenarios is the same 2<sup>m</sup> work as exact enumeration, and the "
                "gate count explodes.", "red")
    with b:
        callout("<b>v1 · reachability oracle.</b> Runs breadth-first search <i>inside</i> the circuit: a node is reached if it was reached, "
                "OR a neighbour was reached AND the link works. No scenario list at all. It trades gates for qubits, because "
                "intermediate answers are kept on extra qubits.", "violet")
    st.latex(r"\mathrm{reach}_r(v)=\mathrm{reach}_{r-1}(v)\;\lor\;\bigvee_{u\sim v}\big(\mathrm{reach}_{r-1}(u)\land \mathrm{works}_{uv}\big)")
    st.caption("OR is built from AND gates with De Morgan's law: A OR B = NOT(NOT A AND NOT B). Constant folding, reuse of identical "
               "AND/OR results, and backward slicing keep the circuit small.")
    v1s, v1amp = c_v1(NET, src, tgt)
    _, v0tof, v0q = q.v0_cost(G, src, tgt)
    st.markdown("**On your current network**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("v0 qubits / Toffoli cost", f"{v0q} / {v0tof:,}")
    k2.metric("v1 qubits / Toffoli cost", f"{v1s['qubits']} / {v1s['toffoli_equiv']:,}")
    k3.metric("v0 exact P(objective=1)", f"{c_v0_amp(NET, src, tgt):.6f}")
    k4.metric("v1 exact P(objective=1)", f"{v1amp:.6f}" if v1amp is not None else "too many qubits to simulate")
    st.divider()
    st.subheader("Resource scaling on street-grid networks")
    rows = c_grid_scaling()
    dfg = pd.DataFrame(rows)
    a, b = st.columns([1.2, 1], gap="large")
    with a:
        fig = go.Figure()
        v0r = [r for r in rows if r["v0 Toffoli cost"]]
        fig.add_trace(go.Scatter(x=[r["links"] for r in v0r], y=[r["v0 Toffoli cost"] for r in v0r], mode="lines+markers",
                                 name="v0 gate cost (Toffoli-equivalents)", line=dict(color=C_MC, width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[r["links"] for r in rows], y=[r["v1 Toffoli cost"] for r in rows], mode="lines+markers",
                                 name="v1 gate cost", line=dict(color=C_V1, width=3), marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[r["links"] for r in rows], y=[r["v1 qubits"] for r in rows], mode="lines+markers",
                                 name="v1 qubits", line=dict(color=C_V1, width=2, dash="dot"), marker=dict(size=8, symbol="diamond")))
        fig.add_trace(go.Scatter(x=[r["links"] for r in rows], y=[r["v0 qubits"] for r in rows], mode="lines+markers",
                                 name="v0 qubits", line=dict(color=C_MC, width=2, dash="dot"), marker=dict(size=8, symbol="diamond")))
        for r in rows:
            fig.add_annotation(x=r["links"], y=math.log10(r["v1 Toffoli cost"]), text=r["grid"], showarrow=False, yshift=-16, font=dict(size=11))
        fig.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), yaxis_type="log",
                          xaxis_title="links in the grid network", yaxis_title="count (log scale)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with b:
        shown_g = dfg.copy()
        for col in ["v0 scenarios marked", "v0 Toffoli cost"]:
            shown_g[col] = shown_g[col].map(lambda v: "skipped" if v is None or pd.isna(v) else f"{int(v):,}")
        shown_g["v1 Toffoli cost"] = shown_g["v1 Toffoli cost"].map(lambda v: f"{int(v):,}")
        st.dataframe(shown_g, hide_index=True, width="stretch")
        st.caption("3×4 grid: v0 is skipped because listing 2¹⁷ = 131,072 scenarios is too slow. Large v1 circuits are counted "
                   "at compile time, not simulated. Cost model: a k-control MCX ≈ 2k − 3 Toffoli gates.")
    st.divider()
    st.subheader("Cross-validation: are the oracles correct?")
    with st.spinner("Checking both oracles against exact enumeration…"):
        cv = pd.DataFrame(c_crossval())
    worst = np.nanmax(cv[["v0 error", "v1 error"]].astype(float).values)
    shown = cv.copy()
    shown["v0 error"] = shown["v0 error"].map(lambda e: f"{e:.1e}")
    shown["v1 error"] = shown["v1 error"].map(lambda e: "too big to simulate" if e is None or pd.isna(e) else f"{e:.1e}")
    st.dataframe(shown, hide_index=True, width="stretch",
                 column_config={"exact P_disc": st.column_config.NumberColumn(format="%.6f")})
    callout(f"Worst disagreement with exact enumeration across {len(cv)} networks: <b>{worst:.1e}</b>, which is computer rounding. "
            "Both oracles compute exactly the right probability. (v1 is checked only where it fits in a 20-qubit statevector.)")
    with st.expander("Notebook Figures 3 and 4"):
        c1, c2 = st.columns(2)
        c1.image(str(FIG_DIR / "resource-scaling.png"), width="stretch")
        c2.image(str(FIG_DIR / "v0-vs-v1.png"), width="stretch")
    say("v0 is our simple, verifiable oracle; v1 turns breadth-first search into a reversible circuit, so it never enumerates "
        "scenarios. It pays with qubits: 254 for a 17-link grid. Both match exact enumeration to rounding error.")

# ---------------------------------------------------------------------------
# 8. Limits & Q&A
# ---------------------------------------------------------------------------
with tabs[7]:
    presenter("limits")
    a, b = st.columns([1, 1.2], gap="large")
    with a:
        st.subheader("Limits we state up front")
        st.markdown(
            "- **Independent failures.** Real failures can be linked (one storm takes out several lines). The model doesn't capture that.\n"
            "- **Small networks only.** Exact checks need a full statevector, so verification stops near 20 qubits.\n"
            "- **Fewer runs is not the same as faster.** The advantage counts circuit runs, not seconds. Compilation, depth and noise all add cost.\n"
            "- **Building the oracle costs something.** v0 needs a classical scenario list; v1 needs many extra qubits.\n"
            "- **Slopes are empirical.** Figure 1 shows a trend from a few runs, not a proof.\n"
            "- **Not a new algorithm.** Amplitude estimation for network reliability has been studied before "
            "([arXiv:2203.10201](https://arxiv.org/abs/2203.10201)). Our contribution is a working, cross-validated tool."
        )
        st.subheader("What we built")
        st.markdown(
            "- A tool that takes a network as a CSV (links + failure chances) and returns the disconnection probability three ways.\n"
            "- Two quantum oracles (scenario list and reachability), cross-checked on 10 networks.\n"
            "- A noise study and resource estimates that show where the quantum advantage breaks down today."
        )
    with b:
        st.subheader("Judge questions")
        faq = [
            ("Does this beat a classical computer today?",
             "No. On today's noisy hardware the deep circuits fail (tab ⑥), and the oracle costs gates or qubits as networks grow (tab ⑦). "
             "The advantage is in the number of circuit runs: about 1/N error instead of 1/√N."),
            ("The v0 oracle lists every bad scenario. Doesn't that already give you the answer?",
             "Yes, which is why v0 is only for validation. v1 never lists scenarios: it computes reachability inside the circuit, "
             "at the price of extra qubits."),
            ("How do you know the quantum circuit is correct?",
             "We compute the exact probability of the objective qubit from the full statevector and compare it with exact enumeration "
             "on 10 networks. The worst error is about 10⁻¹⁴, which is rounding."),
            ("What does the Grover operator actually do?",
             "It rotates the state by 2θ, where P_disc = sin²θ. Applying it k times turns the probability into sin²((2k+1)θ), "
             "which magnifies the angle so it can be measured more precisely."),
            ("Why IQAE rather than the original amplitude estimation?",
             "The original version needs a quantum Fourier transform and many extra qubits. IQAE uses only the Grover operator and "
             "classical post-processing, so the circuits are smaller."),
            ("Why Ry rotations?",
             "Ry(θ) gives amplitude sin(θ/2) on |1⟩. Choosing θ = 2·arcsin√p makes the probability of 'failed' exactly p."),
            ("How are runs counted?",
             "Strictly: each shot of QᵏA runs circuit A (2k+1) times. Qiskit's num_oracle_queries undercounts that, so we use the stricter number."),
            ("Who would use this?",
             "Reliability engineers for power, telecom or water networks. They provide a CSV of links and failure chances and never touch a qubit."),
        ]
        for qn, ans in faq:
            with st.expander(qn):
                st.markdown(ans)
    say("We state our limits clearly: independent failures, small verified networks, and noise that erases the advantage "
        "today. What we contribute is a working, cross-validated pipeline from a network CSV to a quantum estimate.")

st.divider()
st.caption("Q-Resilience · built with Qiskit, qiskit-algorithms (IQAE), Qiskit Aer, NetworkX and Streamlit · "
           "all quantum results come from simulators, not real hardware.")
