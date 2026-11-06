"""
Runs every Qiskit computation for the web app in a separate helper process.

Why: Qiskit's compiled core (its memory allocator) can crash on macOS when called from
the new threads Streamlit starts on each click. In this helper process all Qiskit work
runs on one long-lived main thread, and if it ever crashes, the website survives and
simply starts a fresh helper.
"""
from __future__ import annotations

import io
import time
from functools import lru_cache

import numpy as np


def _probs(net):
    return {(u, v): p for u, v, p in net}


@lru_cache(maxsize=64)
def _problem(net, s, t):
    import qres_core as q
    p = _probs(net)
    return q.build_estimation_problem(q.build_graph(p), p, s, t, fast=True)


def iqae(net, s, t, eps, alpha, shots, seed):
    import qres_core as q
    comp = _problem(net, s, t)
    t0 = time.time()
    r = q.run_iqae(comp.problem, eps, alpha, shots, seed)
    return {
        "est": float(np.clip(r.estimation, 0, 1)),
        "lo": float(r.confidence_interval[0]),
        "hi": float(r.confidence_interval[1]),
        "queries": int(r.num_oracle_queries),
        "calls": q.state_prep_calls(r, shots),
        "powers": [int(k) for k in r.powers[1:]],
        "secs": time.time() - t0,
    }


def profile(net, s, t):
    import qres_core as q
    p = _probs(net)
    return q.profile_resources(q.build_graph(p), p, s, t)


def v1(net, s, t):
    import qres_core as q
    p = _probs(net)
    b, o = q.build_reachability_state_prep(q.build_graph(p), p, s, t)
    amp = q.exact_amplitude(b.to_circuit(), o) if b.n <= 20 else None
    return b.stats(), amp


def v0_amp(net, s, t):
    import qres_core as q
    comp = _problem(net, s, t)
    return q.exact_amplitude(comp.state_preparation, comp.objective_qubit)


def amplification(net, s, t, kmax):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    comp = _problem(net, s, t)
    sp, obj = comp.state_preparation, comp.objective_qubit
    out = []
    qc = QuantumCircuit(sp.num_qubits)
    qc.compose(sp, inplace=True)
    for k in range(kmax + 1):
        if k > 0:
            qc.compose(comp.grover_operator, inplace=True)
        out.append(float(Statevector(qc).probabilities([obj])[1]))
    return out


def oracle_on(net, s, t, mask):
    """Feed one exact failure pattern into the v0 oracle and read the objective qubit."""
    import qres_core as q
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    p = _probs(net)
    g = q.build_graph(p)
    edges = q.ordered_edges(g)
    n = len(edges)
    masks = q.disconnected_masks(g, edges, s, t)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if (mask >> i) & 1:
            qc.x(i)
    qc.compose(q.build_disconnection_oracle(n, masks, n), inplace=True)
    return float(Statevector(qc).probabilities([n])[1])


def noise(net, s, t, k, rho):
    import qres_core as q
    p = _probs(net)
    return q.noisy_estimate(q.build_graph(p), p, s, t, k, rho)


def circuit_png(net, s, t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import qres_core as q
    p = _probs(net)
    sp, *_ = q.build_state_preparation(q.build_graph(p), p, s, t)
    fig = sp.draw("mpl", fold=-1, scale=0.62)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def crossval():
    import networkx as nx
    import qres_core as q
    cases = [(name, P["edges"], P["source"], P["target"]) for name, P in q.PRESETS.items()]
    rng = np.random.default_rng(2026)
    tries = 0
    while len(cases) < 10 and tries < 5000:
        tries += 1
        n = int(rng.integers(4, 6))
        m = int(rng.integers(n, min(6, n * (n - 1) // 2) + 1))
        g = nx.gnm_random_graph(n, m, seed=int(rng.integers(1_000_000)))
        if not (nx.is_connected(g) and nx.has_path(g, 0, n - 1)):
            continue
        p = {(u, v): float(rng.uniform(0.02, 0.3)) for u, v in g.edges()}
        if q.build_reachability_state_prep(g, p, 0, n - 1)[0].n > 20:
            continue
        cases.append((f"Random graph: {n} nodes, {m} links", p, 0, n - 1))
    out = []
    for name, p, s, t in cases:
        g = q.build_graph(p)
        ex = q.exact_enumeration(g, p, s, t)
        sp, _e, o0, _ = q.build_state_preparation(g, p, s, t)
        b1, o1 = q.build_reachability_state_prep(g, p, s, t)
        e0 = abs(q.exact_amplitude(sp, o0) - ex)
        e1 = abs(q.exact_amplitude(b1.to_circuit(), o1) - ex) if b1.n <= 20 else None
        out.append({"network": name, "links": len(p), "exact P_disc": ex, "v0 error": e0,
                    "v1 error": e1, "v1 qubits": b1.n})
    return out


def ping():
    import qres_core  # noqa: F401  (warm up: import Qiskit once)
    return True


# ---------------------------------------------------------------------------
# Helper-process loop: the web app starts `python worker.py` and sends
# (function name, arguments) over stdin; results come back over stdout.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pickle
    import sys

    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must not corrupt the result stream
    while True:
        try:
            fn, args = pickle.load(inp)
        except EOFError:
            break
        try:
            result = (True, globals()[fn](*args))
        except Exception as exc:  # report errors back instead of dying
            result = (False, f"{type(exc).__name__}: {exc}")
        pickle.dump(result, out)
        out.flush()
