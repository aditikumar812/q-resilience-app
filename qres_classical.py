"""
Q-Resilience classical core: everything that does NOT touch Qiskit.

The web app imports only this module directly. All Qiskit work runs in a separate
helper process (worker.py), because Qiskit's compiled core can crash on macOS when it
is called from the short-lived threads Streamlit creates for every click.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Sequence

import networkx as nx
import numpy as np

Edge = tuple

# ---------------------------------------------------------------------------
# Preset networks (failure probabilities are illustrative, not measured data)
# ---------------------------------------------------------------------------
PRESETS = {
    "Two-route demo (notebook default)": {
        "edges": {("s", "a"): 0.10, ("a", "t"): 0.15, ("s", "b"): 0.20, ("b", "t"): 0.10},
        "source": "s", "target": "t",
        "blurb": "Two independent routes from s to t. Small enough to check by hand.",
    },
    "Bridge network": {
        "edges": {("s", "a"): 0.10, ("s", "b"): 0.15, ("a", "b"): 0.05,
                  ("a", "t"): 0.12, ("b", "t"): 0.08},
        "source": "s", "target": "t",
        "blurb": "Adds a cross-link a-b, so the routes share wires and the by-hand shortcut no longer works.",
    },
    "Mini power grid": {
        "edges": {("Plant", "Sub A"): 0.05, ("Plant", "Sub B"): 0.08, ("Sub A", "Sub B"): 0.10,
                  ("Sub A", "Feeder 1"): 0.12, ("Sub B", "Feeder 2"): 0.12,
                  ("Feeder 1", "Hospital"): 0.06, ("Feeder 2", "Hospital"): 0.06},
        "source": "Plant", "target": "Hospital",
        "blurb": "A power plant feeding a hospital through two substations and two feeders, with a tie line. Probabilities are illustrative.",
    },
    "2 x 3 grid": {
        "edges": {(u, v): 0.10 for u, v in
                  nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 3)).edges()},
        "source": 0, "target": 5,
        "blurb": "A street-grid network, corner to corner, every link failing 10% of the time.",
    },
}


def build_graph(edge_probs: dict):
    graph = nx.Graph()
    graph.add_edges_from(edge_probs.keys())
    return graph


def load_network_csv(text: str, source, target):
    """Notebook Cell 2: parse CSV text with columns u, v, p_fail and validate it."""
    rows = list(csv.DictReader(io.StringIO(text.strip())))
    if not rows or not {"u", "v", "p_fail"} <= set(k.strip() for k in rows[0].keys()):
        raise ValueError("The CSV needs the header: u,v,p_fail")
    probs, seen = {}, set()
    for line_no, r in enumerate(rows, start=2):
        r = {k.strip(): (v or "").strip() for k, v in r.items()}
        u, v = r["u"], r["v"]
        try:
            p = float(r["p_fail"])
        except ValueError:
            raise ValueError(f"Line {line_no}: p_fail must be a number, got {r['p_fail']!r}")
        if u == v:
            raise ValueError(f"Line {line_no}: a link can't connect {u} to itself")
        if frozenset((u, v)) in seen:
            raise ValueError(f"Line {line_no}: duplicate link {u}-{v}")
        if not 0.0 < p < 1.0:
            raise ValueError(f"Line {line_no}: p_fail must be between 0 and 1, got {p}")
        seen.add(frozenset((u, v)))
        probs[(u, v)] = p
    g = build_graph(probs)
    if source == target or source not in g or target not in g:
        raise ValueError("Source and target must be two different nodes that appear in the CSV")
    if not nx.has_path(g, source, target):
        raise ValueError("Source and target aren't connected even when every link works")
    return probs


# ---------------------------------------------------------------------------
# Cell 3: classical baselines
# ---------------------------------------------------------------------------
def canonical_edge(u, v) -> Edge:
    return (u, v) if str(u) <= str(v) else (v, u)


def ordered_edges(graph) -> list:
    edges = (canonical_edge(u, v) for u, v in graph.edges())
    return sorted(edges, key=lambda e: (str(e[0]), str(e[1])))


def failure_probability(failure_probs, edge) -> float:
    u, v = edge
    if (u, v) in failure_probs:
        p = float(failure_probs[(u, v)])
    elif (v, u) in failure_probs:
        p = float(failure_probs[(v, u)])
    else:
        raise KeyError(f"Missing failure probability for edge {edge!r}.")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p for {edge!r} must be in [0, 1], got {p}.")
    return p


def validate_endpoints(graph, source, target):
    if source not in graph:
        raise ValueError(f"Source {source!r} not in graph.")
    if target not in graph:
        raise ValueError(f"Target {target!r} not in graph.")


def is_disconnected(graph, edge_list, failed_mask, source, target) -> bool:
    """Depth-first search over the links that still work."""
    if source == target:
        return False
    adjacency = {node: [] for node in graph.nodes()}
    for i, (u, v) in enumerate(edge_list):
        if not ((failed_mask >> i) & 1):
            adjacency[u].append(v)
            adjacency[v].append(u)
    seen, stack = {source}, [source]
    while stack:
        node = stack.pop()
        if node == target:
            return False
        for nbr in adjacency[node]:
            if nbr not in seen:
                seen.add(nbr)
                stack.append(nbr)
    return True


def disconnected_masks(graph, edge_list, source, target) -> list:
    validate_endpoints(graph, source, target)
    if source == target:
        return []
    return [m for m in range(1 << len(edge_list))
            if is_disconnected(graph, edge_list, m, source, target)]


def mask_probability(mask: int, probs: Sequence[float]) -> float:
    p = 1.0
    for i, pf in enumerate(probs):
        p *= pf if ((mask >> i) & 1) else (1.0 - pf)
    return p


def exact_enumeration(graph, failure_probs, source, target) -> float:
    edges = ordered_edges(graph)
    probs = [failure_probability(failure_probs, e) for e in edges]
    masks = disconnected_masks(graph, edges, source, target)
    return float(sum(mask_probability(m, probs) for m in masks))


def _disconnect_table(graph, edges, source, target):
    """Lookup table: table[mask] = True if that failure pattern disconnects s from t."""
    table = np.zeros(1 << len(edges), dtype=bool)
    for m in disconnected_masks(graph, edges, source, target):
        table[m] = True
    return table


def monte_carlo_simulation(graph, failure_probs, source, target, num_samples=20_000, seed=7):
    """
    Same estimator and same random stream as the notebook's Cell 3.
    Draws are generated in one numpy call (identical numbers, row by row),
    and connectivity is looked up from a precomputed table for speed.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    edges = ordered_edges(graph)
    probs = np.array([failure_probability(failure_probs, e) for e in edges])
    rng = np.random.default_rng(seed)
    bits = rng.random((num_samples, len(edges))) < probs
    if len(edges) <= 18:
        weights = (1 << np.arange(len(edges))).astype(np.int64)
        masks = bits.astype(np.int64) @ weights
        hits = _disconnect_table(graph, edges, source, target)[masks]
        return float(hits.mean())
    count = 0
    for row in bits:
        mask = sum(1 << i for i, b in enumerate(row) if b)
        count += is_disconnected(graph, edges, mask, source, target)
    return count / num_samples


def monte_carlo_running(graph, failure_probs, source, target, num_samples, seed=7):
    """Running Monte Carlo estimate after each sample (for the convergence chart)."""
    edges = ordered_edges(graph)
    probs = np.array([failure_probability(failure_probs, e) for e in edges])
    rng = np.random.default_rng(seed)
    bits = rng.random((num_samples, len(edges))) < probs
    weights = (1 << np.arange(len(edges))).astype(np.int64)
    masks = bits.astype(np.int64) @ weights
    hits = _disconnect_table(graph, edges, source, target)[masks]
    return np.cumsum(hits) / np.arange(1, num_samples + 1)


# ---------------------------------------------------------------------------
# Cell 7b: v1 reachability oracle compiler (pure Python until to_circuit)
# ---------------------------------------------------------------------------
ZERO, ONE = -1, -2  # compile-time constants for constant folding


class RevBuilder:
    """Records reversible ops; emits a Qiskit circuit, or only counts resources."""

    def __init__(self):
        self.n, self.ops, self._and, self._or = 0, [], {}, {}

    def alloc(self):
        q = self.n
        self.n += 1
        return q

    def ry(self, q, theta):
        self.ops.append(("ry", q, theta))

    def x(self, q):
        self.ops.append(("x", q))

    def cx(self, c, t):
        self.ops.append(("cx", c, t))

    def and_(self, a, b):
        if ZERO in (a, b):
            return ZERO
        if a == ONE:
            return b
        if b == ONE:
            return a
        key = (min(a, b), max(a, b))
        if key not in self._and:
            out = self.alloc()
            self.ops.append(("mcx", (a, b), out))
            self._and[key] = out
        return self._and[key]

    def or_(self, terms):
        terms = [t for t in dict.fromkeys(terms) if t != ZERO]
        if ONE in terms:
            return ONE
        if not terms:
            return ZERO
        if len(terms) == 1:
            return terms[0]
        key = frozenset(terms)
        if key not in self._or:
            out = self.alloc()
            for t in terms:
                self.x(t)
            self.ops.append(("mcx", tuple(terms), out))  # AND of negations ...
            self.x(out)                                   # ... negated = OR
            for t in terms:
                self.x(t)
            self._or[key] = out
        return self._or[key]

    def to_circuit(self):
        from qiskit import QuantumCircuit  # imported lazily: only the worker process calls this
        qc = QuantumCircuit(self.n)
        for op in self.ops:
            if op[0] == "ry":
                qc.ry(op[2], op[1])
            elif op[0] == "x":
                qc.x(op[1])
            elif op[0] == "cx":
                qc.cx(op[1], op[2])
            elif op[0] == "mcx":
                qc.mcx(list(op[1]), op[2])
        return qc

    def stats(self):
        tof = sum(2 * len(o[1]) - 3 for o in self.ops if o[0] == "mcx")
        kmax = max([len(o[1]) for o in self.ops if o[0] == "mcx"] + [0])
        return {"qubits": self.n, "toffoli_equiv": tof, "work_qubits_extra": max(0, kmax - 2)}


def build_reachability_state_prep(graph, failure_probs, source, target, max_hops=None):
    """v1 compiler. Objective qubit = 1 exactly when source and target are disconnected."""
    validate_endpoints(graph, source, target)
    edges = ordered_edges(graph)
    b = RevBuilder()
    w = [b.alloc() for _ in edges]
    for q, e in zip(w, edges):
        p = failure_probability(failure_probs, e)
        b.ry(q, 2.0 * math.asin(math.sqrt(1.0 - p)))  # |1> = link works
    nbrs = {v: [] for v in graph.nodes()}
    for i, (u, v) in enumerate(edges):
        nbrs[u].append((v, i))
        nbrs[v].append((u, i))
    R = max_hops or (graph.number_of_nodes() - 1)
    needed = {R: {target}}  # backward slicing
    for r in range(R - 1, 0, -1):
        needed[r] = needed[r + 1] | {u for v in needed[r + 1] for u, _ in nbrs[v]}
    reach = {v: (ONE if v == source else ZERO) for v in graph.nodes()}
    for r in range(1, R + 1):
        new = {}
        for v in graph.nodes():
            if v == source:
                new[v] = ONE
            elif v in needed[r]:
                new[v] = b.or_([reach[v]] + [b.and_(reach[u], w[i]) for u, i in nbrs[v]])
        reach = new
    t = reach[target]
    if t in (ONE, ZERO):
        raise ValueError("Target is trivially connected or disconnected")
    obj = b.alloc()
    b.cx(t, obj)
    b.x(obj)
    return b, obj


def v0_cost(graph, source, target):
    """Toffoli-equivalent cost of the v0 oracle (notebook Cell 11b formula)."""
    edges = ordered_edges(graph)
    m = len(edges)
    n_masks = len(disconnected_masks(graph, edges, source, target))
    return n_masks, n_masks * (2 * m - 3), m + 1


# Cell 10 helper
def recover_probability(p_meas: float, k: int) -> float:
    amp = np.sqrt(np.clip(p_meas, 1e-12, 1.0))
    theta = np.arcsin(amp) / (2 * k + 1)
    return float(np.sin(theta) ** 2)


