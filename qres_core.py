"""
Q-Resilience quantum core: the notebook's Qiskit code (Cells 4-7b, 10).
Classical helpers live in qres_classical.py and are re-exported here.
Only the worker process imports this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

try:
    from qiskit.primitives import StatevectorSampler
except ImportError:  # pragma: no cover
    from qiskit_aer.primitives import Sampler as StatevectorSampler

try:
    from qiskit_algorithms import EstimationProblem, IterativeAmplitudeEstimation
except ImportError:  # pragma: no cover
    from qiskit.algorithms import EstimationProblem, IterativeAmplitudeEstimation

from qres_classical import *  # noqa: F401,F403
from qres_classical import (PRESETS, build_graph, ordered_edges, failure_probability, validate_endpoints,
                            is_disconnected, disconnected_masks, mask_probability, exact_enumeration,
                            RevBuilder, ZERO, ONE, build_reachability_state_prep, v0_cost, recover_probability)

# ---------------------------------------------------------------------------
# Cell 4: state preparation + v0 cut-set (minterm) oracle
# ---------------------------------------------------------------------------
def build_disconnection_oracle(num_edges, masks, objective_qubit) -> QuantumCircuit:
    oracle = QuantumCircuit(num_edges + 1, name="Oracle")
    for mask in masks:
        zeros = [q for q in range(num_edges) if not ((mask >> q) & 1)]
        for q in zeros:
            oracle.x(q)
        oracle.mcx(list(range(num_edges)), objective_qubit)
        for q in zeros:
            oracle.x(q)
    return oracle


def select_oracle_masks(graph, edge_list, failure_probs, source, target,
                        prob_floor=0.0, max_minterms=None) -> list:
    masks = disconnected_masks(graph, edge_list, source, target)
    if prob_floor > 0.0:
        probs = [failure_probability(failure_probs, e) for e in edge_list]
        scored = [(m, mask_probability(m, probs)) for m in masks]
        scored = [(m, p) for m, p in scored if p >= prob_floor]
        scored.sort(key=lambda x: -x[1])
        masks = [m for m, _ in scored]
    if max_minterms is not None:
        masks = masks[: int(max_minterms)]
    return masks


def build_state_preparation(graph, failure_probs, source, target,
                            prob_floor=0.0, max_minterms=None, include_oracle=True):
    """Circuit A: Ry encoding of every link, then the v0 oracle on the objective qubit."""
    validate_endpoints(graph, source, target)
    edges = ordered_edges(graph)
    n = len(edges)
    obj_q = n
    qc = QuantumCircuit(n + 1, name="A")
    for q, edge in enumerate(edges):
        p = failure_probability(failure_probs, edge)
        if p > 0.0:
            qc.ry(2.0 * math.asin(math.sqrt(p)), q)
    masks = select_oracle_masks(graph, edges, failure_probs, source, target, prob_floor, max_minterms)
    if include_oracle and masks:
        qc.barrier()
        qc.compose(build_disconnection_oracle(n, masks, obj_q), inplace=True)
    return qc, edges, obj_q, len(masks)


# ---------------------------------------------------------------------------
# Cell 5: estimation problem
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CompiledResilienceProblem:
    problem: EstimationProblem
    state_preparation: QuantumCircuit
    grover_operator: QuantumCircuit
    edge_list: list
    objective_qubit: int
    oracle_terms: int


def build_estimation_problem(graph, failure_probs, source, target, prob_floor=0.0, max_minterms=None, fast=False):
    """
    fast=True (web app only): circuit A is compiled into one equivalent unitary before IQAE.
    It is the same operator, so the estimates are identical, but the simulator no longer
    has to step through dozens of multi-controlled gates. The noise study does not use it.
    """
    state_prep, edges, obj_q, terms = build_state_preparation(
        graph, failure_probs, source, target, prob_floor, max_minterms)
    state_prep = state_prep.copy()
    state_prep.data = [ins for ins in state_prep.data if ins.operation.name != "barrier"]
    run_prep = state_prep
    if fast and state_prep.num_qubits <= 12:
        from qiskit.circuit.library import UnitaryGate
        from qiskit.quantum_info import Operator
        run_prep = QuantumCircuit(state_prep.num_qubits, name="A")
        run_prep.append(UnitaryGate(Operator(state_prep), label="A"), range(state_prep.num_qubits))
    problem = EstimationProblem(state_preparation=run_prep, objective_qubits=[obj_q])
    return CompiledResilienceProblem(problem, state_prep, problem.grover_operator, edges, obj_q, terms)


# ---------------------------------------------------------------------------
# Cell 6: IQAE
# ---------------------------------------------------------------------------
def run_iqae(problem, epsilon=0.02, alpha=0.05, shots=100, seed=7):
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    iqae = IterativeAmplitudeEstimation(epsilon_target=epsilon, alpha=alpha, sampler=sampler)
    return iqae.estimate(problem)


def state_prep_calls(result, shots) -> int:
    """Notebook's strict count: each shot of Q^k A runs A (2k+1) times."""
    return int(shots * sum(2 * k + 1 for k in result.powers[1:]))


# ---------------------------------------------------------------------------
# Cell 7: resource profiler
# ---------------------------------------------------------------------------
def profile_resources(graph, failure_probs, source, target, include_grover=True, transpile_for_basis=False):
    compiled = build_estimation_problem(graph, failure_probs, source, target)
    sp = compiled.state_preparation
    full = QuantumCircuit(sp.num_qubits)
    full.compose(sp, inplace=True)
    if include_grover:
        full.compose(compiled.grover_operator, inplace=True)
    if transpile_for_basis:
        full = transpile(full, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    return {
        "total_qubits": sp.num_qubits,
        "edge_qubits": len(compiled.edge_list),
        "oracle_minterms": compiled.oracle_terms,
        "state_prep_depth": sp.depth(),
        "state_prep_gate_counts": dict(sp.count_ops()),
        "profiled_circuit_depth": full.depth(),
        "profiled_circuit_gate_counts": dict(full.count_ops()),
    }


def exact_amplitude(circuit, obj_qubit) -> float:
    """Exact P(objective = 1) from the full statevector (no sampling)."""
    return float(Statevector(circuit).probabilities([obj_qubit])[1])



# ---------------------------------------------------------------------------
# Cell 10: noise
# ---------------------------------------------------------------------------
def noisy_estimate(graph, failure_probs, source, target, k, rho, shots=4096, seed=42):
    """Run A Q^k with depolarizing noise, recover P_disc. Returns (estimate, depth, cx_count)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    compiled = build_estimation_problem(graph, failure_probs, source, target)
    sp, obj_q = compiled.state_preparation, compiled.objective_qubit
    qc = QuantumCircuit(sp.num_qubits, 1)
    qc.compose(sp, inplace=True)
    for _ in range(k):
        qc.compose(compiled.grover_operator, inplace=True)
    qc.measure(obj_q, 0)
    tqc = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    if rho == 0.0:
        backend = AerSimulator(seed_simulator=seed)
    else:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(rho, 1),
                                       ["u1", "u2", "u3", "rx", "ry", "rz", "x", "sx", "h"])
        nm.add_all_qubit_quantum_error(depolarizing_error(min(2.0 * rho, 0.99), 2), ["cx", "cz", "mcx"])
        backend = AerSimulator(noise_model=nm, seed_simulator=seed)
    counts = backend.run(tqc, shots=shots).result().get_counts()
    p_meas = counts.get("1", 0) / sum(counts.values())
    return recover_probability(p_meas, k), tqc.depth(), tqc.count_ops().get("cx", 0)
