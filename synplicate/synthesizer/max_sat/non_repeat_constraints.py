"""
Utilities for building hard constraints that prevent predicate repetition
along any root-to-leaf path of an acyclic decision diagram.
"""
from typing import Dict, List, Tuple


def encode_hard_constraints(k: int, m: int) -> Tuple[Dict[str, int], List[List[int]]]:
    """Create CNF clauses that enforce predicate non-repetition.

    The encoding follows the specification given in the non-repeatability prompt:
    * ``lambda_{i,p}`` assigns predicate ``p`` to template node ``i``.
    * ``u_{i}`` marks node ``i`` as reachable from the root.
    * ``lambda'_{i,p}`` (``lambda_prime``) is true iff ``u_i`` and ``lambda_{i,p}`` are
      both true, i.e., predicate ``p`` is used at a reachable node.
    * ``tau_{i,c,j}`` routes the outcome ``c`` from node ``i`` to successor ``j``
      where ``c`` is ``0`` (False) or ``1`` (True).
    * ``R_{i,j}`` is true when ``j`` is reachable from ``i``.

    Args:
        k: Number of nodes (topologically ordered 1..k).
        m: Number of predicates.

    Returns:
        A pair ``(var_map, clauses)`` where ``var_map`` maps symbolic variable
        names to integer identifiers and ``clauses`` is a list of CNF clauses
        using those identifiers.
    """

    def new_var(name: str) -> int:
        nonlocal next_var
        next_var += 1
        var_map[name] = next_var
        return next_var

    var_map: Dict[str, int] = {}
    clauses: List[List[int]] = []
    next_var = 0

    lambda_vars = [[new_var(f"lambda_{i}_{p}") for p in range(m)] for i in range(k)]
    tau_vars = [[[new_var(f"tau_{i}_{c}_{j}") for j in range(i + 1, k)] for c in range(2)] for i in range(k)]
    reach_vars = [[new_var(f"R_{i}_{j}") if j > i else None for j in range(k)] for i in range(k)]
    used_vars = [new_var(f"u_{i}") for i in range(k)]
    lambda_prime_vars = [[new_var(f"lambda_prime_{i}_{p}") for p in range(m)] for i in range(k)]

    # Node-predicate assignment (exactly one predicate per node)
    for i in range(k):
        clauses.append([lambda_vars[i][p] for p in range(m)])
        for p1 in range(m):
            for p2 in range(p1 + 1, m):
                clauses.append([-lambda_vars[i][p1], -lambda_vars[i][p2]])

    # Transition completeness (exactly one successor for each outcome)
    for i in range(k):
        for c in range(2):
            clauses.append([tau_vars[i][c][j] for j in range(k - i - 1)])
            for j1 in range(k - i - 1):
                for j2 in range(j1 + 1, k - i - 1):
                    clauses.append([-tau_vars[i][c][j1], -tau_vars[i][c][j2]])

    # Reachable/used nodes u_i: root is used; others require an incoming used parent
    clauses.append([used_vars[0]])
    for i in range(1, k):
        incoming_conj_vars: List[int] = []
        for parent in range(i):
            for c in range(2):
                tau_lit = tau_vars[parent][c][i - parent - 1]
                conj = new_var(f"u_conj_{parent}_{c}_{i}")
                incoming_conj_vars.append(conj)
                clauses.append([-conj, tau_lit])
                clauses.append([-conj, used_vars[parent]])
                clauses.append([-tau_lit, -used_vars[parent], conj])
        clauses.append([-used_vars[i]] + incoming_conj_vars)
        for conj in incoming_conj_vars:
            clauses.append([-conj, used_vars[i]])

    # lambda_prime_{i,p} <-> (u_i & lambda_{i,p})
    for i in range(k):
        for p in range(m):
            lam_prime = lambda_prime_vars[i][p]
            lam = lambda_vars[i][p]
            u_i = used_vars[i]
            clauses.append([-lam_prime, u_i])
            clauses.append([-lam_prime, lam])
            clauses.append([-u_i, -lam, lam_prime])

    # Direct reachability from transitions
    for i in range(k):
        for j_offset, j in enumerate(range(i + 1, k)):
            for c in range(2):
                clauses.append([-tau_vars[i][c][j_offset], reach_vars[i][j]])

    # Transitive reachability
    for i in range(k):
        for mid in range(i + 1, k - 1):
            for j in range(mid + 1, k):
                clauses.append([-reach_vars[i][mid], -reach_vars[mid][j], reach_vars[i][j]])

    # Predicate non-repetition along reachable paths for active predicates only
    for p in range(m):
        for i in range(k):
            for j in range(i + 1, k):
                clauses.append(
                    [
                        -lambda_prime_vars[i][p],
                        -reach_vars[i][j],
                        -lambda_prime_vars[j][p],
                    ]
                )

    return var_map, clauses
