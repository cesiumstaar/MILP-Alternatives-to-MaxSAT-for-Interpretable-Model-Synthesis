# Weighted MaxSAT encoding notes

## Slides artifact
- Confirmed that the provided paper is present at `slides/paper.pdf`.

## Query 1 – Structural hard constraint
The proposed hard clause
\[
\bigwedge_{1\le i<j\le k} \bigwedge_{p\in P} \bigl(R_{i,j} \rightarrow \neg (\lambda'_{i,p} \wedge \lambda'_{j,p})\bigr)
\]
uses the composite variable \(\lambda'_{i,p} \leftrightarrow u_i \wedge \lambda_{i,p}\). Because \(\lambda'_{i,p}\) is true only when node \(i\) is reachable/used (\(u_i\)) **and** actually carries predicate \(p\), the constraint blocks duplicates exclusively along realized root-to-leaf paths. Predicates assigned to unused nodes do not trigger the clause, so redundancy is forbidden only in the final, reachable diagram.

## Query 2 – Balancing correctness and explainability objectives
Weighted MaxSAT solvers simply aggregate soft-clause weights. To obtain a balanced Pareto search, it can be helpful (but is not logically required) to normalize the weight ranges so that the maximum attainable correctness weight \(W_{\mathcal{C}}^{\max}\) is comparable to the explainability weight \(W_{\mathcal{E}}^{\max}\). Equalizing these totals removes scale bias, but you can instead choose any scaling that reflects the desired priority between the two objectives.

## Query 3 – Hard-clause weight
To guarantee every returned model satisfies all hard clauses, assign a hard weight strictly greater than the sum of all soft-clause weights:
\[
W_H > W_{\mathcal{C}}^{\max} + W_{\mathcal{E}}^{\max}.
\]
A common choice is \(W_H = W_{\mathcal{C}}^{\max} + W_{\mathcal{E}}^{\max} + 1\). With this bound, violating any hard clause carries more cost than falsifying every soft clause combined, so every optimal MaxSAT solution must satisfy the hard constraints (including the syntactic constraints \(\phi_{\mathcal{E}}\), semantic constraints \(\phi_{\mathcal{S}}\), and Tseitin auxiliaries).
