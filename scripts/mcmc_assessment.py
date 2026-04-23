"""Adaptive knowledge-component assessment driven by MCMC.

The script loads the prerequisite DAG from ``dag.json`` and runs MCMC to estimate, for every remaining (uncertain)
knowledge component (KC) ``v``, the proportion ``p(v)`` of ideals containing ``v``

At every step it picks

    v* = argmin_v |p(v) - 0.5|

and asks the user whether they know it.

* Answer YES  -> the student knows ``v*`` and, by prerequisite closure,
  every ancestor of ``v*`` in the prereq DAG.
* Answer NO   -> the student does not know ``v*`` and, by prerequisite
  closure, every descendant of ``v*``.

The loop repeats on the remaining (still-uncertain) KCs until every KC
has been classified.
"""

from __future__ import annotations

import json
import os
import random
import sys
from typing import Iterable

import numpy as np


def load_dag(path: str) -> tuple[list[str], list[list[int]], list[list[int]]]:
    """Load the DAG and return (names, successors, predecessors)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    names = [node["name"] for node in data["nodes"]]
    name_to_idx = {name: i for i, name in enumerate(names)}
    n = len(names)

    successors: list[list[int]] = [[] for _ in range(n)]
    predecessors: list[list[int]] = [[] for _ in range(n)]
    for edge in data["edges"]:
        u = name_to_idx[edge["prereq"]]
        v = name_to_idx[edge["concept"]]
        successors[u].append(v)
        predecessors[v].append(u)
    return names, successors, predecessors


def run_mcmc(
    active: list[int],
    successors: list[list[int]],
    predecessors: list[list[int]],
    n_steps: int = 200_000,
    seed: int | None = None,
) -> np.ndarray:
    """Estimate ``p[v]`` for every node in ``active``.

    Implements the same Markov chain as ``mcmc.py`` but restricted to the
    induced subgraph on ``active``. ``p[v]`` is zero for nodes outside
    ``active``.
    """
    if seed is not None:
        random.seed(seed)

    n_total = len(successors)
    active_set = set(active)

    in_A = np.zeros(n_total, dtype=bool)
    out_degree_A = np.zeros(n_total, dtype=np.int64)
    in_degree_B = np.zeros(n_total, dtype=np.int64)
    for v in active:
        in_degree_B[v] = sum(1 for u in predecessors[v] if u in active_set)

    p = np.zeros(n_total, dtype=np.float64)
    last_enter = np.zeros(n_total, dtype=np.int64)

    m = len(active)
    if m == 0:
        return p

    for i in range(n_steps):
        k = active[random.randrange(m)]

        if not in_A[k] and in_degree_B[k] == 0:
            in_A[k] = True
            last_enter[k] = i
            for w in successors[k]:
                if w in active_set:
                    in_degree_B[w] -= 1
            for w in predecessors[k]:
                if w in active_set:
                    out_degree_A[w] += 1

        elif in_A[k] and out_degree_A[k] == 0:
            in_A[k] = False
            p[k] += i - last_enter[k]
            for w in successors[k]:
                if w in active_set:
                    in_degree_B[w] += 1
            for w in predecessors[k]:
                if w in active_set:
                    out_degree_A[w] -= 1

    for k in active:
        if in_A[k]:
            p[k] += n_steps - last_enter[k]

    p /= n_steps
    return p


def reachable(
    start: int,
    neighbors: list[list[int]],
    active_set: set[int],
) -> set[int]:
    """Return the set of nodes reachable from ``start`` (inclusive) through
    ``neighbors``, restricted to ``active_set``."""
    result: set[int] = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in result:
            continue
        result.add(u)
        for w in neighbors[u]:
            if w in active_set and w not in result:
                stack.append(w)
    return result


def ask_yes_no(prompt: str) -> str:
    """Prompt the user until they answer y/n/q."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in {"y", "yes"}:
            return "y"
        if ans in {"n", "no"}:
            return "n"
        if ans in {"q", "quit", "exit"}:
            return "q"
        print("  Please answer 'y' (yes), 'n' (no), or 'q' (quit).")


def format_names(names: list[str], idxs: Iterable[int]) -> str:
    sorted_names = sorted(names[i] for i in idxs)
    return ", ".join(sorted_names) if sorted_names else "(none)"


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dag_path = os.path.join(script_dir, "dag.json")

    names, successors, predecessors = load_dag(dag_path)
    n = len(names)

    known: set[int] = set()
    unknown: set[int] = set()
    active: set[int] = set(range(n))

    n_steps = 200_000
    print(f"Loaded DAG with {n} KCs and "
          f"{sum(len(s) for s in successors)} prerequisite edges.")
    print(f"Running {n_steps} MCMC steps per question.\n")
    print("Answer each question with 'y' (yes), 'n' (no) or 'q' (quit).\n")

    step = 0
    while active:
        step += 1
        active_list = sorted(active)
        p = run_mcmc(active_list, successors, predecessors, n_steps=n_steps)

        best_v = min(active_list, key=lambda v: abs(p[v] - 0.5))
        print(f"--- Step {step} | {len(active)} KC(s) remaining ---")
        print(f"Most informative KC: '{names[best_v]}' "
              f"(estimated p = {p[best_v]:.3f})")

        ans = ask_yes_no(f"Do you know '{names[best_v]}'? [y/n/q]: ")
        if ans == "q":
            print("\nAssessment interrupted by user.")
            break

        if ans == "y":
            newly = reachable(best_v, predecessors, active)
            known |= newly
            active -= newly
            print(f"  -> Marked {len(newly)} KC(s) as KNOWN: "
                  f"{format_names(names, newly)}\n")
        else:
            newly = reachable(best_v, successors, active)
            unknown |= newly
            active -= newly
            print(f"  -> Marked {len(newly)} KC(s) as UNKNOWN: "
                  f"{format_names(names, newly)}\n")

    print("=" * 60)
    print("Assessment complete.")
    print(f"KNOWN ({len(known)}): {format_names(names, known)}")
    print(f"UNKNOWN ({len(unknown)}): {format_names(names, unknown)}")
    if active:
        print(f"UNDETERMINED ({len(active)}): {format_names(names, active)}")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nAssessment aborted.")
        sys.exit(0)
