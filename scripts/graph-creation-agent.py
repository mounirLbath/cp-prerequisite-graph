"""
Adversarial Difficulty-DAG Builder with LangGraph
=================================================

Takes a flat list of knowledge components (KCs) — one per line in `kc.txt` —
and builds a DIRECTED ACYCLIC GRAPH of prerequisite relationships ("easier /
more foundational KC" -> "harder / dependent KC").

The pipeline is a 4-step adversarial pass, each step being exactly ONE LLM
call (as requested):

    1. propose_edges   : the LLM proposes every plausible DIRECT
                         prerequisite edge over the whole node set.
    2. justify_edges   : one call returns a short pro-argument for every
                         proposed edge ("why this edge is correct").
    3. question_edges  : one call returns a short critical counter-argument
                         for every proposed edge ("why this edge might be
                         wrong / redundant / transitive").
    4. judge_edges     : final call: sees edges + justifications +
                         objections, keeps the strong ones, drops the weak,
                         and resolves cycles so the result is a proper DAG.

A final deterministic sweep (`enforce_dag`) removes any residual cycles the
judge left behind (by dropping the lowest-confidence edge on each cycle),
so the output is GUARANTEED acyclic.

Output: `dag.json` with the full node list, kept edges with
    {justification, objection, verdict, confidence} metadata, and a
    topological order.

Run:
    # put OPENAI_API_KEY=sk-... in .env
    python adv.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:")
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("adv_dag")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DAGConfig(BaseModel):
    kc_path: str = Field("kc.txt", description="Input file: one KC per line.")
    out_path: str = Field("dag.json", description="Output JSON file.")
    model: str = Field("gpt-5-mini", description="OpenAI chat model.")
    temperature: float = Field(0.0, description="LLM temperature.")
    # Keep edges whose judge confidence is >= this value (0-1).
    min_confidence: float = Field(0.5, description="Judge confidence cutoff.")


# ---------------------------------------------------------------------------
# Structured LLM outputs
# ---------------------------------------------------------------------------


class Edge(BaseModel):
    """A candidate direct prerequisite edge: prereq -> concept."""

    prereq: str = Field(
        ...,
        description=(
            "The EASIER / more foundational KC, required BEFORE the target. "
            "Must be spelled EXACTLY as in the input list."
        ),
    )
    concept: str = Field(
        ...,
        description=(
            "The HARDER / dependent KC. Must be spelled EXACTLY as in the "
            "input list."
        ),
    )


class ProposedEdges(BaseModel):
    edges: List[Edge] = Field(default_factory=list)


class EdgeJustification(BaseModel):
    prereq: str
    concept: str
    justification: str = Field(
        ...,
        description=(
            "1-2 sentence PRO argument: the concrete reason mastering "
            "`prereq` is a direct prerequisite for `concept` in a "
            "competitive-programming curriculum."
        ),
    )


class JustifiedEdges(BaseModel):
    items: List[EdgeJustification] = Field(default_factory=list)


class EdgeObjection(BaseModel):
    prereq: str
    concept: str
    objection: str = Field(
        ...,
        description=(
            "1-2 sentence CON argument: the strongest reason to DROP this "
            "edge — e.g. the relationship is transitive (implied by another "
            "edge), too weak, topically unrelated, or the ordering is "
            "actually the reverse."
        ),
    )


class QuestionedEdges(BaseModel):
    items: List[EdgeObjection] = Field(default_factory=list)


class JudgedEdge(BaseModel):
    prereq: str
    concept: str
    keep: bool = Field(
        ..., description="True to keep this edge in the final DAG."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0..1 confidence that this edge is a correct DIRECT prerequisite.",
    )
    verdict: str = Field(
        ...,
        description=(
            "1 sentence rationale for the keep/drop decision, weighing the "
            "justification against the objection."
        ),
    )


class JudgedEdges(BaseModel):
    items: List[JudgedEdge] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class DAGState(TypedDict, total=False):
    config: DAGConfig
    nodes: List[str]
    # Edge key -> metadata dict
    proposed: List[Tuple[str, str]]
    justifications: Dict[Tuple[str, str], str]
    objections: Dict[Tuple[str, str], str]
    judged: Dict[Tuple[str, str], Dict[str, Any]]
    final_edges: List[Tuple[str, str]]
    topo_order: List[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_llm(cfg: DAGConfig) -> BaseChatModel:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing. Put it in .env.")
    return ChatOpenAI(model=cfg.model, temperature=cfg.temperature, api_key=api_key)


def _invoke(llm: Any, messages: Any, *, call: str, max_retries: int = 5) -> Any:
    """LLM invoke with structured logging and 429-aware backoff."""
    attempt = 0
    while True:
        attempt += 1
        logger.info("LLM start | call=%s attempt=%s", call, attempt)
        t0 = time.perf_counter()
        try:
            result = llm.invoke(messages)
        except Exception as exc:  # pragma: no cover
            dt = time.perf_counter() - t0
            msg = str(exc)
            is_429 = "429" in msg or "Too Many Requests" in msg
            logger.warning(
                "LLM error | call=%s attempt=%s dt=%.2fs 429=%s err=%s",
                call, attempt, dt, is_429, msg.splitlines()[0][:180],
            )
            if is_429 and attempt <= max_retries:
                sleep = min(30.0, 2.0 ** attempt) + random.uniform(0, 1)
                logger.info("LLM retry | call=%s sleep=%.1fs", call, sleep)
                time.sleep(sleep)
                continue
            raise
        dt = time.perf_counter() - t0
        n_out = len(getattr(result, "items", getattr(result, "edges", [])) or [])
        logger.info("LLM done  | call=%s dt=%.2fs out=%s", call, dt, n_out)
        return result


def _canonicalize(name: str, valid: Dict[str, str]) -> Optional[str]:
    """Map an LLM-emitted name back to the EXACT input-list spelling.

    `valid` maps lowercased-stripped name -> canonical (original-cased) name.
    Returns None if the name cannot be reconciled with any input node.
    """
    key = re.sub(r"\s+", " ", name.strip().lower())
    if key in valid:
        return valid[key]
    # Tolerate a trailing 's' / missing 's' / hyphen/space swaps.
    variants = {
        key,
        key.rstrip("s"),
        key + "s",
        key.replace("-", " "),
        key.replace(" ", "-"),
    }
    for v in variants:
        if v in valid:
            return valid[v]
    return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


_SYSTEM_CONTEXT = (
    "You are building the PREREQUISITE / DIFFICULTY DAG of competitive "
    "programming. The full node set is FIXED — every edge must be between "
    "two nodes taken verbatim from the provided list (same spelling, same "
    "case). An edge `A -> B` means: to understand / solve problems using "
    "B, a learner must first master A. The graph must be ACYCLIC: if "
    "`A -> B` exists, `B -> A` (directly or transitively) must not.\n\n"
    "KCs are competitive-programming knowledge components only. Do not "
    "invent nodes. Do not output meta-terms.\n\n"
    "What counts as a DIRECT prerequisite (any ONE of these is enough):\n"
    "  (a) STRUCTURAL — `concept` is literally implemented on top of "
    "      `prereq` (e.g. segment tree -> segment tree with lazy prop; "
    "      DFS -> articulation points).\n"
    "  (b) ALGORITHMIC REUSE — the standard algorithm for `concept` "
    "      invokes `prereq` as a subroutine, reduction, or inner loop "
    "      (e.g. sorting -> sweep line; union-find -> Kruskal).\n"
    "  (c) CONCEPTUAL — `prereq` is the canonical mental model a learner "
    "      must internalise before `concept` makes sense (e.g. recursion "
    "      -> dynamic programming; graph representation -> BFS).\n"
    "  (d) ANALYSIS — `prereq` is the tool used to analyse `concept` "
    "      (e.g. amortized analysis -> union-find with path compression; "
    "      big-O -> most algorithms). Only use this if `prereq` is on "
    "      the list AND genuinely required to reason about `concept`.\n"
    "  (e) MATHEMATICAL — `concept` relies on the math of `prereq` "
    "      (e.g. modular arithmetic -> number-theoretic transform; "
    "      combinatorics -> inclusion-exclusion).\n"
)


def _propose_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _SYSTEM_CONTEXT
                + "\n\nTASK: emit every plausible DIRECT prerequisite edge "
                "in the node set. Your goal is HIGH RECALL — a later "
                "adversarial judge will prune weak edges, so err on the "
                "side of proposing. Missing a real edge is worse than "
                "proposing a questionable one. Be generous in the number of edges emitted.\n\n"
                "Density target:\n"
                "  * Aim for roughly 3-8 direct prerequisites per "
                "    non-foundational concept (some hubs will have more, "
                "    some will have 1-2). As a rough calibration, total "
                "    edges ≈ 4× to 6× the number of nodes in a healthy "
                "    competitive-programming curriculum.\n"
                "  * Every node except the most foundational ones should "
                "    appear as a `concept` on at least one edge. Every "
                "    node except the most advanced ones should appear as "
                "    a `prereq` on at least one edge. If a node appears "
                "    neither as prereq nor concept, you are probably "
                "    missing edges for it — reconsider.\n\n"
                "Systematic sweep: before finalising, walk the list once "
                "more and, for each node, ask both 'what does this unlock?' "
                "and 'what must I know first?'. Propose across ALL of "
                "(a)-(e) above, not just structural inheritance.\n\n"
                "Concrete seed examples (for intuition only; not exhaustive):\n"
                "  * sorting -> binary search on sorted data; sorting -> "
                "    sweep line; sorting -> convex hull.\n"
                "  * depth-first search -> topological sort; depth-first "
                "    search -> strongly connected components; depth-first "
                "    search -> articulation points.\n"
                "  * dynamic programming -> dp on trees; dp on trees -> "
                "    rerooting; dp -> bitmask dp; dp -> digit dp.\n"
                "  * segment tree -> segment tree with lazy propagation; "
                "    Fenwick tree -> 2D Fenwick tree.\n"
                "  * modular arithmetic -> modular inverse -> "
                "    combinatorics mod p; number-theoretic transform "
                "    needs both modular arithmetic and fast fourier "
                "    transform.\n"
                "  * union-find -> Kruskal's algorithm; offline "
                "    queries + union-find -> small-to-large / DSU on tree.\n\n"
                "Hard rules:\n"
                "  * DIRECT only: if C needs B and B needs A, you should "
                "    still emit A -> B and B -> C (both direct), but NOT "
                "    A -> C (implied transitively).\n"
                "  * Orientation: easier / more foundational -> harder / "
                "    more specialised. When unsure, ask: which one would "
                "    a curriculum teach first?\n"
                "  * No self-loops, no duplicates, no cycles.\n"
                "  * Use the EXACT strings from the input list (same "
                "    spelling, same case, same punctuation).",
            ),
            ("human", "Nodes (one per line):\n{nodes}"),
        ]
    )


def _justify_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _SYSTEM_CONTEXT
                + "\n\nTASK: for EACH proposed edge `prereq -> concept`, "
                "write the STRONGEST 1-2 sentence PRO argument explaining "
                "why mastering `prereq` is a genuine DIRECT prerequisite "
                "for `concept`. Identify which of the relation types "
                "(a)-(e) above best fits, then state it concretely: name "
                "the sub-technique, the reduction, the reused data "
                "structure, or the specific theorem. Good justifications "
                "mention a concrete algorithm, loop, or proof step.\n\n"
                "Preserve the edge list exactly; do not add, drop, or "
                "rename edges. Return exactly one justification per input "
                "edge, with the same spelling.",
            ),
            ("human", "Proposed edges:\n{edges}"),
        ]
    )


def _question_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _SYSTEM_CONTEXT
                + "\n\nTASK: for EACH proposed edge `prereq -> concept`, "
                "write the STRONGEST 1-2 sentence CON argument. Be a "
                "CALIBRATED critic, not a reflexive one — only raise "
                "objections that a judge would actually act on.\n\n"
                "Legitimate objections (in order of severity):\n"
                "  1. INVERTED — `concept` is actually the prereq (strong "
                "     objection; the edge should be flipped or dropped).\n"
                "  2. UNRELATED — the two topics sit in different "
                "     sub-fields with no real methodological link (strong).\n"
                "  3. TRANSITIVE via a SPECIFIC node X that is in the list "
                "     — i.e. prereq -> X -> concept is the direct chain, "
                "     so prereq -> concept is not DIRECT (medium).\n"
                "  4. WEAK / INCIDENTAL — the link is real but peripheral "
                "     (mild; a judge will probably still keep it).\n\n"
                "If the edge looks clearly correct and none of the above "
                "applies cleanly, SAY SO: write 'No strong objection; edge "
                "looks direct via <reason>.' Do NOT fabricate a "
                "transitive-via-X objection unless you can name the "
                "specific intermediate node X that is actually in the "
                "list. Over-objection leads to a sparse, broken graph.\n\n"
                "Preserve the edge list exactly; do not add, drop, or "
                "rename edges. Return exactly one objection per input edge.",
            ),
            ("human", "Proposed edges:\n{edges}"),
        ]
    )


def _judge_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                _SYSTEM_CONTEXT
                + "\n\nTASK: you are the JUDGE. For each edge you see the "
                "justification (PRO) and the objection (CON). Decide "
                "KEEP vs DROP and assign a confidence in [0, 1].\n\n"
                "Default posture: KEEP. A rich prerequisite graph is more "
                "useful than a sparse one, and the user can always filter "
                "by confidence later. Drop an edge only when the CON is "
                "SPECIFIC and DECISIVE.\n\n"
                "Confidence calibration (use this scale):\n"
                "  * 0.90-1.00 — textbook, uncontroversial prerequisite "
                "    (e.g. DFS -> topological sort).\n"
                "  * 0.75-0.89 — clearly direct, standard curriculum link.\n"
                "  * 0.60-0.74 — plausible direct link, some dependence on "
                "    teaching style; keep.\n"
                "  * 0.45-0.59 — marginal; keep if the PRO is concrete, "
                "    drop only if the CON names a specific decisive flaw.\n"
                "  * < 0.45 — drop (inverted, unrelated, or clearly "
                "    transitive via a named node in the list).\n\n"
                "Keep if ANY of: the relation is structural (concept built "
                "on prereq), algorithmic reuse is explicit, or the "
                "conceptual dependency is standard. DROP only if the "
                "objection clearly identifies inversion, topical mismatch, "
                "or transitivity via a SPECIFIC named node that is also "
                "in the list.\n\n"
                "Generic 'could be transitive' is NOT a reason to drop — "
                "only drop for transitivity when the specific intermediate "
                "is named and obviously the direct link. Vague 'weak '\n"
                "connection' is also not a reason to drop.\n\n"
                "Cycle rule: NEVER keep edges that would form a directed "
                "cycle. If two edges in the batch form a cycle, keep the "
                "one with stronger PRO and drop the other (the downstream "
                "DAG-enforcer will also break residual cycles, but you "
                "should do it here when obvious).\n\n"
                "Output exactly one verdict per input edge, with the SAME "
                "prereq/concept spelling. Aim to KEEP at least 60-75 percent "
                "of the edges; if you are dropping more than that, your "
                "bar is too high.",
            ),
            ("human", "Edges with pro/con:\n{items}"),
        ]
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def node_load(state: DAGState) -> DAGState:
    cfg = state["config"]
    path = Path(cfg.kc_path)
    nodes: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            nodes.append(s)
    # dedupe while preserving order
    seen: set = set()
    uniq: List[str] = []
    for n in nodes:
        if n.lower() not in seen:
            seen.add(n.lower())
            uniq.append(n)
    logger.info("Loaded %d KC nodes from %s", len(uniq), cfg.kc_path)
    return {"nodes": uniq}


def node_propose(state: DAGState) -> DAGState:
    cfg = state["config"]
    nodes = state["nodes"]
    llm = _build_llm(cfg).with_structured_output(ProposedEdges)
    prompt = _propose_prompt()
    msgs = prompt.format_messages(nodes="\n".join(f"- {n}" for n in nodes))
    result: ProposedEdges = _invoke(llm, msgs, call="propose")

    valid = {n.lower(): n for n in nodes}
    proposed: List[Tuple[str, str]] = []
    seen_pairs: set = set()
    dropped_bad = 0
    dropped_self = 0
    dropped_dup = 0
    for e in result.edges:
        p = _canonicalize(e.prereq, valid)
        c = _canonicalize(e.concept, valid)
        if not p or not c:
            dropped_bad += 1
            continue
        if p == c:
            dropped_self += 1
            continue
        key = (p, c)
        if key in seen_pairs:
            dropped_dup += 1
            continue
        seen_pairs.add(key)
        proposed.append(key)
    logger.info(
        "Proposed edges: kept=%d bad=%d self=%d dup=%d",
        len(proposed), dropped_bad, dropped_self, dropped_dup,
    )
    return {"proposed": proposed}


def _format_edges(edges: List[Tuple[str, str]]) -> str:
    return "\n".join(f"- {p} -> {c}" for p, c in edges)


def node_justify(state: DAGState) -> DAGState:
    cfg = state["config"]
    edges = state["proposed"]
    if not edges:
        return {"justifications": {}}
    llm = _build_llm(cfg).with_structured_output(JustifiedEdges)
    msgs = _justify_prompt().format_messages(edges=_format_edges(edges))
    result: JustifiedEdges = _invoke(llm, msgs, call="justify")

    valid = {n.lower(): n for n in state["nodes"]}
    out: Dict[Tuple[str, str], str] = {}
    for item in result.items:
        p = _canonicalize(item.prereq, valid)
        c = _canonicalize(item.concept, valid)
        if p and c and (p, c) in set(edges):
            out[(p, c)] = item.justification.strip()
    # Fill blanks so every edge has an entry (judge expects full coverage).
    for e in edges:
        out.setdefault(e, "(no justification returned)")
    logger.info("Justified %d/%d edges", sum(
        1 for e in edges if out[e] != "(no justification returned)"
    ), len(edges))
    return {"justifications": out}


def node_question(state: DAGState) -> DAGState:
    cfg = state["config"]
    edges = state["proposed"]
    if not edges:
        return {"objections": {}}
    llm = _build_llm(cfg).with_structured_output(QuestionedEdges)
    msgs = _question_prompt().format_messages(edges=_format_edges(edges))
    result: QuestionedEdges = _invoke(llm, msgs, call="question")

    valid = {n.lower(): n for n in state["nodes"]}
    out: Dict[Tuple[str, str], str] = {}
    for item in result.items:
        p = _canonicalize(item.prereq, valid)
        c = _canonicalize(item.concept, valid)
        if p and c and (p, c) in set(edges):
            out[(p, c)] = item.objection.strip()
    for e in edges:
        out.setdefault(e, "(no objection returned)")
    logger.info("Questioned %d/%d edges", sum(
        1 for e in edges if out[e] != "(no objection returned)"
    ), len(edges))
    return {"objections": out}


def node_judge(state: DAGState) -> DAGState:
    cfg = state["config"]
    edges = state["proposed"]
    justs = state["justifications"]
    objs = state["objections"]
    if not edges:
        return {"judged": {}}

    payload_lines: List[str] = []
    for p, c in edges:
        payload_lines.append(
            f"- {p} -> {c}\n"
            f"    PRO: {justs.get((p, c), '')}\n"
            f"    CON: {objs.get((p, c), '')}"
        )
    items_block = "\n".join(payload_lines)

    llm = _build_llm(cfg).with_structured_output(JudgedEdges)
    msgs = _judge_prompt().format_messages(items=items_block)
    result: JudgedEdges = _invoke(llm, msgs, call="judge")

    valid = {n.lower(): n for n in state["nodes"]}
    judged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in result.items:
        p = _canonicalize(item.prereq, valid)
        c = _canonicalize(item.concept, valid)
        if not p or not c or (p, c) not in set(edges):
            continue
        judged[(p, c)] = {
            "keep": bool(item.keep),
            "confidence": float(item.confidence),
            "verdict": item.verdict.strip(),
        }
    # Default-keep any edge the judge forgot, at medium confidence.
    for e in edges:
        judged.setdefault(e, {
            "keep": True, "confidence": 0.5, "verdict": "(no verdict; default keep)",
        })
    kept = sum(1 for v in judged.values() if v["keep"])
    logger.info("Judge kept=%d dropped=%d of %d", kept, len(judged) - kept, len(judged))
    return {"judged": judged}


def node_enforce_dag(state: DAGState) -> DAGState:
    """Finalize edges: apply judge's keep + confidence cutoff, then break
    any residual cycles deterministically by dropping the lowest-confidence
    edge on each cycle until the graph is acyclic."""
    cfg = state["config"]
    nodes = state["nodes"]
    judged = state["judged"]

    # Step 1: keep edges the judge approved and whose confidence >= threshold.
    candidates: List[Tuple[str, str, float]] = []
    for (p, c), meta in judged.items():
        if not meta["keep"]:
            continue
        if meta["confidence"] < cfg.min_confidence:
            continue
        candidates.append((p, c, meta["confidence"]))

    # Sort so that higher-confidence edges are preferred when we greedily
    # break cycles.
    candidates.sort(key=lambda t: -t[2])

    adj: Dict[str, set] = {n: set() for n in nodes}
    kept: List[Tuple[str, str]] = []

    def _reaches(src: str, dst: str) -> bool:
        """Is there a directed path from src to dst in the current `adj`?"""
        if src == dst:
            return True
        stack = [src]
        seen = {src}
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v == dst:
                    return True
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return False

    dropped_cycle = 0
    for p, c, _conf in candidates:
        if p not in adj or c not in adj:
            continue
        if _reaches(c, p):
            # Adding p -> c would close a cycle (since c already reaches p).
            dropped_cycle += 1
            continue
        adj[p].add(c)
        kept.append((p, c))

    logger.info(
        "Final DAG: nodes=%d edges=%d (dropped for cycle=%d, below conf=%d)",
        len(nodes), len(kept), dropped_cycle,
        sum(1 for (_, _), m in judged.items()
            if m["keep"] and m["confidence"] < cfg.min_confidence),
    )

    # Topological order (Kahn's algorithm).
    indeg: Dict[str, int] = {n: 0 for n in nodes}
    for u in nodes:
        for v in adj[u]:
            indeg[v] += 1
    frontier = [n for n in nodes if indeg[n] == 0]
    topo: List[str] = []
    while frontier:
        frontier.sort()  # deterministic
        u = frontier.pop(0)
        topo.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                frontier.append(v)
    if len(topo) != len(nodes):  # pragma: no cover - belt & suspenders
        raise RuntimeError("Residual cycle after enforcement (should be impossible).")

    return {"final_edges": kept, "topo_order": topo}


def node_save(state: DAGState) -> DAGState:
    cfg = state["config"]
    nodes = state["nodes"]
    edges = state.get("final_edges", [])
    judged = state.get("judged", {})
    justs = state.get("justifications", {})
    objs = state.get("objections", {})
    topo = state.get("topo_order", [])

    depth: Dict[str, int] = {n: 0 for n in nodes}
    for u in topo:
        for (p, c) in edges:
            if p == u:
                depth[c] = max(depth[c], depth[u] + 1)

    payload = {
        "nodes": [
            {"name": n, "depth": depth[n]} for n in nodes
        ],
        "edges": [
            {
                "prereq": p,
                "concept": c,
                "justification": justs.get((p, c), ""),
                "objection": objs.get((p, c), ""),
                "verdict": judged.get((p, c), {}).get("verdict", ""),
                "confidence": judged.get((p, c), {}).get("confidence", 0.0),
            }
            for (p, c) in edges
        ],
        "topological_order": topo,
        "stats": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "max_depth": max(depth.values()) if depth else 0,
        },
    }
    Path(cfg.out_path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote %s (%d nodes, %d edges)", cfg.out_path, len(nodes), len(edges))
    return {}


# ---------------------------------------------------------------------------
# LangGraph assembly
# ---------------------------------------------------------------------------


def build_graph():
    g = StateGraph(DAGState)
    g.add_node("load", node_load)
    g.add_node("propose", node_propose)
    g.add_node("justify", node_justify)
    g.add_node("question", node_question)
    g.add_node("judge", node_judge)
    g.add_node("enforce_dag", node_enforce_dag)
    g.add_node("save", node_save)

    g.add_edge(START, "load")
    g.add_edge("load", "propose")
    g.add_edge("propose", "justify")
    g.add_edge("justify", "question")
    g.add_edge("question", "judge")
    g.add_edge("judge", "enforce_dag")
    g.add_edge("enforce_dag", "save")
    g.add_edge("save", END)
    return g.compile()


def run(cfg: Optional[DAGConfig] = None) -> Dict[str, Any]:
    cfg = cfg or DAGConfig()
    app = build_graph()
    out = app.invoke({"config": cfg}, {"recursion_limit": 50})
    return {
        "nodes": out.get("nodes", []),
        "edges": out.get("final_edges", []),
        "topo_order": out.get("topo_order", []),
    }


if __name__ == "__main__":
    result = run()
    print(
        f"\nDone. Nodes={len(result['nodes'])} "
        f"Edges={len(result['edges'])} "
        f"Topo-head={result['topo_order'][:5]}"
    )
