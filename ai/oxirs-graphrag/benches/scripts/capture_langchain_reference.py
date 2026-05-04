#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture LangChain GraphRAG reference outputs for the oxirs-graphrag KGQA benchmark.

OPERATOR-ONLY SCRIPT. NEVER EXECUTED IN CI.
============================================

This script is vendored alongside the oxirs-graphrag benchmark harness so that
operators can reproduce a pinned LangChain GraphRAG comparison locally. The
output files it produces (one JSON per question) feed Phase 2 of the
``benches/langchain_kgqa.rs`` benchmark when the ``LANGCHAIN_REF_FIXTURES``
environment variable points at the directory of captures.

Workflow
--------

1. Install the pinned LangChain version locally (see ``--langchain-version``).
2. Run this script against ``webqsp_subset.json``. It will:
     - Load the KG triples and questions.
     - Build an in-memory NetworkX graph.
     - Run the LangChain GraphRAG pipeline once per question.
     - Emit one ``<qid>.json`` file in the output directory containing the
       ranked answer list.
3. Set ``LANGCHAIN_REF_FIXTURES=/path/to/output/dir`` before invoking
   ``cargo bench -p oxirs-graphrag``. Phase 2 of the bench will then
   assert that oxirs-graphrag's Hits@5 lies within 5pp of LangChain's.

Usage
-----

    python3 capture_langchain_reference.py \\
        --fixture ../fixtures/kgqa-bench/webqsp_subset.json \\
        --output  /tmp/langchain-ref \\
        --langchain-version 0.3.0

CI policy
---------

This file is intentionally **not** invoked from any CI job. It carries a
hard runtime check that prints a warning if executed inside a CI runner
(``CI=true``). Do not introduce a pytest test that imports this script;
the benchmark harness shells out to JSON files only.

Output schema
-------------

Each output file ``<qid>.json`` contains::

    {
      "qid":           "WebQSP-XXX",
      "ranked_answers": ["iri", "iri", ...],
      "langchain_version": "0.3.0",
      "captured_at":   "2026-04-30T12:34:56Z"
    }
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def _is_ci() -> bool:
    """Return ``True`` when the script is being executed inside a CI runner.

    Checks the canonical ``CI`` env var (set by GitHub Actions, GitLab CI,
    CircleCI, etc.) and emits a hard warning so that operator-only scripts
    are never silently invoked in automation.
    """
    return os.environ.get("CI", "").lower() in {"1", "true", "yes"}


def _load_fixture(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_langchain_pipeline(kg: List[Dict[str, str]], labels: Dict[str, str]) -> Any:
    """Build a LangChain GraphRAG pipeline from the in-memory KG.

    Imports are intentionally local so that the rest of the script remains
    importable without LangChain installed (operators often inspect the
    docstring before installing the heavy dependency tree).
    """
    try:
        from langchain.chains import GraphQAChain  # type: ignore
        from langchain.graphs import NetworkxEntityGraph  # type: ignore
        from langchain.llms import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - operator-only
        sys.stderr.write(
            "ERROR: This script requires the LangChain runtime. "
            "Install it locally with `pip install 'langchain==0.3.*'`.\n"
            f"Underlying ImportError: {exc}\n"
        )
        sys.exit(2)

    graph = NetworkxEntityGraph()
    for triple in kg:
        graph.add_triple(triple["s"], triple["p"], triple["o"])

    chain = GraphQAChain.from_llm(OpenAI(temperature=0.0), graph=graph)
    chain.graph_labels = labels  # convenience attribute consumed by some chains
    return chain


def _rank_answers(chain: Any, question: str, top_k: int) -> List[str]:
    """Run the LangChain chain for ``question`` and return its top-K answer IRIs.

    Returns an ordered list (best → worst) so downstream Hits@K and MRR
    metrics line up with the oxirs-graphrag bench harness.
    """
    response = chain.invoke({"query": question})
    raw = response.get("result") if isinstance(response, dict) else str(response)
    if not raw:
        return []
    # Each LangChain version returns a slightly different shape. We accept:
    #   1. A newline-separated list of IRIs.
    #   2. A JSON array.
    #   3. A space-separated list of IRIs.
    raw = raw.strip()
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(item) for item in parsed][:top_k]
        except json.JSONDecodeError:
            pass
    if "\n" in raw:
        return [line.strip() for line in raw.splitlines() if line.strip()][:top_k]
    return raw.split()[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture LangChain GraphRAG reference outputs for "
        "oxirs-graphrag KGQA benchmark Phase 2.",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Path to webqsp_subset.json (or compatible KGQA fixture).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which to write per-question JSON captures.",
    )
    parser.add_argument(
        "--langchain-version",
        type=str,
        required=True,
        help="Pinned LangChain version label recorded in each capture.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked answers to capture per question (default: 5).",
    )
    args = parser.parse_args()

    if _is_ci():
        sys.stderr.write(
            "WARNING: This is an operator-only script and should not be "
            "invoked inside CI. Aborting.\n"
        )
        return 1

    fixture = _load_fixture(args.fixture)
    args.output.mkdir(parents=True, exist_ok=True)
    chain = _build_langchain_pipeline(fixture["kg"], fixture.get("labels", {}))

    captured = 0
    for question in fixture["questions"]:
        qid = question["qid"]
        ranked = _rank_answers(chain, question["question"], args.top_k)
        payload: Dict[str, Any] = {
            "qid": qid,
            "ranked_answers": ranked,
            "langchain_version": args.langchain_version,
            "captured_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        out_path = args.output / f"{qid}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        captured += 1

    sys.stderr.write(
        f"Captured {captured} reference outputs to {args.output}\n"
        "Use LANGCHAIN_REF_FIXTURES=$PWD/<output> with cargo bench to enable Phase 2.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
