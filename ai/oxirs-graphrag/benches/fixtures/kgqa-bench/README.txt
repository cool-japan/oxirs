KGQA Benchmark Fixtures
========================

This directory holds vendored datasets used by the
`benches/langchain_kgqa.rs` benchmark harness.

Files
-----

webqsp_subset.json
  Hand-crafted subset of WebQSP-style KGQA questions over a small
  geography/people knowledge graph. The schema matches the WebQSP
  dataset (qid, question, topic_entity, answer_entities) so that
  operator-side LangChain reference captures share the same JSON shape.

  Schema:
    {
      "name": "...",
      "description": "...",
      "license": "Apache-2.0",
      "version": "1.0",
      "kg":     [{"s": "iri", "p": "iri", "o": "iri"}, ...],
      "labels": {"iri": "human label", ...},
      "questions": [
        {
          "qid":             "WebQSP-XXX",
          "question":        "natural language question",
          "topic_entity":    "iri",
          "predicate":       "iri",
          "answer_entities": ["iri", ...]
        },
        ...
      ]
    }

LangChain reference outputs
---------------------------

Phase 2 of the benchmark optionally compares oxirs-graphrag against a
pinned LangChain GraphRAG version. To enable the comparison, point the
`LANGCHAIN_REF_FIXTURES` environment variable at a directory containing
`*.json` files with the following shape:

    {
      "qid":  "WebQSP-XXX",
      "ranked_answers": ["iri", "iri", ...]
    }

These reference outputs are NEVER produced or executed in CI. They are
captured offline by an operator running a pinned LangChain version via
`benches/scripts/capture_langchain_reference.py`.
