# OxiRS Chat - TODO

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Status: Production Ready (Experimental)

**oxirs-chat** provides AI-powered conversational interface for RDF data.

### Features

#### Natural Language Processing
- Improved NL to SPARQL translation
- Context-aware query generation
- Query refinement with multi-turn conversation support
- Intent recognition
- Entity extraction with type detection and confidence scoring
- Coreference resolution with mention tracking
- Sentiment analysis with emotion detection

#### RAG System
- Advanced retrieval strategies (quantum-enhanced, consciousness-aware)
- Vector search integration with embedding providers
- Context window management with sliding window compression
- Result ranking with ML-based relevance scoring
- Multi-modal support with cross-modal reasoning
- Schema-aware generation with schema introspection
- Knowledge graph reasoning with graph traversal
- Semantic caching with similarity-based retrieval

#### Chat Features
- Query suggestions with collaborative filtering
- Explanation generation with reasoning chains
- Data exploration guidance
- Result visualization (RichContentElement system)
- Export to multiple formats (JSON, CSV, RDF formats)
- Collaborative features (session management, real-time sync, cursor sharing)

#### Integration
- Multiple LLM providers (OpenAI, Anthropic, Local/Ollama)
- Custom prompts with Handlebars template system
- Fine-tuning support with training pipeline
- API integration with external services framework
- Webhook support with HMAC verification
- Plugin system with hook-based extensibility
- External knowledge bases (Wikipedia, PubMed connectors)

#### Advanced Features
- Advanced reasoning (chain-of-thought, tree-of-thoughts)
- Production deployment guides (Docker, Kubernetes, Cloud)
- Multi-language support (10 languages: EN, JA, ES, FR, DE, ZH, KO, PT, RU, AR)
- Voice interface (STT/TTS framework with multi-provider support)
- Real-time collaboration (WebSocket, shared sessions)
- Analytics dashboard backend

## Roadmap

### v0.1.0 - Released (January 7, 2026)
- ✅ NL-to-SPARQL, RAG system, multi-LLM providers, webhooks, plugin system

### v0.2.3 - Current Release (March 16, 2026)
- ✅ Additional LLM providers (Cohere, Groq, Mistral)
- ✅ Enhanced context management (context window with sliding compression)
- ✅ Advanced analytics dashboards
- ✅ Conversation history management
- ✅ Custom model fine-tuning workflows
- ✅ Session store for persistent chat sessions
- ✅ Dialogue manager for multi-turn conversation
- ✅ 1195 tests passing

### v0.3.0 - Planned (Q2 2026)
- [x] Web-based chat UI components — Tauri 2.x desktop app, desktop/oxirs-tauri/, chat.html (completed 2026-05-02)
- [x] Visual query builder UI — Tauri 2.x SVG-based SPARQL visual editor, desktop/ui/query_builder.html (completed 2026-05-02)
- [x] Enterprise SSO integration — OIDC + SAML-SP (completed 2026-05-01)
- [x] Long-term support guarantees (policy: docs/policies/lts.md) (completed 2026-05-17 via RFC-001)
- [x] Enterprise features (policy: docs/policies/enterprise.md, decomposed items listed therein) (completed 2026-05-17 via RFC-002)
- [x] Comprehensive documentation (planned 2026-05-01)
  - **Goal:** Top-level rustdoc + admin/user-facing tutorial covering session
    lifecycle, query routing, RAG retrieval path, LLM call paths. Two worked
    examples; doctests so docs can't rot.
  - **Design:**
    - Rewrite `src/lib.rs` `//!` with module overview + quickstart doctest.
    - `docs/tutorial.md` covering: spinning up a chat session against a
      local KG, switching providers, configuring retrieval depth, persisting
      session history.
    - `docs/admin.md` covering deployment topology, environment variables,
      logging, observability hooks.
    - 2 examples: `examples/kg_chat.rs` (chat over an RDF dataset),
      `examples/rag_chat.rs` (chat with vector retrieval over a doc corpus).
    - README quickstart + cross-link to docs.
  - **Files:** `src/lib.rs`, `README.md`, `docs/tutorial.md`, `docs/admin.md`,
    `examples/{kg_chat.rs,rag_chat.rs}`, `tests/examples_compile_test.rs`.
  - **Prerequisites:** existing session/store, query, retrieval modules.
  - **Tests:** lib.rs doctest passes; both examples compile via
    `cargo test --examples -p oxirs-chat`.
  - **Risk:** drift; mitigation: example-as-test.
- [x] Model marketplace integration — HF Hub + Ollama + local GGUF registry (completed 2026-05-02)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

---

*OxiRS Chat v0.2.3 - AI-powered conversational RDF interface*

## Proposed follow-ups

- [ ] Web chat UI components — OVERSIZED: requires a new frontend subcrate. Defer until UI strategy is decided (Tauri? Yew? Leptos? web-only?).
- [x] Visual query builder UI — implemented: SVG canvas with drag-and-drop triple patterns, filter nodes, SPARQL generation/validation, example loader. desktop/ui/query_builder.html + desktop/oxirs-tauri/src/query_builder.rs (completed 2026-05-02).
- [x] Enterprise SSO — OIDC + SAML-SP implemented in `src/sso/` (oidc.rs, saml_sp.rs, session.rs + SsoManagerConfig refactor).
- [x] Long-term support guarantees — RFC published at `docs/policies/lts.md`. (completed 2026-05-17 via RFC-001)
- [x] Enterprise features — decomposed in `docs/policies/enterprise.md`. (completed 2026-05-17 via RFC-002)
- [x] Model marketplace integration — implemented: HuggingFace Hub (offline catalogue), Ollama (local server), local GGUF filesystem registry.
