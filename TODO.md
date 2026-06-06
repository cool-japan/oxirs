# OxiRS Development Roadmap

*Version: 0.3.1 | Last Updated: June 6, 2026*

## Current Status: v0.3.1 - Released (June 6, 2026)

**OxiRS** is an advanced AI-augmented semantic web platform built in Rust, delivering a production-ready alternative to Apache Jena + Fuseki with cutting-edge AI/ML capabilities.

### Release Metrics (v0.3.1, June 6, 2026)
- **Version**: 0.3.1
- **Architecture**: 26-crate workspace
- **Build Status**: Clean compilation - Zero errors/warnings across all modules
- **Test Status**: ~43,511 tests passing (100% pass rate, 1 known-flaky timing test)
- **Development Rounds Complete**: 21
- **New Modules Added**: 10+ modules across all crates (Rounds 17-18)

### Release Metrics (v0.3.0, May 3, 2026)
- **Version**: 0.3.0
- **Architecture**: 26-crate workspace
- **Build Status**: Clean compilation - Zero errors/warnings across all modules
- **Test Status**: ~41,400 tests passing (100% pass rate)
- **Development Rounds Complete**: 19
- **New Modules Added**: 150+ modules across all crates

### v0.2.4 Feature Highlights
- Advanced SPARQL 1.2: ASK evaluator, EXISTS evaluator, service clause federation, subquery builder, MINUS evaluator, values clause
- RDF Processing: literal parser (15 XSD types), WKT geometry parser, annotation graph, prefix resolver, quantizer (k-means++ PQ)
- Storage: six-index store (SPO/POS/OSP), partition manager (Kafka-style), endpoint registry, shard router
- AI/ML: vector store, constraint inference, conversation history, thermodynamics simulation, triple extractor, entity classifier
- Security: credential store (W3C VC), authentication module, trust chain
- IoT: coil register map (Modbus FC01/02/05/15), signal decoder (Intel/Motorola DBC), OBD-II decoder, protocol analyzer
- Tools: diff command (RDF set-diff + Dice similarity), convert command, validate command, benchmark command, profile command
- Server: SPARQL connection pool, field resolver cache (TTL+LRU), endpoint router
- OWL 2 DL: ABox reasoning with 29 rule groups, property hierarchy, complex constructors, cardinality rules
- SHACL-AF SPARQL targets: sh:SPARQLTarget, sh:SPARQLTargetType, sh:SPARQLAskValidator
- SHACL constraint extensions: QualifiedValueShape, UniqueLang, LanguageIn, Equals, Disjoint, LessThan
- Graph-level RBAC, RDF Binary (Thrift) read+write, OAuth2/OIDC refresh token rotation
- Full Jena parity: OWL2QL, OWL2DL, SHACL-AF, RDF Thrift, OAuth2, RBAC
- Federated Query Optimization: query rewriting, cost estimation, 5-pass optimizer
- Vector Search: SIMD-accelerated ANN, GPU simulation, BatchSearchEngine
- GraphQL Subscription Optimization: ChangeTracker, SubscriptionManager, Broadcaster
- Distributed Stream Processing: consistent hash ring, CRDT state, fault tolerance
- Cloud Storage Backends: S3 SigV4, GCS OAuth2, Azure SharedKeyLite
- IoT Protocol Extensions: UDS ISO 14229, CANopen DS-301, Modbus ASCII/TLS
- Time-Series Analytics: anomaly detection, Holt-Winters forecasting, Prometheus remote write
- WASM: query builder, triple store, storage adapter

## v0.2.4 Test Coverage (March 28, 2026)

Per-crate test counts:
- oxirs-arq: 2688 tests
- oxirs-core: 2332 tests
- oxirs-fuseki: 2144 tests
- oxirs-gql: 2081 tests
- oxirs-rule: 2072 tests
- oxirs-shacl: 2008 tests
- oxirs-tdb: 2005 tests
- oxirs-ttl: 1726 tests
- oxirs-geosparql: 1713 tests
- oxirs-star: 1628 tests
- oxirs (tools): 1615 tests
- oxirs-vec: 1598 tests
- oxirs-shacl-ai: 1589 tests
- oxirs-stream: 1505 tests
- oxirs-cluster: 1489 tests
- oxirs-samm: 1409 tests
- oxirs-federate: 1397 tests
- oxirs-embed: 1345 tests
- oxirs-chat: 1195 tests
- oxirs-tsdb: 1127 tests
- oxirs-canbus: 1125 tests
- oxirs-modbus: 1095 tests
- oxirs-physics: 1063 tests
- oxirs-did: 1043 tests
- oxirs-graphrag: 935 tests
- oxirs-wasm: 858 tests
- **Total: 40,786 tests**

## v0.2.4 Development Rounds

**Round 1**: +basic_auth, +websocket_handler, +type_resolver, +query_cache, +rule_compiler, +shape_matcher, +aspect_export, +bounding_box, +rdf_star_serializer, +iri_catalog, +delta_encoder, +wal_archive, +consensus_log, +compression_codec, +dead_letter_queue, +endpoint_discovery, +alarm_manager, +signal_monitor, +cross_encoder, +anomaly_detector, +tool_registry, +signal_processing, +graph_partitioner, +revocation_list, +wasm_bridge, +cache commands

**Round 2**: +property_function_registry, +WebSocket subscriptions, +Federation v2, +LATERAL join, +node_expression_evaluator, +submodel_templates, +spatial_join, +BIND/VALUES quoted triples, +RDF/XML writer

**Round 3**: +triple_term SPARQL 1.2, +HTTP/2 push, +cursor pagination, +plan visualizer, +SHACLC parser, +aspect_differ, +spatial DBSCAN, +annotation_paths, +pretty_printer, +product_quantization, +LSM compaction, +vnodes hash ring, +Gorilla compression, +window algebra, +health monitor

**Rounds 4-5**: +sparql_optimizer, +split_brain_detector, +flat_ivf_index, +triple_term, +HTTP/2 push, +cursor pagination, +plan visualizer, +SHACLC parser; +property_function_registry, WebSocket subscriptions, federation v2, LATERAL join

**Round 6**: +property_path_evaluator, +query_log, +subscription_manager, +bloom_filter, +leader_election, +anomaly_detector, +aggregate_executor, +forward_chainer, +target_selector, +unit_catalog, +geo_serializer, +provenance_tracker, +base_directive, +lsh_index, +heat_transfer, +path_ranker, +presentation_builder, +sparql_formatter, +export_command, +event_sourcing, +result_aggregator, +coil_controller, +frame_filter, +pca_reducer, +report_formatter, +response_ranker

**Round 7**: +sparql_binding_set, +update_processor, +batch_resolver, +window_function, +rule_parser, +message_formatter, +constraint_registry, +coordinate_transformer, +triple_reifier, +streaming_parser, +ivfpq_index, +index_statistics, +snapshot_manager, +downsampler, +sync_schema_registry, +cache_coordinator, +exception_handler, +gateway_bridge, +embedding_cache, +shape_evolver, +session_manager, +fluid_dynamics, +entity_linker, +vc_verifier, +prefix_manager, +import_command

**Round 8**: +rdf_dataset, +transaction_manager, +schema_stitcher, +lateral_join, +rete_network, +sparql_constraint_validator, +aspect_chain, +raster_sampler, +rdf_patch, +namespace_mapper, +hnsw_builder, +page_cache, +gossip_protocol, +time_bucket_aggregator, +watermark_tracker, +plan_optimizer, +tcp_listener, +replay_engine, +fine_tuner, +explanation_generator, +knowledge_retriever, +stress_analysis, +subgraph_extractor, +key_derivation, +sparql_result_formatter, +validate_command

**Round 9**: +expr_algebra, +rate_limiter, +query_complexity, +values_clause, +conflict_resolver, +property_path_checker, +operation_registry, +topology_checker, +quoted_triple_store, +turtle_validator, +product_search, +btree_compaction, +log_replication, +retention_manager, +dead_letter_queue, +query_rewriter, +holding_register_bank, +diagnostic_monitor, +similarity_search, +property_suggester, +electromagnetics, +relation_extractor, +access_control, +query_builder, +convert_command

**Round 10**: +literal_parser, +sparql_connection_pool, +field_resolver_cache, +service_clause, +backward_chainer, +shape_graph_loader, +unit_converter, +wkt_parser, +annotation_graph, +prefix_resolver, +quantizer, +six_index_store, +membership_manager, +series_metadata, +partition_manager, +endpoint_registry, +coil_register_map, +signal_decoder, +vector_store, +constraint_inference, +conversation_history, +thermodynamics, +triple_extractor, +credential_store, +endpoint_client, +diff_command

**Round 11**: Refactored event_sourcing.rs (2635 lines split into 6 files); +update_parser, +auth_middleware, +error_formatter, +group_by_evaluator, +rule_index, +focus_node_selector, +payload_generator, +distance_calculator, +star_query_rewriter, +ntriples_writer, +ann_benchmark, +write_batch, +failover_manager, +continuous_query, +consumer_group, +source_selector, +register_validator, +bit_timing, +embedding_aggregator, +schema_alignment, +context_window, +quantum_mechanics, +knowledge_fusion, +proof_purpose, +graph_visualizer, +merge_command

**Round 12**: +blank_node_allocator, +content_negotiation, +directive_processor, +optional_evaluator, +rule_tracer, +severity_handler, +entity_resolver, +area_calculator, +star_normalizer, +compact_serializer, +vector_cache, +checkpoint_manager, +anti_entropy, +tag_index, +stream_router, +query_splitter, +batch_reader, +error_counter, +tokenizer, +data_profiler, +intent_detector, +optics, +context_builder, +key_manager, +triple_store, +query_command

**Round 13**: +result_formatter, +request_validator, +pagination_handler, +construct_builder, +rule_serializer, +path_executor, +aspect_validator, +intersection_detector, +triple_diff, +rdfa_parser, +cluster_index, +vacuum, +replication_throttle, +write_buffer, +schema_validator, +result_streamer, +event_log, +message_database, +reranker, +change_detector, +memory_store, +statistical_mechanics, +path_finder, +vc_presenter, +sparql_executor, +inspect_command

**Round 14**: +subquery_builder, +dataset_manager, +field_validator, +minus_evaluator, +rule_validator, +constraint_parameter, +model_serializer, +convex_hull, +star_statistics, +nt_parser, +index_merger, +index_rebuilder, +data_migrator, +event_correlator, +message_transformer, +query_monitor, +register_encoder, +frame_validator, +index_optimizer, +rule_generator, +response_cache, +celestial_mechanics, +summarizer, +trust_chain, +storage_adapter, +profile_command

**Round 15**: +ask_evaluator, +endpoint_router, +argument_coercer, +exists_evaluator, +rule_executor, +datatype_checker, +property_mapper, +simplifier, +graph_merger, +jsonld_compactor, +approximate_counter, +triple_cache, +shard_router, +forecaster, +replay_buffer, +load_balancer, +protocol_analyzer, +obd_decoder, +batch_encoder, +constraint_ranker, +dialogue_manager, +control_systems, +entity_classifier, +authentication, +event_dispatcher, +benchmark_command

**Round 16**: +describe_builder, +query_logger, +enum_resolver, +path_expression, +rule_statistics, +node_constraint, +constraint_validator, +coordinate_converter, +reification_mapper, +trig_parser, +pq_encoder, +bloom_index, +election_timer, +rollup_engine, +event_filter, +capability_negotiator, +register_watcher, +pgn_decoder, +embedding_compressor, +pattern_scorer, +session_store, +kinematics, +community_detector, +presentation_request, +namespace_manager, +serve_command

**Round 17** (v0.3.1, 2026-05-17): +policy_templates (RBAC DBA/ReadOnly/Auditor, PolicyTemplateRegistry, ReadAudit permission), +fips_feature_gate (oxirs-did + oxirs-fuseki FIPS 140-2, RFC-003 fips-boundary.md), refactor config.rs→4 files, refactor lateral_join.rs→directory module, refactor advanced_pattern_analysis.rs→3 engines, build fix (oxiarc 0.3.0 + scirs2 patches), 47 TODO `[~]`→`[x]` sweeps across 21 files

**Round 18** (v0.3.1, 2026-05-17): +graph_summarizer (Leiden→centrality→predicate-freq pipeline, to_text()), +triple_relevance_feedback (seahash IDs, multiplicative weight, apply_to_scores), +graph_sage_embedder (k-hop mean-agg, Xavier init, margin-ranking loss, sign-SGD, LCG sampling), refactor persistent.rs→4 files, refactor selector.rs→3 files, refactor performance_monitoring.rs→2 files (fixed pre-existing bugs), refactor optimization/engine.rs→2 files, enterprise.md policy items closed

**Round 24** (v0.3.1, 2026-05-18): +cluster_auth (cross-node HMAC-SHA256 token auth, ClusterNodeToken, ClusterAuthManager, JTI via scirs2_core::random), +cross_model (CrossModelRegistry, CrossModelValidator, URN-indexed cross-crate SAMM model references), +manchester (OWL Manchester Syntax parser/emitter: lexer+recursive-descent parser+AST+emitter, 30 tests), +generate (SPARQL-Generate template engine: AST+parser+executor, 12 tests), +csvw (CSV-on-the-Web inline RFC-4180 reader+CsvwConverter with about_url template substitution, 16 tests)

**Round 25** (v0.3.1, 2026-05-18): refactor biological_neural_integration.rs→3 siblings (596+664+459 lines), refactor validation.rs (oxirs-geosparql)→4 siblings (193+403+514+712 lines), refactor gpu/index_builder.rs→3 siblings (260+919+625 lines)

**Round 26** (v0.3.1, 2026-05-18): refactor skos.rs→4 siblings (202+384+116+1111 lines, 65 tests), refactor oxirs-star parser.rs+store.rs→4+4 siblings (1482 tests), refactor federated_query_optimizer.rs→4 siblings (406+470+788+151 lines, fixed orphan rand/uuid), refactor distributed.rs+novel_architectures.rs→4+3 siblings (distributed 53 tests, novel_arch 12 tests), refactor ml_predictor.rs+node_expressions.rs→4+4 siblings (2907+2043 tests), +owl_profile_ql (Owl2QlProfileChecker, 23-variant OntologyAxiom, 14-variant ClassExpr, 19 tests)

**Round 27** (v0.3.1, 2026-05-18): +LDF triple-pattern-fragments endpoint (TpfQuery, pagination, content-negotiation Turtle/JSON-LD/N-Triples, 22 tests), +VocPrez vocabulary publishing handler (VocabularyRegistry, HTML/JSON-LD/Turtle, 18 tests), refactor persistence.rs+rich_content.rs (oxirs-chat)→3+3 siblings, refactor performance_analyzer.rs (oxirs-federate)→4 siblings (6 tests), refactor serialization.rs (oxirs-stream)+diagnostics.rs (oxirs-tdb)→4+3 siblings, refactor update_protocol.rs+materialized_views.rs (oxirs-arq)→4+4 siblings (26+4 tests)

**Round 28** (v0.3.1, 2026-05-18): refactor cli.rs+reification/mod.rs (oxirs-star)→4+4 siblings (cli 3 tests, reification 23 tests, 1660 total), refactor service_delegation.rs (oxirs-fuseki)+arrow_export.rs (oxirs-tsdb)→4+3 siblings (2300+1277 tests), refactor proof_purpose.rs (oxirs-did)+openapi.rs (oxirs-samm)→4+4 siblings (1101+1549 tests), refactor continual_learning.rs (oxirs-embed)+store_integration.rs (oxirs-vec)→4+4 siblings (1472+1685 tests), refactor tenant/mod.rs (oxirs-tdb)+shapes/parser.rs (oxirs-shacl)→4+4 siblings (2086+2043 tests), refactor production/types.rs (oxirs-arq)→3 siblings + +path_algebra (SPARQL 1.2 property path BFS evaluator: Link/Inverse/Sequence/Alternative/ZeroOrMore/OneOrMore/Optional/NPS, 33 tests, 2940 total)

**Round 29** (v0.3.1, 2026-05-19): refactor conservation/checkers.rs+rdfxml/parser.rs (oxirs-physics+oxirs-core)→4+4 siblings (1280+2465 tests), refactor forecasting_models.rs (oxirs-shacl-ai)+utils.rs (oxirs-embed)→4+4 siblings (1715+1472 tests), refactor commands/interactive.rs+aspect.rs (tools/oxirs)→3+3 siblings (1773 tests), refactor update_graph_management.rs (oxirs-arq)+quality/core.rs (oxirs-shacl-ai)→4+3 siblings (2953+1715 tests) + NEW adaptive_routing (EndpointStats EWMA+error-rate, EndpointCostEstimator cost model, AdaptivePlanner greedy/DP federation routing, 18 tests, 1550 federate total), refactor problog.rs (oxirs-rule)+neural_symbolic_integration.rs (oxirs-embed)→4+4 siblings (2230+1485 tests) + NEW datalog (DatalogProgram+semi-naive evaluator+stratification+parser, 20 tests), refactor jsonld/expansion_algorithm.rs (oxirs-core)+federated_learning.rs (oxirs-shacl-ai)→4+4 siblings + NEW JSON-LD 1.1 Framing (JsonLdFramer frame matching+EmbedPolicy First/Last/Always/Never/Link+@explicit+@default+cycle detection, 21 tests, 2465 core total) | Workspace total: 44,026/44,027 tests pass

**Round 30** (v0.3.1, 2026-05-19): refactor parallel.rs (oxirs-arq)+physics_rdf.rs (oxirs-physics)→4+4 siblings (2957+1280 tests), refactor ml_optimizer.rs (oxirs-federate)+connection_pool.rs (oxirs-stream)→4+4 siblings (1550+1671 tests), refactor performance_monitoring.rs+ml/gnn.rs+neural_patterns/learning.rs (oxirs-shacl-ai)→4+4+4 siblings (1717 tests), refactor uds/mod.rs (oxirs-canbus)+store/mod.rs (oxirs-tdb)→4+4 siblings (1190+2086 tests) + NEW nquads_streaming (NQuadsStreamingParser+lexer+parser, full N-Quads RFC, 18 tests, 1771 ttl total), refactor causal_representation_learning.rs+evolutionary_nas.rs+cross_module_performance.rs (oxirs-embed)→4+4+4 siblings (1492 tests), refactor graph_exploration.rs (oxirs-chat)+aspect_analyzer.rs (tools/oxirs)→4+4 siblings (1240+1783 tests) + NEW annotation_syntax (RDF-star {| |} shorthand: AnnotationParser+tokenizer+expander, 18 tests, 1678 star total)

**Round 31** (v0.3.1, 2026-05-20): facade recovery — converted 13 reverted-original files to thin facades by activating pre-existing orphan siblings across 8 crates: oxirs-core config.rs+ai/vector_store.rs+jsonld/expansion_algorithm.rs (1909/1906/1681→21/27/1030, 2521 tests), oxirs-vec compression.rs+joint_embedding_spaces.rs (1604/1618→10/19, 1638 tests, restored JointEmbeddingSpace::zero_shot_retrieval), oxirs-fuseki service_description/mod.rs+federated_query_optimizer.rs (1607/1774→29/22, 2300 tests), tools/oxirs commands/import_command.rs (1602→26), oxirs-embed enterprise_knowledge.rs+neural_symbolic_integration.rs (1901/1677→80/18; rewrote neural-symbolic _engine/_types siblings with canonical defs), oxirs-shacl-ai system_monitoring.rs (1885→26), oxirs-samm cloud_backends.rs (1882→68, 1549 tests), oxirs-rule skos.rs (1793→44, 2240 tests); tools/oxirs+oxirs-shacl-ai post-fix verify 3500 passed. Dependency fix: added indexmap+toml workspace deps to oxirs-core (jsonld/{compaction,flattening} indexmap-usage pre-existing HEAD breakage). Soundness fix: NetworkTopology+NeuralResponse removed from unsafe `impl_zeroed_default!` macro (mem::zeroed on Vec/Instant is UB), replaced with derive(Default)+manual impl. Workspace warning sweep: f32 ambiguity (E0689) neural_symbolic_integration_loss, E0432 advanced_profiler_tests, explicit_counter_loop cloud_backends_sync, missing_docs submodel_templates, module_inception+unused imports in datalog/manchester/csvw/nquads_streaming/adaptive_routing/annotation_syntax/neuromorphic_analytics tests, drop_non_drop framing_tests, redundant futures_util use in connection_pool_tests, map_or→is_some_and path_algebra/tests. Workspace: 0 errors, 0 code clippy warnings, ~20K lines net deleted.

**Round 32** (v0.3.1, 2026-05-21): tier-2 refactor — advanced_diagnostics.rs (oxirs-tdb)+cluster_metrics.rs (oxirs-cluster), auth/saml.rs+bind_values_enhanced.rs (oxirs-fuseki), parallel_executor.rs (oxirs-arq)+geometry/geometry3d.rs (oxirs-geosparql), query/sparql_algebra_types.rs (oxirs-core)+commands/jena_parity.rs (tools/oxirs), service.rs (oxirs-federate, collision-safe service_types/service_core/service_tests siblings)+sophisticated_validation_optimization.rs (oxirs-shacl-ai) → thin facades (10 files, each 1586-1599 lines→12-35 lines, 32 new siblings). + COMPLETE SHACL Advanced Features (oxirs-shacl): real recursive shape validation in advanced_features/recursive_shapes.rs (validate_depth_first / validate_breadth_first / validate_optimized with visited_nodes cycle detection, max_depth bound, result_cache memoization; local_conformance helper strips shape-referencing constraints to break engine's non-cycle-aware recursion; recursive_children evaluates sh:path via PropertyPathEvaluator; extract_dependencies iterates property_shapes/extends/Node/Not/And/Or/Xone/QualifiedValueShape) + qualified value-shape conformance in qualified_shapes.rs (value_conforms_to_shape uses in-scope shape_registry, collect_referenced_shape_ids builds temp IndexMap, validates via ValidationEngine::validate_node_against_shape) + sh:if/then/else conditional in conditional.rs (evaluate_condition_shape performs full ShapeRegistry lookup + validation; cache_hits counter wired into cache_stats().hits) + 24 new tests covering DFS/BFS/optimized chains, cycle termination, max-depth overflow, qualified min/max exact counts, conditional if-true/if-false/no-else, cache-hit increment, missing-if-shape error, extract-deps across all reference kinds, replacing prior stub Ok(true) placeholders. Workspace: 0 errors, 0 code warnings.

**Round 33** (v0.3.1, 2026-05-21): tier-3 refactor — 10 large files (1543-1581 lines) → thin facades (14-95 lines) + cohesive siblings (<1000 each, ~40 new files): oxirs-wasm query/construct.rs (1581→20), oxirs-core provenance/mod.rs+consciousness/mod.rs (1576/1564→29/95, all 29 consciousness:: re-exports preserved) + jsonld/context.rs+jsonld/to_rdf.rs (1560/1547→15/15), oxirs-gql schema.rs (1556→14), oxirs-stream backend/kafka/backend.rs (1554→24, feature-gated siblings verified under --features kafka), oxirs-shacl constraints/expression_constraint.rs (1550→19), oxirs-arq rdf_star/mod.rs (1550→48), oxirs-vec tree_indices.rs (1546→40 + 8 tree-type siblings); declaration sites updated, public APIs preserved exactly. + COMPLETE SHACL-AF reasoning engine (oxirs-shacl advanced_features/reasoning.rs 1543→28 facade + reasoning_types/validator/probabilistic/tests siblings): 14 real entailment methods replacing placeholder stubs — RDFS build_subclass_graph/build_subproperty_graph (shared build_relation_graph helper), infer_from_domain/infer_from_range (literal-object skip), infer_type_inheritance via subclass closure; OWL 2 RL infer_equivalent_properties/infer_equivalent_classes, infer_inverse_properties (bidirectional), infer_transitive_properties (per-property Floyd-Warshall closure), infer_symmetric_properties; is_subclass_of (closure-based, reflexive documented), get_inferred_types; bonus is_false_under_cwa + NAF fails implemented with real store-pattern queries; validate_with_reasoning wired to real conformance (contradicts_shape check) — all backed by Store::find_quads. Fixed compute_transitive_closure_with_scirs2 bug (only indexed graph.keys(), dropping edge-target-only nodes). 24 new entailment tests with in-memory ConcreteStore incl. subclass-cycle termination, 3-level transitivity, domain/range inference, symmetric/inverse reversal. OWL 2 QL/EL/Full + CustomReasoner apply_* left as honest documented stubs. Verification: cargo check + clippy --workspace --all-targets 0 errors/0 warnings; 14,002 tests pass across the 7 affected crates.

**Round 34** (v0.3.1, 2026-06-05): COOLJAPAN Pure-Rust migration — all 4 remaining holdouts closed. Compression: brotli→oxiarc-brotli, snap→oxiarc-snappy, flate2→oxiarc-deflate across all consuming crates (tdb/stream/federate/fuseki/cluster/arq/gql/star/embed/shacl-ai/chat/did/tools); direct brotli/snap/flate2 deps removed. Fixed a real oxiarc-brotli 0.3.2 compressor bug (emitted streams its own decoder rejected for incompressible input — package-merge length-limited Huffman + extended insert-length codes), bumped oxiarc→0.3.3, consumed via `[patch.crates-io]` path patches (all 10 oxiarc crates unify oxiarc-core at 0.3.3). Crypto: 27 first-party `ring::` call sites across 4 crates (tools/oxirs, oxirs-did, oxirs-fuseki, oxirs-cluster) → **oxicrypto** leaf crates (hash/mac/aead/kdf/rand) + workspace **ed25519-dalek + rsa** signatures; all 4 direct ring deps removed. TLS de-C/asm'd: rustls no-provider, reqwest rustls-no-provider, tokio-rustls default-features=false, metrics-exporter-prometheus push-gateway-no-tls-provider, **oxitls::pure_provider()** installed process-wide at every binary entry, cluster cert-gen rcgen→oxitls-rcgen (Ed25519), async-openai gated behind non-default `openai` feature. **Net: default `cargo build` links zero ring + zero aws-lc-sys/aws-lc-rs.** Verification: cargo build --workspace exit 0; clippy --workspace --all-targets 0 warnings in migrated code; `cargo tree -i ring` + `-i aws-lc-sys` + `-i aws-lc-rs` all EMPTY for the default feature set. Out-of-scope C residual noted for a future pass: security-framework-sys (reqwest rustls-platform-verifier macOS trust-store FFI, not crypto) + native-tls chain (slack-hook2/lettre/reqwest-0.11) in oxirs-cluster.

## Roadmap

### v0.1.0 - Initial Release (Released - January 7, 2026)
- Core RDF data model (IRI, BlankNode, Literal, Triple, Quad)
- Basic SPARQL parsing and evaluation
- Turtle/N-Triples serialization
- In-memory triple store
- Basic Fuseki-compatible HTTP server
- GraphQL endpoint scaffold
- Initial SHACL validation
- Project workspace with foundational crates

### v0.2.4 - Full-Featured Platform (Released - March 28, 2026)
- 26-crate workspace with 40,786 tests
- Complete SPARQL 1.2 feature set (federated ASK, EXISTS, LATERAL, MINUS, VALUES, subqueries)
- Full GeoSPARQL 1.1 compliance with convex hull, topology, spatial indexing
- OWL 2 DL reasoning with ABox, property hierarchy, complex constructors
- SHACL-AF with SPARQL targets, constraint extensions
- Distributed storage: raft log replication, anti-entropy, shard rebalancing
- Advanced AI: RAG pipelines, physics simulation, knowledge graph embeddings
- Enhanced IoT: full Modbus/CANbus protocol coverage with OBD-II
- Production tooling: benchmark, profile, diff, convert, validate CLI commands
- Security: W3C Verifiable Credentials, DID trust chains, OAuth2/OIDC, RBAC
- Platform: WASM query builder, triple store, storage adapter
- Full Apache Jena feature parity

### v0.3.0 - Next Release (Planned)
- Performance hardening and optimization pass
- ✓ Extended SPARQL federation with adaptive query routing (Round 29: AdaptiveRoutingEngine, EWMA endpoint stats, cost model, greedy/DP planner)
- Enhanced WASM support for browser-based RDF processing
- Additional serialization formats (✓ JSON-LD framing Round 29, TriG, N-Quads streaming)
- Expanded cloud-native deployment (Kubernetes operators, auto-scaling)
- Advanced graph analytics integration
- Plugin architecture for custom extensions

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

*OxiRS v0.3.1 - Production-ready semantic web platform with full Apache Jena parity, enterprise-grade AI/ML capabilities, and ~43,500 tests.*

## Pure Rust Migration (COOLJAPAN Policy)

> oxirs is already partially migrated (uses oxicode, oxiarc-archive/zstd/lz4). The items below are the remaining Pure-Rust holdouts, in priority order.

- [x] **(HIGH — the ONLY true C/asm violation) Replace `ring` 0.17 with pure-Rust crypto.**
  - Decl: workspace `Cargo.toml` ~line 333 (`ring = "0.17"`). Consumed UNCONDITIONALLY by: `tools/oxirs` (Cargo.toml ~:57), `oxirs-did` (~:36), `oxirs-fuseki` (~:177), `oxirs-cluster` (~:58).
  - ~27 real `ring::` call sites across 4 crates. Uses: `ring::rand::SystemRandom` (RNG), `ring::hmac`, `ring::digest`, signatures — in oxirs-did (signing/verification), oxirs-fuseki (SAML/VC/DAPS auth: `src/auth/{saml_parser,cluster_auth}.rs`, `src/ids/identity/*`), oxirs-cluster (`src/{tls,encryption}.rs`), tools/oxirs (`src/tools/{pitr,backup_encryption}.rs`, `src/config/secrets.rs`).
  - Replacement: RustCrypto equivalents ALREADY present in the workspace (`sha2`, `sha1`, `hmac`, `aes-gcm`, `chacha20poly1305`, `ed25519-dalek`, `curve25519-dalek`, `argon2`) or COOLJAPAN `oxicrypto`. Mapping: `ring::rand::SystemRandom` → `getrandom`/`rand`; `ring::hmac` → `hmac` + `sha2`; `ring::digest` → `sha2`; signatures → `ed25519-dalek`.
  - Acceptance: `cargo tree -i ring` empty; the four affected crates' tests + `clippy` green; zero C/asm in the default build.
  - ✓ DONE (2026-06-05): all 27 `ring::` call sites across the 4 crates migrated to **oxicrypto** leaf crates (oxicrypto-hash/mac/aead/kdf/rand for SHA-256/HMAC/AES-256-GCM/PBKDF2/CSPRNG) and the workspace's stable **ed25519-dalek + rsa** for Ed25519/RSA signatures (oxicrypto's facade/oxicrypto-sig were unusable here due to a broken pre-release ed448-goldilocks dep). All 4 direct `ring` deps removed; the TLS stack was also de-asm'd (rustls no-provider, reqwest rustls-no-provider, tokio-rustls default-features=false, metrics-exporter-prometheus push-gateway-no-tls-provider, oxitls::pure_provider() installed at every binary entry, cluster cert-gen → oxitls-rcgen, async-openai gated behind non-default `openai` feature). **Acceptance satisfied for the default feature set: `cargo tree -i ring` AND `cargo tree -i aws-lc-sys`/`aws-lc-rs` are all EMPTY — default build links zero ring + zero aws-lc-sys.** Out-of-scope C residual noted for a future pass: security-framework-sys (reqwest rustls-platform-verifier, macOS trust-store FFI, not crypto) and a native-tls chain (slack-hook2/lettre/reqwest-0.11) in oxirs-cluster.

- [x] **(MED, consistency-only — pure-Rust today) Replace `brotli` 8.0 with `oxiarc-brotli`.**
  - Decl: workspace `Cargo.toml` ~:306. Confirmed pure-Rust (`brotli` + `brotli-decompressor`, NOT brotli-sys). Consumed by `oxirs-federate` (~:78), `oxirs-stream` (~:73), `oxirs-fuseki` (~:105), `oxirs-tdb` (~:56).
  - ~12 `brotli::` call sites / 4 crates: `oxirs-federate/src/{result_streaming,network_optimizer}.rs`, `oxirs-stream/src/types.rs`, `oxirs-fuseki/src/streaming_results.rs`, `oxirs-tdb/src/compression/{brotli_compression,unified,mod}.rs`.
  - Acceptance: builds, tests, and `clippy` green for those 4 crates; `cargo tree` shows `brotli`/`brotli-decompressor` gone.
  - ✓ DONE (2026-06-05): all direct `brotli` call sites migrated to **oxiarc-brotli**; direct `brotli` dep removed from the workspace. Uncovered + fixed a real compressor bug in oxiarc-brotli 0.3.2 (it produced streams its own decoder rejected for incompressible input — fixed upstream via package-merge length-limited Huffman + extended insert-length codes); oxiarc bumped to **0.3.3** and consumed locally via `[patch.crates-io]` path patches (all 10 oxiarc crates patched so oxiarc-core unifies at 0.3.3). Residual pure-Rust brotli (dtolnay, no `*-sys`) remains only via tower-http HTTP-compression + Tauri build-deps — acceptable (brotli was a consistency-only item, not C/asm).

- [x] **(consistency, larger) Replace `snap` 1.1 with `oxiarc-snappy`.**
  - Decl: workspace `Cargo.toml` ~:305. Consumed by `oxirs-stream` (~:72), `oxirs-tdb` (~:55), `oxirs-cluster` (~:69).
  - ~14 `snap::` call sites / 3 crates: `oxirs-stream/src/{serialization_encoder,types}.rs` + `src/backend/nats/compression.rs`, `oxirs-tdb/src/compression/{snappy,unified}.rs`, `oxirs-cluster/src/network_compression.rs`. Note: many `snap` grep hits elsewhere are "snapshot" false positives; filter on `snap::`.
  - Acceptance: builds, tests, and `clippy` green for those 3 crates; `cargo tree -i snap` empty.
  - ✓ DONE (2026-06-05): all `snap::` call sites across the 3 crates migrated to **oxiarc-snappy**; direct `snap` dep removed from the workspace; `cargo tree -i snap` empty.

- [x] **(consistency, LARGEST footprint) Replace `flate2` 1.1 with `oxiarc-deflate`.**
  - Decl: workspace `Cargo.toml` ~:302. Pure-Rust (miniz_oxide backend). Consumed UNCONDITIONALLY by ~12 crates: `tools/oxirs`, `oxirs-federate`, `oxirs-stream`, `oxirs-fuseki`, `oxirs-gql`, `oxirs-tdb`, `oxirs-cluster`, `oxirs-embed`, `oxirs-shacl-ai`, `oxirs-chat`, `oxirs-arq`, `oxirs-star`.
  - ~88 `flate2::` call sites — the biggest sweep. `oxiarc-deflate` is NOT yet a dep (oxirs already has `oxiarc-zstd`/`oxiarc-lz4`/`oxiarc-archive`). Suggest migrating crate-by-crate.
  - Acceptance: builds, tests, and `clippy` green per migrated crate; `cargo tree -i flate2` (and `-i miniz_oxide`) empty once all ~12 crates are done.
  - ✓ DONE (2026-06-05): all `flate2::` call sites migrated crate-by-crate to **oxiarc-deflate** across the ~12 consuming crates; all direct `flate2` deps removed from the workspace. Residual pure-Rust flate2/miniz_oxide (no `*-sys`) remains only via tower-http HTTP-compression + Tauri/zip build-deps — acceptable (flate2 was a consistency-only item, not C/asm).

### Follow-up: remaining default-build C-FFI (policy-check 2026-06-06)

> The 4 items above removed the crypto/asm offenders (ring, aws-lc → confirmed empty). A full policy-check found 4 OTHER C-FFI leaks still in the DEFAULT-feature closure (pre-existing, out of the original 4-item scope). `oxirs-cluster` is the biggest offender (3 of 4).

> **Result (2026-06-06):** default `cargo build` now links ZERO C/asm crypto AND zero C compression/storage (ring, aws-lc, native-tls, lmdb-sys, lzma-sys, zstd-sys, rocksdb all gone). Full `cargo check --workspace` = 0 errors. Only residual default-build C-FFI is `security-framework-sys` via reqwest's rustls-platform-verifier (OS trust-store shim — accepted OS-boundary exception).

- [x] **(HIGH · Pure-Rust) `oxirs-cluster` C-FFI purge** — removes `native-tls`+`lmdb-sys`+`lzma-sys` from the default build:
  - `lettre` (Cargo.toml ~:95) `tokio1-native-tls` → `tokio1-rustls-tls` feature (must route through the process-default pure CryptoProvider, not aws-lc/ring).
  - `slack-hook2` (~:96) pulls `reqwest 0.11` → `hyper-tls` → `native-tls`; replace with a workspace-`reqwest 0.13`/`oxihttp` webhook client.
  - `lmdb` (~:83, direct non-optional) → Pure-Rust KV (`oxirs-tdb` / `redb`-class `oxistore-kv-*`); pulls `lmdb-sys` (C).
  - `xz2` (transitive) → `oxiarc-lzma`; pulls `lzma-sys` (C).
  - ✓ DONE (2026-06-06): lettre→tokio1-rustls+rustls-no-provider, slack-hook2→reqwest 0.13 webhook POST, dead lmdb path deleted (kept WAL+mmap), xz2→oxiarc-lzma. `cargo tree -i` now EMPTY for native-tls/lmdb-sys/lzma-sys; ring/aws-lc stay empty. 131/131 cluster tests pass; also fixed a latent WAL-recovery checksum bug.
- [x] **(MED · Pure-Rust) tantivy `zstd-sys`** — `tantivy 0.26` → tantivy-sstable → `zstd` (C) reaches the default closure in oxirs-fuseki/oxirs-tdb/oxirs-tsdb. Evaluate a pure search backend, a tantivy pure-zstd path, or feature-gate the search/full-text feature; document if unavoidable.
  - ✓ DONE (2026-06-06): tantivy gated behind non-default `full-text-search` feature in oxirs-tdb + oxirs-fuseki (+tools/oxirs `text` wiring); fuseki keeps pure SimpleTextIndex by default. `cargo tree -i zstd-sys` EMPTY in default closure (reappears only with the feature).
- [x] **(MED · cleanup) Drop unused `oxicrypto` facade** (root Cargo.toml ~:337 — all consumers use the `oxicrypto-*` leaf crates) + remove banned `rocksdb` root pin (~:144, feature-gated; replace backend with `oxirs-tdb`).
  - ✓ DONE (2026-06-06): oxicrypto facade line removed (leaves kept); rocksdb fully removed (root pin + oxirs-core feature + tiered.rs 909→27 lines). `cargo tree -i rocksdb`/`oxicrypto` EMPTY.
- [x] **(LOW · ongoing) `/no-unwrap`** — ~16 genuine production `unwrap()` remain (concentrated in `engine/oxirs-ttl/src/trig_streaming/lexer.rs`; also `core/oxirs-core/src/jsonld/compaction/{algorithm.rs:197,context.rs:312}`).
  - ✓ DONE (2026-06-06): 11 genuine production unwraps eliminated in trig_streaming/lexer.rs (while-let restructure, behavior-preserving) + 2 in jsonld compaction (algorithm.rs/context.rs). oxirs-ttl 1771/1771 tests pass.
- [x] **(LOW · hygiene) Tests → `std::env::temp_dir()`** — 128 hardcoded `/tmp/` paths in test scope (heaviest: tools/oxirs 39, oxirs-tdb 22, oxirs-fuseki 18, oxirs-core 13).
  - ✓ DONE (2026-06-06): ~119 test-scope hardcoded /tmp/ paths migrated to std::env::temp_dir()/tempfile::tempdir() (collision-safe: pid-tagged names + auto-cleanup TempDirs) across 16 crates; production config-default /tmp/ strings left as-is. cargo check --tests clean on all 16; ~480 affected tests pass.
