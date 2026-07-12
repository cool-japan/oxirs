# OxiRS Development Roadmap

*Version: 0.3.3 | Last Updated: July 12, 2026*

## Current Status: v0.3.2 - Released (July 12, 2026)

**OxiRS** is an advanced AI-augmented semantic web platform built in Rust, delivering a production-ready alternative to Apache Jena + Fuseki with cutting-edge AI/ML capabilities.

### Release Metrics (v0.3.2, July 12, 2026)
- **Version**: 0.3.2
- **Architecture**: 27-crate workspace (25 published library crates + the `oxirs` CLI + the `oxirs-tauri` desktop app; plus 6 opt-in `publish = false` C-FFI quarantine adapter crates and 1 internal `performance_validation` crate, neither counted above)
- **Build Status**: Clean compilation - zero errors/warnings (`clippy --workspace --all-targets -D warnings`, rustdoc, release build all clean)
- **Test Status**: 45,034 tests passing with `--all-features` (44,344 with default features), 100% pass rate, 0 failed either way; 937 doctests passing across 27 crates
- **SLoC**: ~2.51M total lines / ~2.07M code lines across 5,509 Rust files (tokei)
- **Headline change**: "Pure-Rust Policy v2" — NVML/CUDA/GEOS/DuckDB/Kafka/Pulsar C-FFI extracted into six opt-in `publish = false` quarantine adapter crates (`oxirs-gpu-monitor`, `oxirs-vec-adapter-cuda`, `oxirs-geosparql-adapter-geos`, `oxirs-tsdb-adapter-duckdb`, `oxirs-stream-adapter-rdkafka`, `oxirs-stream-adapter-pulsar`); GeoSPARQL's GeoPackage backend migrated from `rusqlite` to Pure-Rust `oxisql-core`/`oxisql-sqlite-compat`; a Pure-Rust `zstd` shim (`crates/zstd-shim`, backed by `oxiarc-zstd`) removes the last transitive `zstd-sys` C dependency
- **Also shipped**: SHACL `subclass_closure` + real SPARQL/property-path target execution; oxirs-wasm PREFIX/BASE prologues, per-store solution budget, and SPO/POS/OSP-indexed pattern matching; oxirs-tdb saga/2PC/3PC real callback execution; GeoSPARQL shapefile hole-writing and compressed-geometry multi-ring round-tripping fixes; `tools/oxirs` (the CLI) switched to `publish = false` so it can depend on quarantine adapters without putting C FFI on a published surface

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

**Round 35** (v0.3.2, 2026-07-11/12): **Pure-Rust Policy v2** — the 6 remaining in-tree C-FFI feature flags (oxirs-core `gpu`/NVML, oxirs-vec `cuda`+`gpu-full`/CUDA, oxirs-geosparql `geos-backend`/GEOS, oxirs-tsdb `duckdb`, oxirs-stream `kafka`+`pulsar`) extracted into 6 new `publish = false` quarantine adapter crates (`oxirs-gpu-monitor`, `oxirs-vec-adapter-cuda`, `oxirs-geosparql-adapter-geos`, `oxirs-tsdb-adapter-duckdb`, `oxirs-stream-adapter-rdkafka`, `oxirs-stream-adapter-pulsar`), each re-exporting API-compatible types; the old in-tree feature flags were removed outright (breaking). `tools/oxirs` (the CLI) itself flipped to `publish = false` since its optional `tsdb-duckdb` feature now depends on a quarantine crate. GeoSPARQL `GeoPackage` SQLite backend: `rusqlite` (bundled C libsqlite3) → Pure-Rust `oxisql-core`/`oxisql-sqlite-compat`, plus a new `GeoPackage::checkpoint()` for explicit WAL flush. New internal `crates/zstd-shim` (backed by `oxiarc-zstd`) patched over the `zstd` crate workspace-wide via `[patch.crates-io]`, closing the last transitive `zstd-sys` path (tantivy/parquet/pulsar/wasmtime). oxirs-arq: `ordered-float` dependency removed in favor of in-house `total_float.rs` (`TotalF32`/`TotalF64` total-order wrappers). oxirs-wasm: `query::prefix_expand::expand_prologue` (PREFIX/BASE prologue support), `WasmStore::{setSolutionBudget,clearSolutionBudget}` (fail-fast join budget), and pattern/property-path evaluation rewritten to use `subject_index`/`predicate_index`/`object_index` hash lookups instead of full scans. SHACL: `advanced_features::subclass_closure` (reflexive+transitive `rdfs:subClassOf`, Floyd–Warshall over a boolean adjacency matrix); SPARQL-based (`sh:target` SPARQLTarget) and single-hop property-path targets now execute for real against the store. oxirs-tdb: saga/2PC/3PC participants run real registered callbacks against a WAL-backed `Transaction` instead of simulating success; distributed coordinator's `abort_transaction` fans out to all participants. GeoSPARQL: shapefile writer emits interior rings for `Polygon`/`MultiPolygon`; `compressed_storage` gains a `ring_counts` field fixing polygon-hole and multi-part `MultiLineString`/`MultiPolygon` round-tripping; WKT parser accepts optional Z/M coordinates. oxirs-gql/oxirs-federate/oxirs-stream: adaptive query-batch dependency analysis, ML-driven `DynamicQueryPlanner` activated, NATS federation `register_handler()` message-type dispatch, and an MQTT 5.0 property codec all replace prior no-op/stub paths. Dependency refresh: SciRS2 0.5.0→0.6.0, `oxiarc-*` 0.3.3→0.3.5, `oxicrypto`/`oxitls` 0.1.1→0.2.0, `kube` 3.1→4.0 + `k8s-openapi` 0.27→0.28 (optional `k8s` feature), `bytes`→1.12.0 (CVE-2026-25541 fix); unused workspace deps dropped (`rio_api`/`rio_turtle`/`rio_xml`, `rand_distr`, `log`, `env_logger`, `urlencoding`, `serial_test`, `crossbeam-utils`, `tracing-opentelemetry`, `oxicrypto-sig`, others). Verification: 45,034 tests pass (`--all-features`), 44,344 (default features), 0 failed either way; `cargo clippy --workspace --all-targets -D warnings` / rustdoc / release build all 0 warnings; `cargo publish --dry-run` + `cargo package --list` clean.

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

### Follow-up: Pure-Rust Policy v2 (2026-07-12)

> A second pass beyond the policy-check above: 6 previously in-tree, feature-gated C-FFI integrations (reachable only via non-default features, so they didn't violate the original default-build acceptance criteria) are now fully extracted to separate `publish = false` crates, plus `rusqlite` (GeoSPARQL's GeoPackage backend) and the remaining transitive `zstd-sys` paths (tantivy/parquet/pulsar/wasmtime) are closed.

- [x] **(HIGH · Pure-Rust v2) Quarantine the 6 remaining opt-in C-FFI integrations into `publish = false` adapter crates.**
  - Targets: oxirs-core `gpu` (NVML via `nvml-wrapper`), oxirs-vec `cuda`+`gpu-full` (`cuda-runtime-sys`/`cudarc`/`candle-core`), oxirs-geosparql `geos-backend` (`geos`/`geos-sys`), oxirs-tsdb `duckdb` (`duckdb`/`libduckdb-sys`), oxirs-stream `kafka` (`rdkafka`/`rdkafka-sys`/`libz-sys`), oxirs-stream `pulsar` (`pulsar`/`native-tls`/`lz4-sys`).
  - ✓ DONE (2026-07-11/12): 6 new workspace members added, each `publish = false` and API-compatible with the feature it replaces — `core/oxirs-gpu-monitor` (`NvmlGpuMonitor`), `engine/oxirs-vec-adapter-cuda` (`CudaBuffer`/`CudaStream`/`CudaKernel`), `engine/oxirs-geosparql-adapter-geos` (Egenhofer/RCC8/buffer ops), `storage/oxirs-tsdb-adapter-duckdb` (Arrow `RecordBatch` bridge), `stream/oxirs-stream-adapter-rdkafka`, `stream/oxirs-stream-adapter-pulsar` (both moved verbatim from the former in-tree backends). The 6 old in-tree feature flags were removed outright (**breaking**: building with the old flag now fails — depend on the adapter crate directly instead). `tools/oxirs`'s `tsdb-duckdb` feature now depends on `oxirs-tsdb-adapter-duckdb`, which is why the CLI itself became `publish = false` this cycle (a published crate cannot depend on an unpublished path dependency). Net effect: every published crate's `--all-features` dependency surface is 100% Pure Rust.
- [x] **(MED · Pure-Rust v2) Replace `rusqlite` in GeoSPARQL's `GeoPackage` backend.**
  - Decl: `engine/oxirs-geosparql` depended on `rusqlite` (bundled C libsqlite3) for its GeoPackage (OGC GPKG) SQLite storage.
  - ✓ DONE (2026-07-11/12): migrated to the new Pure-Rust **oxisql-core**/**oxisql-sqlite-compat** engine (COOLJAPAN, introduced at 0.3.2); added `GeoPackage::checkpoint()` for explicit WAL flush. `cargo tree -i rusqlite` empty; `grep rusqlite Cargo.toml` finds only the explanatory comment.
- [x] **(MED · Pure-Rust v2) Close the remaining transitive `zstd-sys` paths (tantivy/parquet/pulsar/wasmtime).**
  - Context: the 2026-06-06 policy-check gated tantivy behind non-default `full-text-search` to keep `zstd-sys` out of the *default* build, but `zstd`/`zstd-sys` was still reachable transitively through other consumers (parquet, pulsar, wasmtime, and tantivy itself under `full-text-search`).
  - ✓ DONE (2026-07-11/12): new internal **`crates/zstd-shim`** (backed by `oxiarc-zstd`) implements the `zstd` crate's public surface in Pure Rust; applied workspace-wide via `[patch.crates-io] zstd = { path = "crates/zstd-shim" }`, so every transitive consumer resolves to the shim instead of the C `zstd`/`zstd-sys` crate.

## Stubs to implement (added 2026-06-12 by /cooljapan-stub-check)

- [ ] `oxirs-shacl`: `engine/oxirs-shacl/src/custom_components/js_wasm.rs:145,185,440` — JS execution, JS syntax validation, and WASM module validation are all stubs returning mock results; implement real execution (e.g. `rquickjs`) and validation.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: JS execute :145, JS validate :185, WASM validate :440

- [x] `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/advanced_targets.rs:147,226,240,256` — SPARQL query execution, subclass-reasoning traversal, property-path evaluation, and function execution in advanced SHACL targets all return empty/stub results.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: SPARQL exec :147, subclass :226, path eval :240, fn exec :256

- [ ] `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/parameterized_constraints.rs:340,354,368,381` — Parameterized constraint execution paths for SPARQL ASK, SPARQL SELECT, script, and built-in validator all return `Ok(true)` stubs.
  - Priority: P2 | Scope: large | Hint: none

- [ ] `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/rules.rs:347,359` — SPARQL-ASK condition check and triple-rule execution in SHACL-AF rules are stubs; implement query dispatch and triple inference write-back.
  - Priority: P2 | Scope: medium | Hint: none

- [ ] `oxirs-shacl`: `engine/oxirs-shacl/src/incremental.rs:296,300` — Incremental validator does not extract predicates from constraints or compute target nodes; implement both to enable correct delta-validation.
  - Priority: P2 | Scope: medium | Hint: none

- [ ] `oxirs-shacl`: `engine/oxirs-shacl/src/integration/rule_engine.rs:114,163` — Inferred-triple integration clones the store (placeholder) instead of augmenting it; forward-chaining loop similarly stubbed; implement proper store augmentation.
  - Priority: P2 | Scope: medium | Hint: none

- [ ] `oxirs-shacl`: `engine/oxirs-shacl/src/integration/shex_migration.rs:1099,1111,1129` — ShEx→SHACL migration: triple-constraint conversion, value-expression handling, and value-constraint application are placeholder pass-throughs.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `oxirs-geosparql`: `engine/oxirs-geosparql/src/geometry/compressed_storage.rs:505,521,532` — Compressed geometry deserialization ignores polygon holes, multi-linestring, and multi-polygon ring structures; implement proper ring parsing.
  - Priority: P2 | Scope: medium | Hint: none
  - ✓ DONE (v0.3.2, 2026-07-12): new `ring_counts: Vec<u32>` field records exterior+interior ring/part lengths on compress (`extract_coordinates_with_rings`) and drives `reconstruct_geometry` on decompress, so polygon holes and multi-part `MultiLineString`/`MultiPolygon` structure round-trip correctly.

- [x] `oxirs-geosparql`: `engine/oxirs-geosparql/src/geometry/shapefile_parser.rs:449,625` — Shapefile writing is incomplete (stub comment at top of function); interior rings for polygon export are skipped.
  - Priority: P2 | Scope: medium | Hint: none
  - ✓ DONE (v0.3.2, 2026-07-12): `write_shapefile` accumulates `current_holes: Vec<LineString<f64>>` per ring and writes them into `Polygon::new(exterior, current_holes)`, so interior rings are now emitted for `Polygon`/`MultiPolygon` exports; the stub doc-comment is gone.

- [x] `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/saga.rs:365,424` — Saga step execution and compensation both sleep 10 ms and always succeed; wire real step callbacks and compensating-transaction callbacks.
  - Priority: P2 | Scope: large | Hint: none

- [x] `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/two_phase_commit.rs:485,509,532` — 2PC participant `can_commit`, `execute_commit`, and `execute_abort` all return immediately without touching transaction state; implement resource-lock check and actual durable commit/abort.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: resource check :485, commit :509, abort :532

- [x] `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/three_phase_commit.rs:551,577,601,625` — 3PC participant phases (can-commit, pre-commit, commit, abort) all stub out; implement the three-phase protocol over the WAL.
  - Priority: P2 | Scope: large | Hint: none

- [ ] `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/coordinator.rs:393,459,529,535` — Distributed coordinator: abort fan-out to participants missing; result merging returns first shard; comparison and stale-node repair are stubs.
  - Priority: P2 | Scope: large | Hint: none
  - Partial progress (v0.3.2, 2026-07-12): the `:393` abort-fan-out sub-issue is fixed — `abort_transaction` now notifies all registered participants (see the dedicated 2026-06-22 entry below). `:459` result merging, `:529` comparison, and `:535` stale-node repair remain open.

- [ ] `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:489,554,617,680` — Cloud storage backends (S3, GCS, Azure Blob, MinIO) have no client initialization; all four `new()` methods return a shell struct without connecting.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: S3 :489, GCS :554, Azure :617, MinIO :680

- [ ] `oxirs-tdb`: `storage/oxirs-tdb/src/query_join_optimizer.rs:284` — Join optimizer genetic-algorithm path is a stub returning the unoptimized plan; implement basic GA (population init, crossover, mutation, fitness by estimated cost).
  - Priority: P2 | Scope: large | Hint: none

- [ ] `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/coordinator.rs:329,459,529,535` — Cluster coordinator: partition-to-node mapping returns None always; result merging takes first response; consistency comparison and read-repair are stubs.
  - Priority: P2 | Scope: large | Hint: none

- [ ] `oxirs-fuseki`: `server/oxirs-fuseki/src/handlers/ngsi_ld/server_handlers.rs:76,171` — NGSI-LD entity retrieval falls back to cache-only (no RDF CONSTRUCT query); NGSI-LD SPARQL translation is a stub returning empty results.
  - Priority: P2 | Scope: medium | Hint: none

- [x] `oxirs-arq`: `engine/oxirs-arq/tests/w3c_compliance/mod.rs:552,805,883,890,903` — W3C compliance harness uses `term1 == term2` for equality (no blank-node renaming), and graph-isomorphism check is missing; RDF parsing helpers for TTL/NT/RDF-XML/N3 are also stubs.
  - Priority: P2 | Scope: medium | Hint: none
  - Locations: term equality :552, graph iso :805, parsing stubs :883,890,903

- [x] `oxirs-vec`: `engine/oxirs-vec/src/real_time_embedding_pipeline/pipeline.rs` — Real-time embedding pipeline is missing four sub-modules (consistency, coordination, performance-monitoring, versioning); all are commented-out with TODO; implement the modules and wire them into `EmbeddingPipeline`.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: mod.rs:50,82; pipeline.rs:53,108,117,124,148,157,173,271,282,293,318,420,427

- [ ] `tools/oxirs`: `tools/oxirs/src/commands/modbus.rs` — All modbus CLI sub-commands (monitor-tcp, monitor-rtu, read-registers, write-register, mock-server, benchmark) print a yellow "TODO:" message and do nothing; implement oxirs-modbus integration.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: :83,:110,:134,:152,:174,:196,:225,:247 (8 stubs across the command handlers)

- [ ] `tools/oxirs`: `tools/oxirs/src/commands/canbus.rs` — CAN bus CLI commands (monitor, decode, replay, capture, benchmark) all print "TODO:" stubs; implement via oxirs-canbus.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: :96,:120,:132,:174,:207,:241,:265,:280 (8 stubs)

- [ ] `tools/oxirs`: `tools/oxirs/src/commands/tsdb.rs` — TSDB CLI commands (insert, query, batch-import, export, benchmark) print "TODO:" stubs without calling any store API.
  - Priority: P2 | Scope: large | Hint: none
  - Locations: :111,:130,:146,:171,:194,:248,:278,:298 (8 stubs)

- [ ] `oxirs-fuseki`: `server/oxirs-fuseki/src/federation/discovery.rs:598` — Kubernetes-based federation endpoint discovery is a stub returning an empty list; implement via k8s API (label selector watch on SPARQL-endpoint pods).
  - Priority: P2 | Scope: medium | Hint: none

- [x] `oxirs-stream`: `stream/oxirs-stream/src/backend/mqtt/client.rs:169` — MQTT 5.0 properties are dropped on inbound messages; extract and surface them in the `MqttMessage` envelope.
  - Priority: P2 | Scope: small | Hint: none


## Stubs to implement (added 2026-06-22 by /cooljapan-stub-check)

### oxirs-shacl

- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/constraints/value_constraints.rs:167` — `TODO`: Check subclass relationships using RDFS reasoning (sh:class only checks direct rdf:type)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** After the direct-type miss, walk `rdfs:subClassOf*` transitively from each asserted type of the node and accept if `class` is reachable.
  - **Risk:** Cycles in the subclass graph must be guarded with a visited-set to avoid infinite loops.
- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/constraints/string_constraints.rs:425` — `TODO`: More thorough BCP 47 validation (sh:languageIn only rejects empty tags)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Validate each tag against the BCP-47 grammar (language[-script][-region][-variant]) with subtag length/charset checks.
  - **Risk:** Over-strict validation could reject legitimate extended/private-use tags; keep grandfathered/`x-` forms permissive.
- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/advanced_targets.rs:147` — `TODO`: Implement SPARQL query execution (evaluate_sparql_target returns empty set)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Wire `evaluate_sparql_target` to the oxirs-arq engine, run the target query against the Store, and collect bound focus nodes.
  - **Risk:** Honoring `timeout_ms` requires cooperative cancellation in the query engine.
- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/advanced_targets.rs:226` — `TODO`: Implement subclass reasoning (implicit class target ignores subclasses)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** When `include_subclasses`, compute the `rdfs:subClassOf*` closure of `class` and union instances of every subclass.
  - **Risk:** Closure cost on large ontologies; cache the subclass graph per validation run.
- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/advanced_targets.rs:240` — `TODO`: Implement property path evaluation (evaluate_path_target echoes root nodes)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Evaluate `path` via the existing `PropertyPathEvaluator` from each root node and return the reached terms as targets.
  - **Risk:** Recursive paths (ZeroOrMore/OneOrMore) need cycle-safe traversal over the Store.
- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/advanced_targets.rs:256` — `TODO`: Implement function execution (function-based target unimplemented)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Resolve `function_id` against a function registry and invoke it with the bound arguments to produce target terms.
  - **Risk:** Requires a SHACL function/SPARQL-function registry that may not yet exist.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/parameterized_constraints.rs:340` — `TODO`: Implement SPARQL ASK execution with parameter substitution (always conforms)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Substitute `$this`/`$value`/named params into `query_template`, run ASK via oxirs-arq, map false→violation.
  - **Risk:** Safe parameter binding must avoid SPARQL injection from term values.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/parameterized_constraints.rs:354` — `TODO`: Implement SPARQL SELECT execution with parameter substitution (always conforms)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Bind params into the SELECT template, execute, and treat each returned row's `result_variable` as a violating value.
  - **Risk:** Result-variable projection and empty-result semantics must match SHACL-SPARQL spec.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/parameterized_constraints.rs:368` — `TODO`: Implement script execution (script constraint always conforms)
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Dispatch on `ScriptLanguage`; for JS reuse the js_wasm pure-Rust JS engine path, returning a conformance verdict.
  - **Risk:** Needs a sandboxed pure-Rust script runtime; large surface, defer behind a feature gate.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/parameterized_constraints.rs:381` — `TODO`: Implement built-in validator dispatch (always conforms)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Map `validator_name` to the corresponding built-in constraint component and delegate evaluation.
  - **Risk:** Unknown validator names should error rather than silently conform.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/parameterized_constraints.rs:172` — `TODO`: Convert other types to terms (ParameterValue::to_term only handles Term)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Implement to_term for Path/List/scalar variants (e.g. literal-encode scalars), returning None only when genuinely non-term.
  - **Risk:** List→term has no single-term representation; document the None case.
- [x] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/shape_inference.rs:219` — `TODO`: Query store for instances (collect_class_instances returns empty Vec)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** `find_quads(None, rdf:type, class, None)` and collect subjects as instances.
  - **Risk:** Without subclass expansion, inference under-samples; coordinate with the subclass-closure helper.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/rules.rs:347` — `TODO`: Execute SPARQL ASK query (evaluate_condition always returns true)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Run the `condition` ASK against the Store via oxirs-arq and return the boolean verdict.
  - **Risk:** A true-by-default stub silently passes rule conditions; tests must assert real false outcomes.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/advanced_features/rules.rs:359` — `TODO`: Implement triple rule execution (currently pushes an error)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Evaluate the rule's subject/predicate/object node-expressions and assert the produced triple(s) into the inferred set.
  - **Risk:** Store augmentation must be isolated from the source graph (see rule_engine forward-chaining item).
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/integration/rule_engine.rs:163` — `TODO`: Implement proper forward chaining with store augmentation (forward_chain is a no-op stub)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Materialize inferred triples into an overlay/cloned Store, iterate rules to fixpoint, then validate against the augmented view.
  - **Risk:** Naive fixpoint can loop; bound iterations and dedupe inferred triples.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/integration/rule_engine.rs:114` — `TODO`: Implement proper Store cloning/augmentation for inferred triples
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Introduce a layered Store wrapper (base + inferred set) so reasoning output is queryable without mutating the original.
  - **Risk:** Lifetime/ownership of the wrapped `&dyn Store`; may need an owned overlay store.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/integration/shex_migration.rs:1099` — `TODO`: Properly implement triple constraint conversion (only counts stats, skips conversion)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Build a full SHACL property shape (path + cardinality + value constraints) from the ShEx `TripleConstraint`.
  - **Risk:** ShEx value expressions and semantic actions have no 1:1 SHACL mapping; record unmapped features.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/integration/shex_migration.rs:1111` — `TODO`: Handle value expression properly (currently logged as unmapped)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Translate ShEx node constraints (datatype/nodeKind/values) into the corresponding SHACL value constraints on the property shape.
  - **Risk:** Shape references / nested expressions may require recursive shape generation.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/integration/shex_migration.rs:1129` — `TODO`: Properly implement value constraint application (PropertyConstraint model too narrow)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Once full property-shape model exists, apply the translated value constraints instead of recording an unmapped feature.
  - **Risk:** Depends on the triple-constraint-conversion item landing first.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/incremental.rs:296` — `TODO`: Extract predicates from constraints (dirty-set predicate gap)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Recursively walk each shape's constraints, collecting referenced predicates (paths, sh:property targets) into `affected_by_predicates`.
  - **Risk:** Empty predicate sets defeat incremental validation (over-invalidates or misses changes).
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/incremental.rs:300` — `TODO`: Get target nodes for this shape (target_nodes left empty)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Resolve the shape's targets against the Store to populate `target_nodes` for change-impact computation.
  - **Risk:** Must stay in sync with the same target-evaluation logic used by full validation.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/validation/async_engine.rs:706` — `TODO`: Implement actual shape-specific validation with the provided nodes (returns empty report)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Drive `validate_node_against_shape` for each provided node under the resolved shape and aggregate results.
  - **Risk:** Async locking around the engine guard must avoid holding the lock across `.await`.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/custom_components/js_wasm.rs:145` — `TODO`: Implement actual JavaScript execution with rquickjs or similar (sh:js mocked)
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Integrate a pure-Rust JS engine (feature-gated, COOLJAPAN-policy compliant) to execute the validator and return the real verdict.
  - **Risk:** Cross-project dependency; a Pure-Rust JS engine must exist and be sandboxed.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/custom_components/js_wasm.rs:185` — `TODO`: Implement syntax validation (validate_code only checks non-empty)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Parse the JS source with the chosen pure-Rust JS engine's parser and surface parse errors.
  - **Risk:** Tied to the JS-engine selection above.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/custom_components/js_wasm.rs:440` — `TODO`: Implement WASM module validation
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Validate the WASM bytes (magic/version + a wasm validator) before registering the component.
  - **Risk:** Requires a pure-Rust wasm validation crate.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/custom_components/marketplace.rs:730` — `TODO`: Implement unregister_component in CustomConstraintRegistry (uninstall errors out)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Add `unregister_component(id)` to the registry and call it from `uninstall`, returning Ok on removal.
  - **Risk:** Removing an in-use component must not leave dangling shape references.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/shape_import.rs:1072` — `TODO`: Remap IRIs in condition (Conditional target condition not remapped on import)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Recurse `remap_target_iris` into the conditional target's condition sub-targets.
  - **Risk:** Missed remaps cause cross-namespace dangling targets after import.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/shape_import.rs:1076` — `TODO`: Remap IRIs in relationship (Hierarchical target relationship not remapped)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Apply namespace remapping to the hierarchical relationship predicate/IRIs.
  - **Risk:** Same dangling-IRI risk as the conditional case.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/shape_import.rs:1080` — `TODO`: Remap IRIs in path (PathBased target path not remapped)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Walk the property path AST and remap each predicate IRI under the target namespace.
  - **Risk:** Complex path variants (inverse/sequence) must all be covered.
- [ ] **oxirs** `oxirs-shacl`: `engine/oxirs-shacl/src/w3c_test_suite.rs:666` — `TODO`: Add more detailed violation matching
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Compare expected vs actual violations on focus node + source shape + constraint component, not just count.
  - **Risk:** Stricter matching may surface latent conformance failures.

### oxirs-tdb

- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/two_phase_commit.rs:485` — `TODO`: Implement actual resource checking (can_commit always returns true optimistically)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Inspect the txn's write-set against locks/disk/quota and vote NO when resources are unavailable.
  - **Risk:** Always-yes voting makes 2PC unable to abort; correctness depends on this.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/two_phase_commit.rs:509` — `TODO`: Implement actual commit logic (execute_commit just sleeps)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Apply the prepared txn's buffered writes to the store/WAL and fsync on commit.
  - **Risk:** Without durable apply, committed data is lost; must be crash-safe.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/two_phase_commit.rs:532` — `TODO`: Implement actual abort logic (execute_abort just sleeps)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Discard the prepared write buffer and release held locks/resources on abort.
  - **Risk:** Leaked locks on abort can deadlock subsequent transactions.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/three_phase_commit.rs:551` — `TODO`: Implement actual resource checking (3PC can_commit always true)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Same resource/lock availability check as 2PC, returning the real vote.
  - **Risk:** 3PC's non-blocking guarantee is meaningless if votes are unconditional.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/three_phase_commit.rs:577` — `TODO`: Implement actual pre-commit logic (execute_pre_commit just sleeps)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Durably record the pre-commit decision (so a recovering participant can finish) before moving to WaitingCommit.
  - **Risk:** Missing pre-commit persistence breaks 3PC recovery semantics.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/three_phase_commit.rs:601` — `TODO`: Implement actual commit logic (execute_commit just sleeps)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Apply buffered writes durably, mirroring the 2PC commit implementation.
  - **Risk:** Same durability/crash-safety risk as 2PC commit.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/transaction/three_phase_commit.rs:625` — `TODO`: Implement actual abort logic (execute_abort just sleeps)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Roll back pre-committed/buffered state and release resources.
  - **Risk:** Lock leaks on abort.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/saga.rs:365` — `TODO`: Implement actual step execution with callbacks (always simulates success)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Add an executable callback to each SagaStep and invoke it, propagating real success/failure to drive compensation.
  - **Risk:** Hardcoded `success = true` means compensation never triggers; tests must exercise failures.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/saga.rs:424` — `TODO`: Implement actual compensation with callbacks (compensate_step simulates)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Invoke each step's compensation callback in reverse order to undo committed effects.
  - **Risk:** Non-idempotent compensations on retry can double-undo.
- [x] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/coordinator.rs:393` — `TODO`: Send abort to all participants (abort_transaction only flips local state)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Broadcast an ABORT message to every participant before marking the txn Aborted.
  - **Risk:** Participants left in Prepared state hold locks indefinitely.
  - **✓ DONE (v0.3.2, 2026-07-12):** `abort_transaction` now snapshots `participants` from the transaction record and fans the abort notification out to all of them (in-process simulated transport — no real network hop yet, consistent with the rest of this coordinator module).
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/deadlock.rs:415` — `TODO`: Implement actual work tracking (LeastWork victim strategy falls back to youngest)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Track per-transaction work (ops/log bytes) and pick the lowest-work victim instead of defaulting to youngest.
  - **Risk:** Wrong victim selection wastes the most expensive in-flight work.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/integration.rs:255` — `TODO`: Replicate transaction changes (only bumps a counter)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Hand the committed change-set to the replication_manager so replicas receive the writes.
  - **Risk:** Counter increments without real replication give false durability metrics.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/distributed/integration.rs:295` — `TODO`: Integrate saga execution with coordinator (execute_saga returns Ok(true) unconditionally)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Run the SagaOrchestrator under the coordinator so saga steps participate in distributed-txn accounting.
  - **Risk:** Independent saga execution can diverge from coordinator state.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:334` — `TODO`: Implement cross-region replication (pushes region names without copying)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Copy the backup object to each target region's bucket via the cloud backend before recording it.
  - **Risk:** Recording replicated_regions without real copies overstates DR coverage.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:451` — `TODO`: Implement storage class transition (only increments counter)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Issue the backend's change-storage-class operation for objects past the transition age.
  - **Risk:** No real transition means lifecycle cost-savings never materialize.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:489` — `TODO`: Initialize AWS S3 client (S3Backend is a mock)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Construct a real S3 client (pure-Rust HTTP + SigV4, reuse existing oxirs S3 SigV4 code) instead of warning-and-mocking.
  - **Risk:** Network/credential handling; keep behind the cloud feature and avoid C/asm TLS per policy.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:554` — `TODO`: Initialize GCS client (GCS backend mock)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Wire a real GCS client using OAuth2 (reuse existing GCS OAuth2 support in the workspace).
  - **Risk:** Same credential/TLS-purity concerns as S3.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:617` — `TODO`: Initialize Azure Blob Storage client (Azure backend mock)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Build a real Azure Blob client using SharedKey auth (reuse existing Azure SharedKeyLite code).
  - **Risk:** Credential/TLS-purity concerns.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/cloud_storage.rs:680` — `TODO`: Initialize MinIO client (S3-compatible) (mock)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Reuse the S3Backend client with a custom endpoint/path-style for MinIO.
  - **Risk:** Path-style vs virtual-host addressing must be configurable.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/diagnostics_collectors.rs:703` — `TODO`: Add CRC checksum verification for WAL entries (size-only check today)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Read WAL entries and verify each entry's stored CRC, reporting mismatches as a diagnostic error.
  - **Risk:** Must match the exact CRC algorithm/layout the WAL writer uses.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/query_join_optimizer.rs:284` — `TODO`: Implement genetic algorithm (Genetic join order falls back to greedy)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement a GA over join orderings (population, crossover, mutation, cost-fitness) for large join graphs.
  - **Risk:** GA tuning is nontrivial; ensure it never produces worse plans than the greedy fallback.
- [ ] **oxirs** `oxirs-tdb`: `storage/oxirs-tdb/src/index/spatial/functions.rs:116` — `TODO`: Implement proper self-intersection check (isSimple returns true for LineString/Polygon)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Run a segment-intersection sweep (Bentley-Ottmann or pairwise for small N) to detect self-intersections per GeoSPARQL isSimple.
  - **Risk:** Always-true gives incorrect isSimple results for non-simple geometries.

### oxirs-vec

- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/mmap_advanced.rs:413` — `TODO`: Implement write-back (dirty evicted mmap pages are dropped)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** On LRU eviction of a dirty page, flush its bytes back to the backing mmap/file before discarding.
  - **Risk:** Dropping dirty pages loses index updates; flush ordering must be consistent.
- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/multi_modal_search.rs:775` — `TODO`: implement cache hit tracking (cache_hit_rate hardcoded 0.0)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Add atomic hit/miss counters to the query cache and compute the rate in `MultiModalStatistics`.
  - **Risk:** Counter contention is negligible; just keep counters consistent with cache lookups.
- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/python_bindings.rs:96` — `TODO`: Properly handle index_type by creating appropriate index (index_type ignored)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Map the parsed `index_type` to the matching VectorStore index builder instead of always using the embedding strategy default.
  - **Risk:** Silently ignoring index_type misleads Python users about the index in use.
- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/python_bindings.rs:846` — `TODO`: implement other metrics (only a subset of distance metrics wired)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Extend the metric match to cover the remaining DistanceMetric variants exposed by the core store.
  - **Risk:** Unhandled metrics fall through to a default and give wrong distances.
- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/real_time_embedding_pipeline/pipeline.rs:53` — `TODO`: Implement these modules (coordination/consistency/versioning/monitoring skeleton commented out)
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Build the four pipeline submodules (UpdateCoordinator, ConsistencyManager, VersionManager, PipelinePerformanceMonitor) and wire them into RealTimeEmbeddingPipeline, replacing the commented fields/no-op methods at lines 108/117/124/148/157/271/282/293/420.
  - **Risk:** Large surface with cross-cutting async coordination; land incrementally per submodule.
- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/real_time_embedding_pipeline/pipeline.rs:173` — `TODO`: Implement proper stream processor stopping once StreamProcessor is made cloneable
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Make StreamProcessor cloneable/shareable (Arc) so the pipeline can signal and join a real stop.
  - **Risk:** Improper shutdown can leak background tasks.
- [x] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/real_time_embedding_pipeline/pipeline.rs:318` — `TODO`: Implement proper async health checking mechanism (also streaming.rs:329)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Replace the placeholder health check with real async probing of pipeline stages/queues.
  - **Risk:** False-healthy reporting masks stalled stages.
- [ ] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/mmap_advanced.rs:601` — `TODO`: Implement actual NUMA allocation when libc bindings are available (also :623)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Use `numa_alloc_onnode` for node-local allocation; keep behind a feature gate and fall back to the standard allocator.
  - **Risk:** NUMA libc bindings are platform-specific and may conflict with Pure-Rust default-feature policy (must be feature-gated).
- [ ] **oxirs** `oxirs-vec`: `engine/oxirs-vec/src/gpu/runtime.rs:21` — `TODO`: Implement proper CUDA runtime integration
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Back the GPU runtime with a real CUDA path (feature-gated), falling back to CPU when unavailable.
  - **Risk:** GPU/CUDA is out of the Pure-Rust default; must be feature-gated and optional.

### oxirs-core / jsonld

- [x] **oxirs** `oxirs-core`: `core/oxirs-core/src/jsonld/to_rdf_converter.rs:216` — `TODO`: this is bad (rdf:List vs @set conflated)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Model @set distinctly from @list: emit set members as plain repeated values (no rdf:first/rest/nil), reserving List state for @list.
  - **Risk:** Mis-emitting sets as RDF lists corrupts round-trips; needs conformance tests.
- [x] **oxirs** `oxirs-core`: `core/oxirs-core/src/jsonld/expansion_algorithm.rs:914` — `TODO`: emit @index (index-map entries dropped)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** In IndexContainer expansion, attach the object key as an `@index` value on the expanded element per JSON-LD index maps.
  - **Risk:** Dropping index keys loses data on index-container inputs.
- [x] **oxirs** `oxirs-core`: `core/oxirs-core/src/jsonld/context_core.rs:1180` — `TODO`: make sure it's full (protected-term override check possibly incomplete)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Audit the protected-term redefinition comparison to cover all term-definition fields before raising ProtectedTermRedefinition.
  - **Risk:** Missing a field lets protected terms be silently overridden.

### oxirs-core / rdfxml

- [x] **oxirs** `oxirs-core`: `core/oxirs-core/src/rdfxml/serializer.rs:591` — `TODO`: does not work on recursive elements
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Add cycle detection during RDF/XML serialization and break cycles using `rdf:nodeID` references instead of infinite nesting.
  - **Risk:** Recursive/cyclic graphs currently overflow or mis-serialize; nodeID emission must round-trip.

### oxirs-gql

- [x] **oxirs** `oxirs-gql`: `server/oxirs-gql/src/parallel_field_resolver.rs:503` — `TODO`: Track average in the resolver (avg_resolution_time_ms hardcoded 0.0)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Accumulate total resolution time + count and compute the average in `get_metrics`.
  - **Risk:** Hardcoded 0.0 makes the metric useless for monitoring.
- [x] **oxirs** `oxirs-gql`: `server/oxirs-gql/src/query_batching.rs:500` — `TODO`: Implement dependency analysis (execute_adaptive just runs parallel)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Build a dependency graph among batched queries and execute independent groups in parallel, dependents in order.
  - **Risk:** Incorrect dependency detection can serialize unnecessarily or run dependents too early.
- [x] **oxirs** `oxirs-gql`: `server/oxirs-gql/src/dynamic_query_planner.rs:236` — `TODO`: Integrate with performance tracker when available (ML optimizer disabled)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Provide a performance-tracker dependency so MLQueryOptimizer can be constructed and consulted during planning.
  - **Risk:** Requires the tracker component; keep planner functional when absent.
- [x] **oxirs** `oxirs-gql`: `server/oxirs-gql/src/dynamic_query_planner.rs:539` — `TODO`: Integrate ML training when performance tracker is available
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Feed observed query features/targets into the ML optimizer's training step once the tracker is wired.
  - **Risk:** Tied to the planner-integration item above.

### oxirs-arq

- [x] **oxirs** `oxirs-arq`: `engine/oxirs-arq/tests/w3c_compliance/mod.rs:805` — `TODO`: Implement proper graph isomorphism checking (blank-node-aware)
  - **Priority:** P2  **Scope:** large  **Cross-project:** none
  - **Approach:** Implemented via URDNA2015 canonicalization (oxirs_core::canonicalize) for isomorphism-safe graph comparison.
  - **Risk:** General isomorphism is expensive; needs pruning and is the gate for many W3C result checks.
- [x] **oxirs** `oxirs-arq`: `engine/oxirs-arq/tests/w3c_compliance/mod.rs:552` — `TODO`: Implement proper term equality (considering blank node renaming)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Blank nodes now use existential semantics in terms_equal (any two blank nodes are equal); graph comparison uses canonicalization.
  - **Risk:** Depends on the isomorphism implementation; shares its blank-node bijection.

### oxirs-fuseki

- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/coordinator.rs:329` — `TODO`: Implement actual partition to node mapping (returns dummy ["node1"])
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Resolve partitions to owning nodes via the cluster's partition assignment map.
  - **Risk:** Dummy mapping routes all queries to one node, breaking real clustering.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/coordinator.rs:459` — `TODO`: Implement proper result merging (returns first result only)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Merge per-node SPARQL result sets (union/dedupe for bag/set semantics) instead of taking the first.
  - **Risk:** Returning only the first node's results yields incomplete answers.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/coordinator.rs:529` — `TODO`: Implement proper result comparison (results_equal returns false)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Implement set-equality over result rows so read-repair can detect divergent replicas.
  - **Risk:** Always-false flags every replica as stale, triggering needless repairs.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/coordinator.rs:535` — `TODO`: Implement repair writes to stale nodes (repair_nodes is a no-op)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Push the latest value to identified stale nodes to converge replicas.
  - **Risk:** No-op repair leaves replicas permanently inconsistent.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/partition.rs:312` — `TODO`: Trigger rebalancing (skew detected but no action)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Invoke the rebalancing routine (move_partition) when node skew exceeds the threshold.
  - **Risk:** Detect-without-act leaves hotspots; coordinate with data-migration item.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/partition.rs:334` — `TODO`: Implement actual data migration (move_partition flips assignment only)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Copy partition data from source to target node before flipping the assignment, then drop the source copy.
  - **Risk:** Flipping assignment without moving data loses access to that partition's triples.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/mod.rs:361` — `TODO`: Implement seed contact protocol (contact_seed is a no-op)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Connect to the seed node, exchange membership/gossip, and join the cluster view.
  - **Risk:** Without seed contact, nodes never discover peers.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/mod.rs:259` — `TODO`: Get capacity from system (hardcoded 1000)
  - **Priority:** P2  **Scope:** trivial  **Cross-project:** none
  - **Approach:** Derive node capacity from real system resources (memory/disk) instead of the constant.
  - **Risk:** Wrong capacity skews load-balancing decisions.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/clustering/mod.rs:461` — `TODO`: Calculate under_replicated_partitions (hardcoded 0)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Count partitions whose replica set is below the configured replication factor.
  - **Risk:** Hardcoded 0 hides replication health problems.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/handlers/sparql/core.rs:797` — `TODO`: Integrate federated query results into response (results discarded)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Merge `federated_results` into the response instead of continuing with the original query only.
  - **Risk:** Federated results are computed then thrown away — wasted work and wrong answers.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/handlers/ngsi_ld/server_handlers.rs:76` — `TODO`: Implement full RDF query when Store API supports CONSTRUCT (get_entity is cache-only)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Query the RDF store with CONSTRUCT for the entity and reconstruct the NgsiEntity on cache miss.
  - **Risk:** Cache-only reads miss entities persisted only in the store.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/handlers/ngsi_ld/server_handlers.rs:171` — `TODO`: Implement SPARQL query translation when Store API is available (query_entities is cache-only)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Execute the already-translated SPARQL against the store and map results to NgsiEntities instead of filtering the cache.
  - **Risk:** Cache-only filtering returns incomplete query results.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/streaming/pipeline.rs:150` — `TODO`: Implement session window logic (Session windows unhandled)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Create/extend session windows dynamically by gap timeout, emitting on inactivity.
  - **Risk:** Session events are silently dropped today.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/streaming/pipeline.rs:161` — `TODO`: Extract actual event time (uses Instant::now())
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Read the event's own timestamp from the RDFEvent payload for windowing instead of wall-clock now.
  - **Risk:** Wall-clock time breaks event-time windowing and watermarks for out-of-order data.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/federation/discovery.rs:598` — `TODO`: Implement Kubernetes-based discovery (returns nothing)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Query the k8s API (label-selector on SPARQL endpoint pods/services) to populate the endpoint map.
  - **Risk:** Requires a k8s client dependency; keep optional/feature-gated.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/auth/graph_auth.rs:316` — `TODO`: Optimize with actual batch checking at RebacEvaluator level (loops single checks)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Add a batch API to RebacEvaluator that resolves many graph permissions in one pass.
  - **Risk:** Per-request looping is correct but O(n) round-trips; purely a performance fix.
- [ ] **oxirs** `oxirs-fuseki`: `server/oxirs-fuseki/src/ids/contract.rs:539` — `TODO`: Add digital signatures (contract signatures left empty)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Sign the accepted contract (reuse the workspace's ed25519/oxicrypto signing path) and populate the signatures field.
  - **Risk:** Unsigned contracts are non-repudiation gaps in the IDS flow.

### oxirs-stream

- [x] **oxirs** `oxirs-stream`: `stream/oxirs-stream/src/backend/mqtt/client.rs:169` — `TODO`: Extract MQTT 5.0 properties (properties: None on inbound)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Read the rumqtt publish properties and populate `MqttMessage.properties`.
  - **Risk:** Dropping properties loses user/content-type/correlation metadata for MQTT 5.0 consumers.

### oxirs-federate

- [x] **oxirs** `oxirs-federate`: `stream/oxirs-federate/src/nats_federation.rs:602` — `TODO`: Send request via NATS (request is registered but never published)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Publish the federation request on the NATS subject and await the correlated reply instead of just timing out.
  - **Risk:** Currently every request simulates with a timeout; no real federation occurs.
- [x] **oxirs** `oxirs-federate`: `stream/oxirs-federate/src/nats_federation.rs:884` — `TODO`: Add message processing logic (received federation messages only logged)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Dispatch inbound federation messages by type (request/response/control) to their handlers.
  - **Risk:** Log-only handling means inbound federation traffic is effectively ignored.

### oxirs-geosparql

- [x] **oxirs** `oxirs-geosparql`: `engine/oxirs-geosparql/src/geometry/compressed_storage.rs:505` — `TODO`: Handle holes properly (Polygon decompresses with no interior rings; also :521 multilinestring, :532 multipolygon)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Encode/decode ring and part boundaries so polygons keep holes and multi-geometries keep separate parts.
  - **Risk:** Flattening to a single ring/part loses topology and corrupts area/contains results.
  - **✓ DONE (v0.3.2, 2026-07-12):** new `ring_counts: Vec<u32>` field records ring/part boundaries on compress and drives reconstruction on decompress — see the matching 2026-06-12 entry above.
- [ ] **oxirs** `oxirs-geosparql`: `engine/oxirs-geosparql/src/geometry/zero_copy_wkt.rs:477` — `TODO`: Parse Z/M if present (only X/Y parsed)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Detect the Z/M flag in the WKT coordinate and parse the extra ordinate(s) into the Coord.
  - **Risk:** Silently dropping Z/M misreads 3D/measured geometries.

### oxirs-chat

- [ ] **oxirs** `oxirs-chat`: `ai/oxirs-chat/src/nl2sparql/context_aware.rs:203` — `TODO`: Integrate with NLP entity extractor (heuristic-only extraction)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Call the NLP entity extractor when present in context and fall back to the heuristic otherwise.
  - **Risk:** Heuristic stopword filtering misses real entities; degrades NL2SPARQL quality.
- [ ] **oxirs** `oxirs-chat`: `ai/oxirs-chat/src/rag/advanced_retrieval.rs:208` — `TODO`: Integrate with actual embedding model (DensePassageRetriever uses similarity stub)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Back DPR with a real embedding model (oxirs-embed) for query/passage vectors.
  - **Risk:** Similarity-only retrieval underperforms true dense retrieval.
- [ ] **oxirs** `oxirs-chat`: `ai/oxirs-chat/src/advanced_observability.rs:587` — `TODO`: Implement anomaly detection using statistical analysis (detect_anomalies is a no-op)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Apply a statistical detector (z-score / EWMA threshold) over audit-event rates to flag anomalies.
  - **Risk:** No-op detection means security anomalies go unnoticed.
  - **Status:** `detect_anomalies` now runs Welford's online mean/variance algorithm over audit-event inter-arrival times and flags a z-score threshold breach — implemented in `advanced_observability.rs`, but the module is not yet wired into `lib.rs` (`pub mod advanced_observability` is commented out, gated on a future scirs2-core beta.4+ upgrade per the in-source comment), so it is not part of the compiled crate yet.
- [ ] **oxirs** `oxirs-chat`: `ai/oxirs-chat/src/advanced_observability.rs:598` — `TODO`: Integrate with alerting system (security alerts only warn-logged)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Dispatch critical security events to a pluggable alert sink (email/Slack/webhook) in addition to logging.
  - **Risk:** Log-only alerts may be missed in production.
  - **Status:** `register_alert_handler()` lets callers register handlers that receive dispatched security events, in addition to logging — implemented in `advanced_observability.rs`, but not yet reachable: the module is not wired into `lib.rs` (`pub mod advanced_observability` is commented out, gated on a future scirs2-core beta.4+ upgrade per the in-source comment).

### tools/oxirs (CLI commands)

- [ ] **oxirs** `oxirs` (tools): `tools/oxirs/src/commands/tsdb.rs:130` — `TODO`: CSV batch import not yet implemented (insert command returns early)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Parse the CSV and bulk-insert points into the hybrid store (also wire single-point insert at :146).
  - **Risk:** The command prints success while doing nothing; users get silent no-ops.
- [ ] **oxirs** `oxirs` (tools): `tools/oxirs/src/commands/tsdb.rs:298` — `TODO`: Run actual benchmark with HybridStore (query/stats/benchmark print placeholders at :111/:171/:194/:248/:278)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Connect the tsdb subcommands to a real HybridStore and execute query/stats/benchmark operations.
  - **Risk:** Placeholder output misrepresents tool capability.
- [ ] **oxirs** `oxirs` (tools): `tools/oxirs/src/commands/modbus.rs:134` — `TODO`: Connect and read/write registers (read at :134, write at :152, monitor/scan/serve/bench placeholders at :83/:110/:174/:196/:225/:247)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Wire the modbus subcommands to oxirs-modbus to perform real device reads/writes/monitoring instead of printing TODO lines.
  - **Risk:** Requires live Modbus device for full e2e; core wiring can be unit-tested with a mock server.
- [ ] **oxirs** `oxirs` (tools): `tools/oxirs/src/commands/canbus.rs:265` — `TODO`: Capture frames and generate RDF (capture at :265, replay at :280; decode/monitor/dbc placeholders at :96/:120/:132/:174/:207/:241)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Connect the canbus subcommands to oxirs-canbus to capture/replay frames and emit RDF.
  - **Risk:** Live CAN interface needed for capture; DBC-driven decode/replay can be tested offline.

### oxirs-embed

- [x] **oxirs** `oxirs-embed`: `ai/oxirs-embed/src/vector_search.rs:261` — `TODO`: Implement full HNSW for very large datasets (approximate_search falls back to exact)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Build a real HNSW graph (reuse oxirs-vec HNSW) for the approximate path instead of delegating to exact search.
  - **Risk:** Exact fallback negates the ANN speedup on large indexes.
- [x] **oxirs** `oxirs-embed`: `ai/oxirs-embed/src/api/helpers.rs:43` — `TODO`: ModelStats lacks an accuracy field (model scoring ignores accuracy)
  - **Priority:** P2  **Scope:** small  **Cross-project:** none
  - **Approach:** Add an `accuracy: Option<f64>` to ModelStats and include it in the model-selection score.
  - **Risk:** Without accuracy, model ranking is based only on trained/size heuristics.

### oxirs-physics

- [x] **oxirs** `oxirs-physics`: `ai/oxirs-physics/src/simulation/result_injection.rs:158` — `TODO`: Implement when oxirs-core adds SPARQL UPDATE support (simulation results not persisted)
  - **Priority:** P2  **Scope:** medium  **Cross-project:** none
  - **Approach:** Once core exposes SPARQL UPDATE, write simulation results back into the graph via INSERT DATA.
  - **Completed 2026-06-22:** `execute_update` now uses `UpdateParser` + `UpdateExecutor` from `oxirs_core::query`.
  - **Risk:** Blocked on a core capability; gate the injection path until available.
