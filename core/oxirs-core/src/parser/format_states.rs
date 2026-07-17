//! Retired: hand-rolled line-based Turtle/TriG state machines.
//!
//! This module used to implement `TurtleParserState`/`TrigParserState`, a
//! per-line statement splitter used by [`crate::parser::Parser`] for
//! `RdfFormat::Turtle` and `RdfFormat::TriG`. It was unfixable piecemeal:
//! splitting on raw `;`/`,` before any quote-aware tokenization corrupted
//! literals containing those characters, and `data.lines()` processing
//! collapsed newlines inside triple-quoted string literals into spaces.
//! It also never implemented comma object lists, RDF collections (`( ... )`),
//! or blank-node property lists (`[ ... ]`) -- all core Turtle/TriG syntax.
//!
//! `Parser::parse_turtle`/`Parser::parse_trig` in `parser/mod.rs` now
//! delegate to the real, spec-compliant, oxttl-backed grammar via
//! [`crate::format::RdfParser`] instead, so this module is no longer
//! referenced anywhere (`mod format_states;` was removed from
//! `parser/mod.rs`) and is kept only as a documented historical marker.
