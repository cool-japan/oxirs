//! # Federated Query Optimizer — Test Suite
//!
//! Activates the federated query optimizer test suite. The tests themselves
//! live in [`federated_query_tests`](crate::federated_query_tests) and exercise
//! the public APIs of the sibling `federated_query_*` implementation modules.

#![cfg(test)]

#[path = "federated_query_tests.rs"]
mod federated_query_tests;
