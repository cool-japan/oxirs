//! # RequestIdGenerator - Trait Implementations
//!
//! This module contains trait implementations for `RequestIdGenerator`.
//!
//! ## Implemented Traits
//!
//! - `MakeRequestId`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use axum::{
    extract::{Path, Query, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Json, Response},
    routing::{delete, get, post},
    Router,
};
use tower_http::request_id::{MakeRequestId, RequestId};
use uuid::Uuid;

use super::types::RequestIdGenerator;

impl MakeRequestId for RequestIdGenerator {
    fn make_request_id<B>(&mut self, _request: &axum::http::Request<B>) -> Option<RequestId> {
        let request_id = Uuid::new_v4().to_string();
        axum::http::HeaderValue::from_str(&request_id)
            .ok()
            .map(RequestId::from)
    }
}
