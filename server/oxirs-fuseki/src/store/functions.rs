//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::config::DatasetConfig;
use crate::error::{FusekiError, FusekiResult};
use oxirs_core::model::*;
use oxirs_core::parser::{Parser, RdfFormat as CoreRdfFormat};
use oxirs_core::query::{QueryEngine, QueryResult as CoreQueryResult};
use oxirs_core::serializer::Serializer;
use oxirs_core::{RdfStore, Store as CoreStore};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
/// Type alias for dataset storage mapping
type DatasetMap = Arc<RwLock<HashMap<String, Arc<RwLock<dyn CoreStore>>>>>;
