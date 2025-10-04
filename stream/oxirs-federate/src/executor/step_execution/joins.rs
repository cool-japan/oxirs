//! Join operations for federated queries

use anyhow::Result;
use tracing::{debug, warn};
use super::super::types::*;

pub fn perform_graphql_join(
    left: &GraphQLResponse,
    right: &GraphQLResponse,
) -> Result<GraphQLResponse> {
    debug!("Performing GraphQL join operation");

    // Combine errors from both responses
    let mut combined_errors = left.errors.clone();
    combined_errors.extend(right.errors.clone());

    // Merge extensions if they exist
    let combined_extensions = match (&left.extensions, &right.extensions) {
        (Some(left_ext), Some(right_ext)) => Some(merge_json_values(left_ext, right_ext)?),
        (Some(ext), None) | (None, Some(ext)) => Some(ext.clone()),
        (None, None) => None,
    };

    // Merge the data fields intelligently
    let merged_data = merge_graphql_data(&left.data, &right.data)?;

    Ok(GraphQLResponse {
        data: merged_data,
        errors: combined_errors,
        extensions: combined_extensions,
    })
}

/// Merge two GraphQL data values
fn merge_graphql_data(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> Result<serde_json::Value> {
    match (left, right) {
        // Both are objects - merge their fields
        (serde_json::Value::Object(left_obj), serde_json::Value::Object(right_obj)) => {
            let mut merged = left_obj.clone();

            for (key, right_value) in right_obj {
                match merged.get(key) {
                    Some(left_value) => {
                        // Field exists in both - recursively merge
                        let merged_value = merge_graphql_data(left_value, right_value)?;
                        merged.insert(key.clone(), merged_value);
                    }
                    None => {
                        // Field only exists in right - add it
                        merged.insert(key.clone(), right_value.clone());
                    }
                }
            }

            Ok(serde_json::Value::Object(merged))
        }

        // Both are arrays - concatenate them
        (serde_json::Value::Array(left_arr), serde_json::Value::Array(right_arr)) => {
            let mut merged = left_arr.clone();
            merged.extend(right_arr.clone());
            Ok(serde_json::Value::Array(merged))
        }

        // One is array, other is not - convert non-array to array and concatenate
        (serde_json::Value::Array(arr), value) | (value, serde_json::Value::Array(arr)) => {
            let mut merged = arr.clone();
            if !value.is_null() {
                merged.push(value.clone());
            }
            Ok(serde_json::Value::Array(merged))
        }

        // Both are null - return null
        (serde_json::Value::Null, serde_json::Value::Null) => Ok(serde_json::Value::Null),

        // One is null - return the non-null value
        (serde_json::Value::Null, value) | (value, serde_json::Value::Null) => Ok(value.clone()),

        // Different primitive types - prefer left value but warn
        (left_val, right_val) => {
            warn!(
                "GraphQL join: conflicting values for same field - using left value. Left: {:?}, Right: {:?}",
                left_val, right_val
            );
            Ok(left_val.clone())
        }
    }
}

/// Merge two JSON values generically
pub(crate) fn merge_json_values(
    left: &serde_json::Value,
    right: &serde_json::Value,
) -> Result<serde_json::Value> {
    match (left, right) {
        (serde_json::Value::Object(left_obj), serde_json::Value::Object(right_obj)) => {
            let mut merged = left_obj.clone();
            for (key, value) in right_obj {
                merged.insert(key.clone(), value.clone());
            }
            Ok(serde_json::Value::Object(merged))
        }
        (serde_json::Value::Array(left_arr), serde_json::Value::Array(right_arr)) => {
            let mut merged = left_arr.clone();
            merged.extend(right_arr.clone());
            Ok(serde_json::Value::Array(merged))
        }
        _ => Ok(left.clone()), // For non-container types, prefer left value
    }
}
