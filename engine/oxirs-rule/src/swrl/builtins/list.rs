//! SWRL List Built-in Functions
//!
//! This module implements list operations for SWRL rules including:
//! - Construction: make_list
//! - Access: list_first, list_rest, list_nth
//! - Modification: list_append, list_insert, list_remove
//! - Transformation: list_reverse, list_sort
//! - Set operations: list_union, list_intersection, member
//! - Properties: list_length, list_concat

use anyhow::Result;

use super::super::types::SwrlArgument;
use super::utils::*;

pub(crate) fn builtin_list_concat(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() < 2 {
        return Err(anyhow::anyhow!("listConcat requires at least 2 arguments"));
    }

    // In a full implementation, this would handle RDF lists
    // For now, treat as string concatenation of comma-separated values
    let mut concat_result = String::new();
    for arg in &args[0..args.len() - 1] {
        if !concat_result.is_empty() {
            concat_result.push(',');
        }
        concat_result.push_str(&extract_string_value(arg)?);
    }

    let expected = extract_string_value(&args[args.len() - 1])?;
    Ok(concat_result == expected)
}

pub(crate) fn builtin_list_length(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("listLength requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let length_val = extract_numeric_value(&args[1])? as usize;

    // Simple implementation: count comma-separated items
    let items: Vec<&str> = list_str.split(',').collect();
    Ok(items.len() == length_val)
}

pub(crate) fn builtin_member(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("member requires exactly 2 arguments"));
    }

    let element = extract_string_value(&args[0])?;
    let list_str = extract_string_value(&args[1])?;

    // Simple implementation: check if element is in comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    Ok(items.contains(&element.as_str()))
}

pub(crate) fn builtin_list_first(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("first requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple implementation: get first item from comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    if items.is_empty() {
        Ok(expected.is_empty())
    } else {
        Ok(items[0] == expected)
    }
}

pub(crate) fn builtin_list_rest(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("rest requires exactly 2 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    // Simple implementation: get all but first item from comma-separated list
    let items: Vec<&str> = list_str.split(',').collect();
    if items.len() <= 1 {
        Ok(expected.is_empty())
    } else {
        let rest = items[1..].join(",");
        Ok(rest == expected)
    }
}

pub(crate) fn builtin_list_nth(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("nth requires exactly 3 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

    // Simple implementation: get nth item from comma-separated list (0-indexed)
    let items: Vec<&str> = list_str.split(',').collect();
    if index >= items.len() {
        return Err(anyhow::anyhow!(
            "Index {} out of bounds for list of length {}",
            index,
            items.len()
        ));
    }
    Ok(items[index] == expected)
}

pub(crate) fn builtin_list_append(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("append requires exactly 3 arguments"));
    }

    let list_str = extract_string_value(&args[0])?;
    let item = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    // Simple implementation: append item to comma-separated list
    let result = if list_str.is_empty() {
        item
    } else {
        format!("{list_str},{item}")
    };
    Ok(result == expected)
}

pub(crate) fn builtin_make_list(args: &[SwrlArgument]) -> Result<bool> {
    if args.is_empty() {
        return Err(anyhow::anyhow!("makeList requires at least 1 argument"));
    }

    let values: Result<Vec<String>> = args[..args.len() - 1]
        .iter()
        .map(extract_string_value)
        .collect();
    let values = values?;
    let expected = extract_string_value(&args[args.len() - 1])?;

    let result = values.join(",");
    Ok(result == expected)
}

pub(crate) fn builtin_list_insert(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 4 {
        return Err(anyhow::anyhow!(
            "listInsert requires exactly 4 arguments: list, index, item, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let item = extract_string_value(&args[2])?;
    let expected = extract_string_value(&args[3])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    if index <= items.len() {
        items.insert(index, item);
        let result = items.join(",");
        Ok(result == expected)
    } else {
        Ok(false)
    }
}

pub(crate) fn builtin_list_remove(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listRemove requires exactly 3 arguments: list, index, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    if index < items.len() {
        items.remove(index);
        let result = items.join(",");
        Ok(result == expected)
    } else {
        Ok(false)
    }
}

pub(crate) fn builtin_list_reverse(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "listReverse requires exactly 2 arguments: list, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    items.reverse();
    let result = items.join(",");

    Ok(result == expected)
}

pub(crate) fn builtin_list_sort(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "listSort requires exactly 2 arguments: list, result"
        ));
    }

    let list_str = extract_string_value(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    let mut items: Vec<String> = list_str.split(',').map(|s| s.to_string()).collect();
    items.sort();
    let result = items.join(",");

    Ok(result == expected)
}

pub(crate) fn builtin_list_union(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listUnion requires exactly 3 arguments: list1, list2, result"
        ));
    }

    let list1_str = extract_string_value(&args[0])?;
    let list2_str = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let items1: Vec<&str> = list1_str.split(',').collect();
    let items2: Vec<&str> = list2_str.split(',').collect();

    let mut union: Vec<String> = items1.iter().map(|&s| s.to_string()).collect();
    for item in items2 {
        if !items1.contains(&item) {
            union.push(item.to_string());
        }
    }

    let result = union.join(",");
    Ok(result == expected)
}

pub(crate) fn builtin_list_intersection(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listIntersection requires exactly 3 arguments: list1, list2, result"
        ));
    }

    let list1_str = extract_string_value(&args[0])?;
    let list2_str = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let items1: Vec<&str> = list1_str.split(',').collect();
    let items2: Vec<&str> = list2_str.split(',').collect();

    let intersection: Vec<String> = items1
        .iter()
        .filter(|&&item| items2.contains(&item))
        .map(|&s| s.to_string())
        .collect();

    let result = intersection.join(",");
    Ok(result == expected)
}
