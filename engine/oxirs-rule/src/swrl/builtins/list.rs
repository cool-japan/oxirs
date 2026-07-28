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

    // Concatenate the item sequences of all input lists (comma-encoded literals);
    // an RDF-list resource argument fails loud via `extract_list_items`.
    let mut items: Vec<String> = Vec::new();
    for arg in &args[0..args.len() - 1] {
        items.extend(extract_list_items(arg)?);
    }

    let expected = extract_string_value(&args[args.len() - 1])?;
    Ok(items.join(",") == expected)
}

pub(crate) fn builtin_list_length(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("listLength requires exactly 2 arguments"));
    }

    let items = extract_list_items(&args[0])?;
    let length_val = extract_numeric_value(&args[1])? as usize;

    Ok(items.len() == length_val)
}

pub(crate) fn builtin_member(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("member requires exactly 2 arguments"));
    }

    let element = extract_string_value(&args[0])?;
    let items = extract_list_items(&args[1])?;

    Ok(items.contains(&element))
}

pub(crate) fn builtin_list_first(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("first requires exactly 2 arguments"));
    }

    let items = extract_list_items(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    match items.first() {
        Some(first) => Ok(*first == expected),
        None => Ok(expected.is_empty()),
    }
}

pub(crate) fn builtin_list_rest(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!("rest requires exactly 2 arguments"));
    }

    let items = extract_list_items(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    if items.len() <= 1 {
        Ok(expected.is_empty())
    } else {
        Ok(items[1..].join(",") == expected)
    }
}

pub(crate) fn builtin_list_nth(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("nth requires exactly 3 arguments"));
    }

    let items = extract_list_items(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

    // Get nth item from the list (0-indexed).
    match items.get(index) {
        Some(item) => Ok(*item == expected),
        None => Err(anyhow::anyhow!(
            "Index {} out of bounds for list of length {}",
            index,
            items.len()
        )),
    }
}

pub(crate) fn builtin_list_append(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!("append requires exactly 3 arguments"));
    }

    let mut items = extract_list_items(&args[0])?;
    let item = extract_string_value(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    // Append the item to the list. An empty comma-encoded literal ("") splits to
    // a single empty element; treat that as the empty list for append.
    if items == [String::new()] {
        items.clear();
    }
    items.push(item);
    Ok(items.join(",") == expected)
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

    let mut items = extract_list_items(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let item = extract_string_value(&args[2])?;
    let expected = extract_string_value(&args[3])?;

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

    let mut items = extract_list_items(&args[0])?;
    let index = extract_numeric_value(&args[1])? as usize;
    let expected = extract_string_value(&args[2])?;

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

    let mut items = extract_list_items(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    items.reverse();
    Ok(items.join(",") == expected)
}

pub(crate) fn builtin_list_sort(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 2 {
        return Err(anyhow::anyhow!(
            "listSort requires exactly 2 arguments: list, result"
        ));
    }

    let mut items = extract_list_items(&args[0])?;
    let expected = extract_string_value(&args[1])?;

    items.sort();
    Ok(items.join(",") == expected)
}

pub(crate) fn builtin_list_union(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listUnion requires exactly 3 arguments: list1, list2, result"
        ));
    }

    let items1 = extract_list_items(&args[0])?;
    let items2 = extract_list_items(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let mut union = items1.clone();
    for item in items2 {
        if !items1.contains(&item) {
            union.push(item);
        }
    }

    Ok(union.join(",") == expected)
}

pub(crate) fn builtin_list_intersection(args: &[SwrlArgument]) -> Result<bool> {
    if args.len() != 3 {
        return Err(anyhow::anyhow!(
            "listIntersection requires exactly 3 arguments: list1, list2, result"
        ));
    }

    let items1 = extract_list_items(&args[0])?;
    let items2 = extract_list_items(&args[1])?;
    let expected = extract_string_value(&args[2])?;

    let intersection: Vec<String> = items1
        .iter()
        .filter(|item| items2.contains(item))
        .cloned()
        .collect();

    Ok(intersection.join(",") == expected)
}
