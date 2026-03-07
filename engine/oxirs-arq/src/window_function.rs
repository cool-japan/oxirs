// SPARQL-style window functions (ROW_NUMBER, RANK, DENSE_RANK, NTILE, LAG, LEAD, CumSum, CumCount)
// Added in v1.1.0 Round 7

use std::collections::HashMap;

/// A value that can appear in a window function computation.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowValue {
    Integer(i64),
    Float(f64),
    Text(String),
    Null,
}

impl PartialOrd for WindowValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp_window(other))
    }
}

impl WindowValue {
    fn cmp_window(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        match (self, other) {
            (WindowValue::Null, WindowValue::Null) => Ordering::Equal,
            (WindowValue::Null, _) => Ordering::Less,
            (_, WindowValue::Null) => Ordering::Greater,
            (WindowValue::Integer(a), WindowValue::Integer(b)) => a.cmp(b),
            (WindowValue::Float(a), WindowValue::Float(b)) => {
                a.partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (WindowValue::Integer(a), WindowValue::Float(b)) => {
                (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal)
            }
            (WindowValue::Float(a), WindowValue::Integer(b)) => {
                a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal)
            }
            (WindowValue::Text(a), WindowValue::Text(b)) => a.cmp(b),
            (WindowValue::Integer(_) | WindowValue::Float(_), WindowValue::Text(_)) => {
                Ordering::Less
            }
            (WindowValue::Text(_), WindowValue::Integer(_) | WindowValue::Float(_)) => {
                Ordering::Greater
            }
        }
    }

    /// Try to extract a numeric f64 value.
    #[allow(dead_code)]
    fn as_f64(&self) -> Option<f64> {
        match self {
            WindowValue::Integer(i) => Some(*i as f64),
            WindowValue::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// A single row in a window computation, mapping column names to values.
#[derive(Debug, Clone)]
pub struct WindowRow {
    pub values: HashMap<String, WindowValue>,
}

impl WindowRow {
    pub fn new(values: HashMap<String, WindowValue>) -> Self {
        Self { values }
    }

    pub fn get(&self, column: &str) -> Option<&WindowValue> {
        self.values.get(column)
    }

    pub fn set(&mut self, column: String, value: WindowValue) {
        self.values.insert(column, value);
    }
}

/// Window function variants.
#[derive(Debug, Clone)]
pub enum WindowFunc {
    RowNumber,
    Rank,
    DenseRank,
    Ntile(usize),
    Lag {
        column: String,
        offset: usize,
        default: Option<WindowValue>,
    },
    Lead {
        column: String,
        offset: usize,
        default: Option<WindowValue>,
    },
    CumSum {
        column: String,
    },
    CumCount,
}

/// Full window specification: partition + order + function.
#[derive(Debug, Clone)]
pub struct WindowSpec {
    pub partition_by: Vec<String>,
    pub order_by: Vec<(String, bool)>, // (column_name, ascending)
    pub func: WindowFunc,
    pub output_column: String,
}

/// Errors that can occur during window function application.
#[derive(Debug)]
pub enum WindowError {
    ColumnNotFound(String),
    InvalidNtile(String),
    InvalidOffset(String),
    NonNumericColumn(String),
}

impl std::fmt::Display for WindowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::ColumnNotFound(c) => write!(f, "Column not found: {c}"),
            WindowError::InvalidNtile(m) => write!(f, "Invalid NTILE: {m}"),
            WindowError::InvalidOffset(m) => write!(f, "Invalid offset: {m}"),
            WindowError::NonNumericColumn(c) => write!(f, "Non-numeric column: {c}"),
        }
    }
}

impl std::error::Error for WindowError {}

/// Compute a partition key string for a row.
fn partition_key(row: &WindowRow, partition_by: &[String]) -> String {
    partition_by
        .iter()
        .map(|col| match row.values.get(col) {
            Some(WindowValue::Integer(i)) => format!("i:{i}"),
            Some(WindowValue::Float(f)) => format!("f:{f}"),
            Some(WindowValue::Text(t)) => format!("t:{t}"),
            Some(WindowValue::Null) | None => "null".to_string(),
        })
        .collect::<Vec<_>>()
        .join("|")
}

/// Compare two rows by the order_by specification.
fn cmp_rows(a: &WindowRow, b: &WindowRow, order_by: &[(String, bool)]) -> std::cmp::Ordering {
    for (col, ascending) in order_by {
        let av = a.values.get(col).unwrap_or(&WindowValue::Null);
        let bv = b.values.get(col).unwrap_or(&WindowValue::Null);
        let ord = av.cmp_window(bv);
        if ord != std::cmp::Ordering::Equal {
            return if *ascending { ord } else { ord.reverse() };
        }
    }
    std::cmp::Ordering::Equal
}

/// Apply a single window function over all rows, returning (row, computed_value) pairs.
pub fn apply_window(
    rows: &[WindowRow],
    spec: &WindowSpec,
) -> Result<Vec<(WindowRow, WindowValue)>, WindowError> {
    if rows.is_empty() {
        return Ok(vec![]);
    }

    // Collect original indices grouped by partition key
    let mut partitions: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, row) in rows.iter().enumerate() {
        let key = partition_key(row, &spec.partition_by);
        partitions.entry(key).or_default().push(idx);
    }

    // Sort indices within each partition by order_by
    for indices in partitions.values_mut() {
        indices.sort_by(|&a, &b| cmp_rows(&rows[a], &rows[b], &spec.order_by));
    }

    let mut results: Vec<Option<WindowValue>> = vec![None; rows.len()];

    for indices in partitions.values() {
        compute_partition_values(rows, indices, &spec.func, &spec.order_by, &mut results)?;
    }

    Ok(rows
        .iter()
        .enumerate()
        .map(|(i, row)| (row.clone(), results[i].take().unwrap_or(WindowValue::Null)))
        .collect())
}

fn compute_partition_values(
    rows: &[WindowRow],
    sorted_indices: &[usize],
    func: &WindowFunc,
    order_by: &[(String, bool)],
    results: &mut [Option<WindowValue>],
) -> Result<(), WindowError> {
    let n = sorted_indices.len();
    match func {
        WindowFunc::RowNumber => {
            for (rank, &orig_idx) in sorted_indices.iter().enumerate() {
                results[orig_idx] = Some(WindowValue::Integer((rank + 1) as i64));
            }
        }
        WindowFunc::Rank => {
            let mut current_rank = 1usize;
            let mut i = 0;
            while i < n {
                // Find the extent of the tie group
                let mut j = i + 1;
                while j < n
                    && cmp_rows(&rows[sorted_indices[i]], &rows[sorted_indices[j]], order_by)
                        == std::cmp::Ordering::Equal
                {
                    j += 1;
                }
                // All rows from i..j get current_rank
                for &orig_idx in &sorted_indices[i..j] {
                    results[orig_idx] = Some(WindowValue::Integer(current_rank as i64));
                }
                current_rank += j - i; // skip ranks for ties
                i = j;
            }
        }
        WindowFunc::DenseRank => {
            let mut current_rank = 1usize;
            let mut i = 0;
            while i < n {
                let mut j = i + 1;
                while j < n
                    && cmp_rows(&rows[sorted_indices[i]], &rows[sorted_indices[j]], order_by)
                        == std::cmp::Ordering::Equal
                {
                    j += 1;
                }
                for &orig_idx in &sorted_indices[i..j] {
                    results[orig_idx] = Some(WindowValue::Integer(current_rank as i64));
                }
                current_rank += 1; // dense: no gaps
                i = j;
            }
        }
        WindowFunc::Ntile(buckets) => {
            if *buckets == 0 {
                return Err(WindowError::InvalidNtile(
                    "NTILE bucket count must be > 0".to_string(),
                ));
            }
            let total = n;
            let buckets = *buckets;
            for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
                // Standard NTILE formula
                let bucket = (pos * buckets / total) + 1;
                results[orig_idx] = Some(WindowValue::Integer(bucket as i64));
            }
        }
        WindowFunc::Lag {
            column,
            offset,
            default,
        } => {
            for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
                let value = if pos >= *offset {
                    let source_idx = sorted_indices[pos - offset];
                    rows[source_idx]
                        .values
                        .get(column)
                        .cloned()
                        .unwrap_or(WindowValue::Null)
                } else {
                    default.clone().unwrap_or(WindowValue::Null)
                };
                results[orig_idx] = Some(value);
            }
        }
        WindowFunc::Lead {
            column,
            offset,
            default,
        } => {
            for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
                let value = if pos + offset < n {
                    let source_idx = sorted_indices[pos + offset];
                    rows[source_idx]
                        .values
                        .get(column)
                        .cloned()
                        .unwrap_or(WindowValue::Null)
                } else {
                    default.clone().unwrap_or(WindowValue::Null)
                };
                results[orig_idx] = Some(value);
            }
        }
        WindowFunc::CumSum { column } => {
            let mut running_sum = 0.0f64;
            let mut running_int: Option<i64> = Some(0);
            for &orig_idx in sorted_indices.iter() {
                let val = rows[orig_idx]
                    .values
                    .get(column)
                    .unwrap_or(&WindowValue::Null);
                match val {
                    WindowValue::Integer(i) => {
                        running_sum += *i as f64;
                        running_int = running_int.and_then(|s| s.checked_add(*i));
                    }
                    WindowValue::Float(f) => {
                        running_sum += f;
                        running_int = None;
                    }
                    WindowValue::Null => {
                        // NULLs contribute 0
                    }
                    WindowValue::Text(_) => {
                        return Err(WindowError::NonNumericColumn(column.clone()));
                    }
                }
                let result_val = if let Some(int_sum) = running_int {
                    WindowValue::Integer(int_sum)
                } else {
                    WindowValue::Float(running_sum)
                };
                results[orig_idx] = Some(result_val);
            }
        }
        WindowFunc::CumCount => {
            for (pos, &orig_idx) in sorted_indices.iter().enumerate() {
                results[orig_idx] = Some(WindowValue::Integer((pos + 1) as i64));
            }
        }
    }
    Ok(())
}

/// Apply multiple window functions, adding their output columns to each row.
pub fn apply_windows(
    mut rows: Vec<WindowRow>,
    specs: &[WindowSpec],
) -> Result<Vec<WindowRow>, WindowError> {
    for spec in specs {
        let result_pairs = apply_window(&rows, spec)?;
        for (idx, (_row, value)) in result_pairs.into_iter().enumerate() {
            rows[idx].values.insert(spec.output_column.clone(), value);
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(pairs: &[(&str, WindowValue)]) -> WindowRow {
        let mut values = HashMap::new();
        for (k, v) in pairs {
            values.insert(k.to_string(), v.clone());
        }
        WindowRow { values }
    }

    fn int_spec(func: WindowFunc, order_col: &str, out: &str) -> WindowSpec {
        WindowSpec {
            partition_by: vec![],
            order_by: vec![(order_col.to_string(), true)],
            func,
            output_column: out.to_string(),
        }
    }

    fn partitioned_spec(
        func: WindowFunc,
        partition_col: &str,
        order_col: &str,
        out: &str,
    ) -> WindowSpec {
        WindowSpec {
            partition_by: vec![partition_col.to_string()],
            order_by: vec![(order_col.to_string(), true)],
            func,
            output_column: out.to_string(),
        }
    }

    // ---- ROW_NUMBER ----

    #[test]
    fn test_row_number_sequential() {
        let rows = vec![
            make_row(&[("n", WindowValue::Integer(3))]),
            make_row(&[("n", WindowValue::Integer(1))]),
            make_row(&[("n", WindowValue::Integer(2))]),
        ];
        let spec = int_spec(WindowFunc::RowNumber, "n", "rn");
        let result = apply_window(&rows, &spec).unwrap();
        // Collect (n_value, rn_value)
        let pairs: Vec<(i64, i64)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["n"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                let rn = match val {
                    WindowValue::Integer(i) => *i,
                    _ => 0,
                };
                (n, rn)
            })
            .collect();
        // Sorted by n: 1->1, 2->2, 3->3
        for (n, rn) in &pairs {
            assert_eq!(*rn, *n, "Expected rn == n for n={n}");
        }
        let mut nums: Vec<i64> = pairs.iter().map(|(_, rn)| *rn).collect();
        nums.sort();
        assert_eq!(nums, vec![1, 2, 3]);
    }

    #[test]
    fn test_row_number_empty() {
        let rows: Vec<WindowRow> = vec![];
        let spec = int_spec(WindowFunc::RowNumber, "n", "rn");
        let result = apply_window(&rows, &spec).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_row_number_single() {
        let rows = vec![make_row(&[("n", WindowValue::Integer(42))])];
        let spec = int_spec(WindowFunc::RowNumber, "n", "rn");
        let result = apply_window(&rows, &spec).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1, WindowValue::Integer(1));
    }

    // ---- RANK ----

    #[test]
    fn test_rank_no_ties() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(20))]),
            make_row(&[("v", WindowValue::Integer(30))]),
        ];
        let spec = int_spec(WindowFunc::Rank, "v", "r");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 2, 3]);
    }

    #[test]
    fn test_rank_with_ties() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(20))]),
        ];
        let spec = int_spec(WindowFunc::Rank, "v", "r");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        // Two tied at rank 1, then gap → rank 3
        assert_eq!(ranks, vec![1, 1, 3]);
    }

    #[test]
    fn test_rank_all_tied() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(5))]),
            make_row(&[("v", WindowValue::Integer(5))]),
            make_row(&[("v", WindowValue::Integer(5))]),
        ];
        let spec = int_spec(WindowFunc::Rank, "v", "r");
        let result = apply_window(&rows, &spec).unwrap();
        for (_, val) in &result {
            assert_eq!(*val, WindowValue::Integer(1));
        }
    }

    // ---- DENSE_RANK ----

    #[test]
    fn test_dense_rank_with_ties() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(20))]),
        ];
        let spec = int_spec(WindowFunc::DenseRank, "v", "dr");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        // No gaps: 1, 1, 2
        assert_eq!(ranks, vec![1, 1, 2]);
    }

    #[test]
    fn test_dense_rank_no_ties() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(1))]),
            make_row(&[("v", WindowValue::Integer(2))]),
            make_row(&[("v", WindowValue::Integer(3))]),
        ];
        let spec = int_spec(WindowFunc::DenseRank, "v", "dr");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 2, 3]);
    }

    #[test]
    fn test_dense_rank_multiple_ties() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(10))]),
            make_row(&[("v", WindowValue::Integer(20))]),
            make_row(&[("v", WindowValue::Integer(20))]),
            make_row(&[("v", WindowValue::Integer(30))]),
        ];
        let spec = int_spec(WindowFunc::DenseRank, "v", "dr");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 1, 2, 2, 3]);
    }

    // ---- NTILE ----

    #[test]
    fn test_ntile_4_even() {
        let rows: Vec<WindowRow> = (1i64..=8)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(WindowFunc::Ntile(4), "v", "nt");
        let result = apply_window(&rows, &spec).unwrap();
        let mut buckets: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        buckets.sort();
        // 8 rows, 4 buckets → 2 per bucket: 1,1,2,2,3,3,4,4
        assert_eq!(buckets, vec![1, 1, 2, 2, 3, 3, 4, 4]);
    }

    #[test]
    fn test_ntile_3_uneven() {
        let rows: Vec<WindowRow> = (1i64..=7)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(WindowFunc::Ntile(3), "v", "nt");
        let result = apply_window(&rows, &spec).unwrap();
        let mut buckets: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        buckets.sort();
        assert_eq!(buckets.len(), 7);
        // All values 1, 2, or 3
        for b in &buckets {
            assert!(*b >= 1 && *b <= 3, "bucket out of range: {b}");
        }
    }

    #[test]
    fn test_ntile_zero_error() {
        let rows = vec![make_row(&[("v", WindowValue::Integer(1))])];
        let spec = int_spec(WindowFunc::Ntile(0), "v", "nt");
        assert!(matches!(
            apply_window(&rows, &spec),
            Err(WindowError::InvalidNtile(_))
        ));
    }

    #[test]
    fn test_ntile_1() {
        let rows: Vec<WindowRow> = (1i64..=5)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(WindowFunc::Ntile(1), "v", "nt");
        let result = apply_window(&rows, &spec).unwrap();
        for (_, val) in &result {
            assert_eq!(*val, WindowValue::Integer(1));
        }
    }

    // ---- LAG ----

    #[test]
    fn test_lag_offset_1_no_default() {
        let rows: Vec<WindowRow> = (1i64..=4)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lag {
                column: "v".to_string(),
                offset: 1,
                default: None,
            },
            "v",
            "lag",
        );
        let result = apply_window(&rows, &spec).unwrap();
        // Sorted by v: 1,2,3,4. LAG(1): null,1,2,3
        let mut sorted_pairs: Vec<(i64, &WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (n, val)
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        assert_eq!(sorted_pairs[0].1, &WindowValue::Null);
        assert_eq!(sorted_pairs[1].1, &WindowValue::Integer(1));
        assert_eq!(sorted_pairs[2].1, &WindowValue::Integer(2));
        assert_eq!(sorted_pairs[3].1, &WindowValue::Integer(3));
    }

    #[test]
    fn test_lag_with_default() {
        let rows: Vec<WindowRow> = (1i64..=3)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lag {
                column: "v".to_string(),
                offset: 1,
                default: Some(WindowValue::Integer(-1)),
            },
            "v",
            "lag",
        );
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (n, val.clone())
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        assert_eq!(sorted_pairs[0].1, WindowValue::Integer(-1));
        assert_eq!(sorted_pairs[1].1, WindowValue::Integer(1));
        assert_eq!(sorted_pairs[2].1, WindowValue::Integer(2));
    }

    #[test]
    fn test_lag_offset_2() {
        let rows: Vec<WindowRow> = (1i64..=5)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lag {
                column: "v".to_string(),
                offset: 2,
                default: Some(WindowValue::Integer(0)),
            },
            "v",
            "lag",
        );
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (n, val.clone())
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        assert_eq!(sorted_pairs[0].1, WindowValue::Integer(0)); // default
        assert_eq!(sorted_pairs[1].1, WindowValue::Integer(0)); // default
        assert_eq!(sorted_pairs[2].1, WindowValue::Integer(1));
        assert_eq!(sorted_pairs[3].1, WindowValue::Integer(2));
        assert_eq!(sorted_pairs[4].1, WindowValue::Integer(3));
    }

    // ---- LEAD ----

    #[test]
    fn test_lead_offset_1_no_default() {
        let rows: Vec<WindowRow> = (1i64..=4)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lead {
                column: "v".to_string(),
                offset: 1,
                default: None,
            },
            "v",
            "lead",
        );
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (n, val.clone())
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        assert_eq!(sorted_pairs[0].1, WindowValue::Integer(2));
        assert_eq!(sorted_pairs[1].1, WindowValue::Integer(3));
        assert_eq!(sorted_pairs[2].1, WindowValue::Integer(4));
        assert_eq!(sorted_pairs[3].1, WindowValue::Null);
    }

    #[test]
    fn test_lead_with_default() {
        let rows: Vec<WindowRow> = (1i64..=3)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lead {
                column: "v".to_string(),
                offset: 1,
                default: Some(WindowValue::Integer(99)),
            },
            "v",
            "lead",
        );
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (n, val.clone())
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        assert_eq!(sorted_pairs[2].1, WindowValue::Integer(99));
    }

    // ---- CumSum ----

    #[test]
    fn test_cum_sum_integers() {
        let rows: Vec<WindowRow> = (1i64..=5)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::CumSum {
                column: "v".to_string(),
            },
            "v",
            "cs",
        );
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (n, val.clone())
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        assert_eq!(sorted_pairs[0].1, WindowValue::Integer(1));
        assert_eq!(sorted_pairs[1].1, WindowValue::Integer(3));
        assert_eq!(sorted_pairs[2].1, WindowValue::Integer(6));
        assert_eq!(sorted_pairs[3].1, WindowValue::Integer(10));
        assert_eq!(sorted_pairs[4].1, WindowValue::Integer(15));
    }

    #[test]
    fn test_cum_sum_floats() {
        let rows = vec![
            make_row(&[
                ("v", WindowValue::Float(1.5)),
                ("k", WindowValue::Integer(1)),
            ]),
            make_row(&[
                ("v", WindowValue::Float(2.5)),
                ("k", WindowValue::Integer(2)),
            ]),
            make_row(&[
                ("v", WindowValue::Float(1.0)),
                ("k", WindowValue::Integer(3)),
            ]),
        ];
        let spec = WindowSpec {
            partition_by: vec![],
            order_by: vec![("k".to_string(), true)],
            func: WindowFunc::CumSum {
                column: "v".to_string(),
            },
            output_column: "cs".to_string(),
        };
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, f64)> = result
            .iter()
            .map(|(row, val)| {
                let k = match row.values["k"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                let cs = match val {
                    WindowValue::Float(f) => *f,
                    WindowValue::Integer(i) => *i as f64,
                    _ => 0.0,
                };
                (k, cs)
            })
            .collect();
        sorted_pairs.sort_by_key(|(k, _)| *k);
        assert!((sorted_pairs[0].1 - 1.5).abs() < 1e-10);
        assert!((sorted_pairs[1].1 - 4.0).abs() < 1e-10);
        assert!((sorted_pairs[2].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_cum_sum_text_error() {
        let rows = vec![make_row(&[("v", WindowValue::Text("hello".to_string()))])];
        let spec = int_spec(
            WindowFunc::CumSum {
                column: "v".to_string(),
            },
            "v",
            "cs",
        );
        assert!(matches!(
            apply_window(&rows, &spec),
            Err(WindowError::NonNumericColumn(_))
        ));
    }

    // ---- CumCount ----

    #[test]
    fn test_cum_count() {
        let rows: Vec<WindowRow> = (1i64..=5)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(WindowFunc::CumCount, "v", "cc");
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted_pairs: Vec<(i64, i64)> = result
            .iter()
            .map(|(row, val)| {
                let n = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                let cc = match val {
                    WindowValue::Integer(i) => *i,
                    _ => 0,
                };
                (n, cc)
            })
            .collect();
        sorted_pairs.sort_by_key(|(n, _)| *n);
        for (i, (_, cc)) in sorted_pairs.iter().enumerate() {
            assert_eq!(*cc, (i + 1) as i64);
        }
    }

    // ---- Partitioned ----

    #[test]
    fn test_row_number_partitioned() {
        let rows = vec![
            make_row(&[
                ("g", WindowValue::Text("A".into())),
                ("v", WindowValue::Integer(2)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("A".into())),
                ("v", WindowValue::Integer(1)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("B".into())),
                ("v", WindowValue::Integer(3)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("B".into())),
                ("v", WindowValue::Integer(1)),
            ]),
        ];
        let spec = partitioned_spec(WindowFunc::RowNumber, "g", "v", "rn");
        let result = apply_window(&rows, &spec).unwrap();
        let mut pairs: Vec<(String, i64, i64)> = result
            .iter()
            .map(|(row, val)| {
                let g = match &row.values["g"] {
                    WindowValue::Text(s) => s.clone(),
                    _ => String::new(),
                };
                let v = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                let rn = match val {
                    WindowValue::Integer(i) => *i,
                    _ => 0,
                };
                (g, v, rn)
            })
            .collect();
        pairs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        // Group A: v=1->rn=1, v=2->rn=2
        let group_a: Vec<_> = pairs.iter().filter(|(g, _, _)| g == "A").collect();
        assert_eq!(group_a.len(), 2);
        assert_eq!(group_a[0].2, 1);
        assert_eq!(group_a[1].2, 2);
        // Group B: v=1->rn=1, v=3->rn=2
        let group_b: Vec<_> = pairs.iter().filter(|(g, _, _)| g == "B").collect();
        assert_eq!(group_b.len(), 2);
        assert_eq!(group_b[0].2, 1);
        assert_eq!(group_b[1].2, 2);
    }

    #[test]
    fn test_rank_partitioned_with_ties() {
        let rows = vec![
            make_row(&[
                ("g", WindowValue::Text("A".into())),
                ("v", WindowValue::Integer(5)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("A".into())),
                ("v", WindowValue::Integer(5)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("A".into())),
                ("v", WindowValue::Integer(10)),
            ]),
        ];
        let spec = partitioned_spec(WindowFunc::Rank, "g", "v", "r");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 1, 3]);
    }

    #[test]
    fn test_dense_rank_partitioned() {
        let rows = vec![
            make_row(&[
                ("g", WindowValue::Text("X".into())),
                ("v", WindowValue::Integer(5)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("X".into())),
                ("v", WindowValue::Integer(5)),
            ]),
            make_row(&[
                ("g", WindowValue::Text("X".into())),
                ("v", WindowValue::Integer(10)),
            ]),
        ];
        let spec = partitioned_spec(WindowFunc::DenseRank, "g", "v", "dr");
        let result = apply_window(&rows, &spec).unwrap();
        let mut ranks: Vec<i64> = result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        ranks.sort();
        assert_eq!(ranks, vec![1, 1, 2]);
    }

    // ---- ORDER_BY descending ----

    #[test]
    fn test_row_number_descending() {
        let rows: Vec<WindowRow> = (1i64..=4)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = WindowSpec {
            partition_by: vec![],
            order_by: vec![("v".to_string(), false)], // descending
            func: WindowFunc::RowNumber,
            output_column: "rn".to_string(),
        };
        let result = apply_window(&rows, &spec).unwrap();
        let mut pairs: Vec<(i64, i64)> = result
            .iter()
            .map(|(row, val)| {
                let v = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                let rn = match val {
                    WindowValue::Integer(i) => *i,
                    _ => 0,
                };
                (v, rn)
            })
            .collect();
        pairs.sort_by_key(|(v, _)| *v);
        // v=1 gets rn=4 (last in descending order), v=4 gets rn=1
        assert_eq!(pairs[0].1, 4); // v=1
        assert_eq!(pairs[3].1, 1); // v=4
    }

    #[test]
    fn test_rank_descending() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(30))]),
            make_row(&[("v", WindowValue::Integer(20))]),
            make_row(&[("v", WindowValue::Integer(10))]),
        ];
        let spec = WindowSpec {
            partition_by: vec![],
            order_by: vec![("v".to_string(), false)],
            func: WindowFunc::Rank,
            output_column: "r".to_string(),
        };
        let result = apply_window(&rows, &spec).unwrap();
        let mut pairs: Vec<(i64, i64)> = result
            .iter()
            .map(|(row, val)| {
                let v = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                let r = match val {
                    WindowValue::Integer(i) => *i,
                    _ => 0,
                };
                (v, r)
            })
            .collect();
        pairs.sort_by_key(|(v, _)| *v);
        // desc order: 30->1, 20->2, 10->3
        assert_eq!(pairs[0].1, 3); // v=10 gets rank 3
        assert_eq!(pairs[2].1, 1); // v=30 gets rank 1
    }

    // ---- apply_windows (multiple) ----

    #[test]
    fn test_apply_windows_multiple_specs() {
        let rows: Vec<WindowRow> = (1i64..=3)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let specs = vec![
            int_spec(WindowFunc::RowNumber, "v", "rn"),
            int_spec(
                WindowFunc::CumSum {
                    column: "v".to_string(),
                },
                "v",
                "cs",
            ),
        ];
        let result = apply_windows(rows, &specs).unwrap();
        assert_eq!(result.len(), 3);
        for row in &result {
            assert!(row.values.contains_key("rn"));
            assert!(row.values.contains_key("cs"));
        }
    }

    #[test]
    fn test_apply_windows_empty() {
        let rows: Vec<WindowRow> = vec![];
        let specs = vec![int_spec(WindowFunc::RowNumber, "v", "rn")];
        let result = apply_windows(rows, &specs).unwrap();
        assert!(result.is_empty());
    }

    // ---- WindowValue ordering ----

    #[test]
    fn test_window_value_null_smallest() {
        assert_eq!(
            WindowValue::Null.cmp_window(&WindowValue::Integer(0)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            WindowValue::Integer(0).cmp_window(&WindowValue::Null),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            WindowValue::Null.cmp_window(&WindowValue::Null),
            std::cmp::Ordering::Equal
        );
    }

    #[test]
    fn test_window_value_int_vs_float() {
        assert_eq!(
            WindowValue::Integer(1).cmp_window(&WindowValue::Float(2.0)),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            WindowValue::Float(1.5).cmp_window(&WindowValue::Integer(1)),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn test_window_value_text_ordering() {
        assert_eq!(
            WindowValue::Text("apple".into()).cmp_window(&WindowValue::Text("banana".into())),
            std::cmp::Ordering::Less
        );
    }

    // ---- Null handling in data ----

    #[test]
    fn test_cum_sum_with_nulls() {
        let rows = vec![
            make_row(&[
                ("v", WindowValue::Integer(1)),
                ("k", WindowValue::Integer(1)),
            ]),
            make_row(&[("v", WindowValue::Null), ("k", WindowValue::Integer(2))]),
            make_row(&[
                ("v", WindowValue::Integer(3)),
                ("k", WindowValue::Integer(3)),
            ]),
        ];
        let spec = WindowSpec {
            partition_by: vec![],
            order_by: vec![("k".to_string(), true)],
            func: WindowFunc::CumSum {
                column: "v".to_string(),
            },
            output_column: "cs".to_string(),
        };
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let k = match row.values["k"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (k, val.clone())
            })
            .collect();
        sorted.sort_by_key(|(k, _)| *k);
        assert_eq!(sorted[0].1, WindowValue::Integer(1));
        assert_eq!(sorted[1].1, WindowValue::Integer(1)); // null contributes 0
        assert_eq!(sorted[2].1, WindowValue::Integer(4));
    }

    #[test]
    fn test_lag_empty_partition() {
        let rows: Vec<WindowRow> = vec![];
        let spec = int_spec(
            WindowFunc::Lag {
                column: "v".to_string(),
                offset: 1,
                default: None,
            },
            "v",
            "lag",
        );
        let result = apply_window(&rows, &spec).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lead_large_offset() {
        let rows: Vec<WindowRow> = (1i64..=3)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lead {
                column: "v".to_string(),
                offset: 10,
                default: Some(WindowValue::Integer(-99)),
            },
            "v",
            "lead",
        );
        let result = apply_window(&rows, &spec).unwrap();
        for (_, val) in &result {
            assert_eq!(*val, WindowValue::Integer(-99));
        }
    }

    #[test]
    fn test_ntile_more_buckets_than_rows() {
        let rows = vec![
            make_row(&[("v", WindowValue::Integer(1))]),
            make_row(&[("v", WindowValue::Integer(2))]),
        ];
        let spec = int_spec(WindowFunc::Ntile(5), "v", "nt");
        let result = apply_window(&rows, &spec).unwrap();
        // 2 rows, 5 buckets: each row gets a different bucket (1 and 2 roughly)
        assert_eq!(result.len(), 2);
        for (_, val) in &result {
            match val {
                WindowValue::Integer(b) => assert!(*b >= 1 && *b <= 5, "bucket out of range: {b}"),
                _ => panic!("Expected integer bucket"),
            }
        }
    }

    #[test]
    fn test_as_f64() {
        assert_eq!(WindowValue::Integer(5).as_f64(), Some(5.0));
        assert_eq!(WindowValue::Float(2.71).as_f64(), Some(2.71));
        assert_eq!(WindowValue::Text("x".into()).as_f64(), None);
        assert_eq!(WindowValue::Null.as_f64(), None);
    }

    #[test]
    fn test_window_row_get_set() {
        let mut row = make_row(&[("a", WindowValue::Integer(1))]);
        assert_eq!(row.get("a"), Some(&WindowValue::Integer(1)));
        assert_eq!(row.get("b"), None);
        row.set("b".to_string(), WindowValue::Float(2.0));
        assert_eq!(row.get("b"), Some(&WindowValue::Float(2.0)));
    }

    #[test]
    fn test_cum_count_matches_row_number() {
        let rows: Vec<WindowRow> = (1i64..=6)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let rn_spec = int_spec(WindowFunc::RowNumber, "v", "rn");
        let cc_spec = int_spec(WindowFunc::CumCount, "v", "cc");
        let rn_result = apply_window(&rows, &rn_spec).unwrap();
        let cc_result = apply_window(&rows, &cc_spec).unwrap();
        let mut rn_vals: Vec<i64> = rn_result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        let mut cc_vals: Vec<i64> = cc_result
            .iter()
            .map(|(_, v)| match v {
                WindowValue::Integer(i) => *i,
                _ => 0,
            })
            .collect();
        rn_vals.sort();
        cc_vals.sort();
        assert_eq!(rn_vals, cc_vals);
    }

    #[test]
    fn test_lead_offset_2() {
        let rows: Vec<WindowRow> = (1i64..=5)
            .map(|i| make_row(&[("v", WindowValue::Integer(i))]))
            .collect();
        let spec = int_spec(
            WindowFunc::Lead {
                column: "v".to_string(),
                offset: 2,
                default: Some(WindowValue::Integer(0)),
            },
            "v",
            "lead",
        );
        let result = apply_window(&rows, &spec).unwrap();
        let mut sorted: Vec<(i64, WindowValue)> = result
            .iter()
            .map(|(row, val)| {
                let v = match row.values["v"] {
                    WindowValue::Integer(i) => i,
                    _ => 0,
                };
                (v, val.clone())
            })
            .collect();
        sorted.sort_by_key(|(v, _)| *v);
        assert_eq!(sorted[0].1, WindowValue::Integer(3));
        assert_eq!(sorted[1].1, WindowValue::Integer(4));
        assert_eq!(sorted[2].1, WindowValue::Integer(5));
        assert_eq!(sorted[3].1, WindowValue::Integer(0));
        assert_eq!(sorted[4].1, WindowValue::Integer(0));
    }
}
