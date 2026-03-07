//! Stream windowing functions: tumbling, sliding, and session windows.
//!
//! Provides stateless window computation over ordered time-series data points.

use std::collections::HashMap;

/// A single data point in a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct DataPoint {
    /// Epoch milliseconds.
    pub timestamp_ms: i64,
    /// Numeric value carried by this point.
    pub value: f64,
    /// Optional grouping key for `by_key` operations.
    pub key: Option<String>,
}

/// Specifies how windows are formed.
#[derive(Debug, Clone, PartialEq)]
pub enum WindowType {
    /// Non-overlapping windows of fixed `size_ms`.
    Tumbling(i64),
    /// Overlapping windows: each window spans `size_ms`, advanced by `step_ms`.
    Sliding { size_ms: i64, step_ms: i64 },
    /// Windows that close after `gap_ms` of inactivity.
    Session { gap_ms: i64 },
}

/// A window containing zero or more data points.
#[derive(Debug, Clone, PartialEq)]
pub struct Window {
    /// Inclusive start timestamp (ms).
    pub start_ms: i64,
    /// Exclusive end timestamp (ms).
    pub end_ms: i64,
    /// Data points that fall inside this window.
    pub points: Vec<DataPoint>,
}

impl Window {
    /// Returns `true` if this window contains no data points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Window span in milliseconds (`end_ms - start_ms`).
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Number of data points in this window.
    pub fn point_count(&self) -> usize {
        self.points.len()
    }
}

/// Result of applying a window function to a data stream.
#[derive(Debug, Clone)]
pub struct WindowResult {
    /// All produced windows (may include empty windows for tumbling/sliding).
    pub windows: Vec<Window>,
    /// Total number of data points processed across all windows.
    pub total_points: usize,
}

/// Aggregated statistics for a single window.
#[derive(Debug, Clone)]
pub struct WindowAggregation {
    /// The window this aggregation describes.
    pub window: Window,
    /// Number of data points.
    pub count: usize,
    /// Sum of all values.
    pub sum: f64,
    /// Minimum value (f64::MAX if empty).
    pub min: f64,
    /// Maximum value (f64::MIN if empty).
    pub max: f64,
    /// Mean of all values (0.0 if empty).
    pub mean: f64,
}

/// Stateless windowing utility.
pub struct WindowFunction;

impl WindowFunction {
    /// Apply the specified `WindowType` to `data` and return the result.
    pub fn apply(data: &[DataPoint], window_type: &WindowType) -> WindowResult {
        let windows = match window_type {
            WindowType::Tumbling(size_ms) => Self::tumbling(data, *size_ms),
            WindowType::Sliding { size_ms, step_ms } => Self::sliding(data, *size_ms, *step_ms),
            WindowType::Session { gap_ms } => Self::session(data, *gap_ms),
        };
        let total_points = windows.iter().map(|w| w.points.len()).sum();
        WindowResult {
            windows,
            total_points,
        }
    }

    /// Compute non-overlapping (tumbling) windows of `size_ms` milliseconds.
    ///
    /// Windows are aligned to multiples of `size_ms` from epoch 0.
    /// Empty windows between the first and last occupied window are included.
    pub fn tumbling(data: &[DataPoint], size_ms: i64) -> Vec<Window> {
        if data.is_empty() || size_ms <= 0 {
            return Vec::new();
        }

        // Determine the range of windows needed.
        let min_ts = data.iter().map(|p| p.timestamp_ms).min().unwrap_or(0);
        let max_ts = data.iter().map(|p| p.timestamp_ms).max().unwrap_or(0);

        let first_window_start = floor_div(min_ts, size_ms) * size_ms;
        let last_window_start = floor_div(max_ts, size_ms) * size_ms;

        let mut windows = Vec::new();
        let mut current = first_window_start;
        while current <= last_window_start {
            let start = current;
            let end = current + size_ms;
            let points: Vec<DataPoint> = data
                .iter()
                .filter(|p| p.timestamp_ms >= start && p.timestamp_ms < end)
                .cloned()
                .collect();
            windows.push(Window {
                start_ms: start,
                end_ms: end,
                points,
            });
            current += size_ms;
        }
        windows
    }

    /// Compute overlapping (sliding) windows of `size_ms`, advanced by `step_ms`.
    ///
    /// A window `[t, t+size_ms)` is created for every `t = floor(first/step)*step, ...`
    /// up to the last timestamp.  If `step_ms >= size_ms` the windows do not overlap.
    pub fn sliding(data: &[DataPoint], size_ms: i64, step_ms: i64) -> Vec<Window> {
        if data.is_empty() || size_ms <= 0 || step_ms <= 0 {
            return Vec::new();
        }

        let min_ts = data.iter().map(|p| p.timestamp_ms).min().unwrap_or(0);
        let max_ts = data.iter().map(|p| p.timestamp_ms).max().unwrap_or(0);

        let first_window_start = floor_div(min_ts, step_ms) * step_ms;

        let mut windows = Vec::new();
        let mut current = first_window_start;
        loop {
            let start = current;
            let end = current + size_ms;
            // Keep windows that could contain at least the latest point.
            if start > max_ts {
                break;
            }
            let points: Vec<DataPoint> = data
                .iter()
                .filter(|p| p.timestamp_ms >= start && p.timestamp_ms < end)
                .cloned()
                .collect();
            windows.push(Window {
                start_ms: start,
                end_ms: end,
                points,
            });
            current += step_ms;
        }
        windows
    }

    /// Compute session windows: a new window opens on each point and closes
    /// when the gap to the next point exceeds `gap_ms`.
    pub fn session(data: &[DataPoint], gap_ms: i64) -> Vec<Window> {
        if data.is_empty() || gap_ms <= 0 {
            return Vec::new();
        }

        // Sort by timestamp (clone to avoid mutating caller's slice).
        let mut sorted: Vec<DataPoint> = data.to_vec();
        sorted.sort_by_key(|p| p.timestamp_ms);

        let mut windows = Vec::new();
        let mut session_points: Vec<DataPoint> = Vec::new();

        for point in sorted {
            if let Some(last) = session_points.last() {
                if point.timestamp_ms - last.timestamp_ms > gap_ms {
                    // Close current session.
                    let start = session_points.first().map(|p| p.timestamp_ms).unwrap_or(0);
                    let end = session_points
                        .last()
                        .map(|p| p.timestamp_ms + 1)
                        .unwrap_or(1);
                    windows.push(Window {
                        start_ms: start,
                        end_ms: end,
                        points: std::mem::take(&mut session_points),
                    });
                }
            }
            session_points.push(point);
        }

        // Flush the remaining session.
        if !session_points.is_empty() {
            let start = session_points.first().map(|p| p.timestamp_ms).unwrap_or(0);
            let end = session_points
                .last()
                .map(|p| p.timestamp_ms + 1)
                .unwrap_or(1);
            windows.push(Window {
                start_ms: start,
                end_ms: end,
                points: session_points,
            });
        }

        windows
    }

    /// Compute aggregated statistics for a single `window`.
    pub fn aggregate(window: &Window) -> WindowAggregation {
        let count = window.points.len();
        if count == 0 {
            return WindowAggregation {
                window: window.clone(),
                count: 0,
                sum: 0.0,
                min: f64::MAX,
                max: f64::MIN,
                mean: 0.0,
            };
        }
        let sum: f64 = window.points.iter().map(|p| p.value).sum();
        let min = window
            .points
            .iter()
            .map(|p| p.value)
            .fold(f64::MAX, f64::min);
        let max = window
            .points
            .iter()
            .map(|p| p.value)
            .fold(f64::MIN, f64::max);
        let mean = sum / count as f64;
        WindowAggregation {
            window: window.clone(),
            count,
            sum,
            min,
            max,
            mean,
        }
    }

    /// Aggregate every window in a `WindowResult`.
    pub fn aggregate_all(result: &WindowResult) -> Vec<WindowAggregation> {
        result.windows.iter().map(Self::aggregate).collect()
    }

    /// Group `data` by `DataPoint::key`, apply the window type to each group,
    /// and return a map of key → `WindowResult`.
    ///
    /// Points with `key = None` are grouped under the empty string `""`.
    pub fn by_key(data: &[DataPoint], window_type: &WindowType) -> HashMap<String, WindowResult> {
        let mut groups: HashMap<String, Vec<DataPoint>> = HashMap::new();
        for point in data {
            let k = point.key.clone().unwrap_or_default();
            groups.entry(k).or_default().push(point.clone());
        }
        groups
            .into_iter()
            .map(|(k, pts)| (k, Self::apply(&pts, window_type)))
            .collect()
    }
}

/// Integer floor division (handles negative timestamps correctly).
fn floor_div(a: i64, b: i64) -> i64 {
    let d = a / b;
    // Adjust if the division rounded toward zero instead of floor.
    if (a ^ b) < 0 && d * b != a {
        d - 1
    } else {
        d
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(timestamps: &[(i64, f64)]) -> Vec<DataPoint> {
        timestamps
            .iter()
            .map(|&(ts, v)| DataPoint {
                timestamp_ms: ts,
                value: v,
                key: None,
            })
            .collect()
    }

    fn keyed(ts: i64, value: f64, key: &str) -> DataPoint {
        DataPoint {
            timestamp_ms: ts,
            value,
            key: Some(key.to_string()),
        }
    }

    // --- Tumbling window tests ---

    #[test]
    fn test_tumbling_basic() {
        let data = pts(&[(0, 1.0), (500, 2.0), (1000, 3.0), (1500, 4.0)]);
        let windows = WindowFunction::tumbling(&data, 1000);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].points.len(), 2); // 0, 500
        assert_eq!(windows[1].points.len(), 2); // 1000, 1500
    }

    #[test]
    fn test_tumbling_alignment() {
        // All points in the same window
        let data = pts(&[(100, 1.0), (200, 2.0), (300, 3.0)]);
        let windows = WindowFunction::tumbling(&data, 1000);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].start_ms, 0);
        assert_eq!(windows[0].end_ms, 1000);
    }

    #[test]
    fn test_tumbling_empty_data() {
        let windows = WindowFunction::tumbling(&[], 1000);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_tumbling_single_point() {
        let data = pts(&[(5000, 42.0)]);
        let windows = WindowFunction::tumbling(&data, 1000);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].points.len(), 1);
        assert_eq!(windows[0].start_ms, 5000);
    }

    #[test]
    fn test_tumbling_with_empty_intermediate_window() {
        // Gap between ts=0 and ts=3000 → windows [0,1000), [1000,2000), [2000,3000), [3000,4000)
        let data = pts(&[(0, 1.0), (3500, 2.0)]);
        let windows = WindowFunction::tumbling(&data, 1000);
        // 4 windows: 0..1000, 1000..2000, 2000..3000, 3000..4000
        assert_eq!(windows.len(), 4);
        // Middle windows should be empty
        assert!(windows[1].is_empty());
        assert!(windows[2].is_empty());
    }

    #[test]
    fn test_tumbling_exact_boundary() {
        // ts=1000 is exactly on the boundary → belongs to [1000,2000)
        let data = pts(&[(999, 1.0), (1000, 2.0)]);
        let windows = WindowFunction::tumbling(&data, 1000);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].points.len(), 1); // ts=999
        assert_eq!(windows[1].points.len(), 1); // ts=1000
    }

    // --- Sliding window tests ---

    #[test]
    fn test_sliding_basic() {
        let data = pts(&[(0, 1.0), (200, 2.0), (400, 3.0), (600, 4.0)]);
        let windows = WindowFunction::sliding(&data, 500, 200);
        // Each window is 500ms wide, step 200ms
        assert!(!windows.is_empty());
        // Check overlaps: point at ts=200 should appear in multiple windows
        let windows_containing_200: Vec<_> = windows
            .iter()
            .filter(|w| w.points.iter().any(|p| p.timestamp_ms == 200))
            .collect();
        assert!(
            windows_containing_200.len() > 1,
            "sliding windows must overlap"
        );
    }

    #[test]
    fn test_sliding_no_overlap_when_step_ge_size() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0)]);
        // step >= size → effectively tumbling
        let windows = WindowFunction::sliding(&data, 1000, 1000);
        // No point should appear in more than one window
        for (i, w) in windows.iter().enumerate() {
            for (j, other) in windows.iter().enumerate() {
                if i == j {
                    continue;
                }
                for p in &w.points {
                    assert!(
                        !other
                            .points
                            .iter()
                            .any(|q| q.timestamp_ms == p.timestamp_ms),
                        "point should not appear in multiple windows when step >= size"
                    );
                }
            }
        }
    }

    #[test]
    fn test_sliding_empty_data() {
        let windows = WindowFunction::sliding(&[], 500, 100);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_sliding_single_point() {
        let data = pts(&[(500, 7.0)]);
        let windows = WindowFunction::sliding(&data, 1000, 500);
        // At least one window must contain the point
        assert!(windows.iter().any(|w| !w.is_empty()));
    }

    #[test]
    fn test_sliding_step_greater_than_size() {
        let data = pts(&[(0, 1.0), (100, 2.0), (200, 3.0)]);
        // step > size → gaps between windows
        let windows = WindowFunction::sliding(&data, 50, 200);
        // Windows of 50ms wide should capture points individually
        for w in &windows {
            assert!(w.points.len() <= 1);
        }
    }

    // --- Session window tests ---

    #[test]
    fn test_session_basic_gap_splitting() {
        // Two clusters separated by > gap_ms
        let data = pts(&[(0, 1.0), (100, 2.0), (200, 3.0), (5000, 4.0), (5100, 5.0)]);
        let windows = WindowFunction::session(&data, 500);
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].points.len(), 3);
        assert_eq!(windows[1].points.len(), 2);
    }

    #[test]
    fn test_session_no_split_within_gap() {
        let data = pts(&[(0, 1.0), (100, 2.0), (200, 3.0)]);
        let windows = WindowFunction::session(&data, 200);
        // 200ms gap exactly → points 0-100 and 100-200 are within gap
        assert_eq!(windows.len(), 1);
    }

    #[test]
    fn test_session_single_point() {
        let data = pts(&[(1000, 9.9)]);
        let windows = WindowFunction::session(&data, 500);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].points.len(), 1);
    }

    #[test]
    fn test_session_empty_data() {
        let windows = WindowFunction::session(&[], 500);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_session_multiple_gaps() {
        let data = pts(&[(0, 1.0), (2000, 2.0), (4000, 3.0), (4100, 4.0)]);
        let windows = WindowFunction::session(&data, 500);
        assert_eq!(windows.len(), 3);
    }

    // --- Aggregate tests ---

    #[test]
    fn test_aggregate_count() {
        let data = pts(&[(0, 1.0), (100, 2.0), (200, 3.0)]);
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        let agg = WindowFunction::aggregate(&win);
        assert_eq!(agg.count, 3);
    }

    #[test]
    fn test_aggregate_sum() {
        let data = pts(&[(0, 10.0), (100, 20.0), (200, 30.0)]);
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        let agg = WindowFunction::aggregate(&win);
        assert!((agg.sum - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_min() {
        let data = pts(&[(0, 5.0), (100, 1.0), (200, 3.0)]);
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        let agg = WindowFunction::aggregate(&win);
        assert!((agg.min - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_max() {
        let data = pts(&[(0, 5.0), (100, 1.0), (200, 3.0)]);
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        let agg = WindowFunction::aggregate(&win);
        assert!((agg.max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_mean() {
        let data = pts(&[(0, 2.0), (100, 4.0), (200, 6.0)]);
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        let agg = WindowFunction::aggregate(&win);
        assert!((agg.mean - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_empty_window() {
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: Vec::new(),
        };
        let agg = WindowFunction::aggregate(&win);
        assert_eq!(agg.count, 0);
        assert!((agg.sum - 0.0).abs() < 1e-9);
        assert!((agg.mean - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_all() {
        let data = pts(&[(0, 1.0), (500, 2.0), (1000, 3.0)]);
        let result = WindowFunction::apply(&data, &WindowType::Tumbling(1000));
        let aggs = WindowFunction::aggregate_all(&result);
        assert_eq!(aggs.len(), result.windows.len());
    }

    // --- by_key tests ---

    #[test]
    fn test_by_key_groups_correctly() {
        let data = vec![
            keyed(0, 1.0, "a"),
            keyed(100, 2.0, "b"),
            keyed(200, 3.0, "a"),
            keyed(300, 4.0, "b"),
        ];
        let groups = WindowFunction::by_key(&data, &WindowType::Tumbling(1000));
        assert_eq!(groups.len(), 2);
        let a_result = groups.get("a").expect("key a must exist");
        assert_eq!(a_result.total_points, 2);
        let b_result = groups.get("b").expect("key b must exist");
        assert_eq!(b_result.total_points, 2);
    }

    #[test]
    fn test_by_key_none_mapped_to_empty_string() {
        let data = pts(&[(0, 1.0), (100, 2.0)]);
        let groups = WindowFunction::by_key(&data, &WindowType::Tumbling(1000));
        assert!(groups.contains_key(""));
    }

    #[test]
    fn test_by_key_multiple_keys() {
        let data = vec![keyed(0, 1.0, "x"), keyed(0, 2.0, "y"), keyed(0, 3.0, "z")];
        let groups = WindowFunction::by_key(&data, &WindowType::Tumbling(1000));
        assert_eq!(groups.len(), 3);
    }

    // --- WindowResult.total_points ---

    #[test]
    fn test_window_result_total_points() {
        let data = pts(&[(0, 1.0), (500, 2.0), (1000, 3.0), (1500, 4.0)]);
        let result = WindowFunction::apply(&data, &WindowType::Tumbling(1000));
        // Each point appears in exactly one tumbling window.
        assert_eq!(result.total_points, 4);
    }

    #[test]
    fn test_window_result_total_points_sliding_overlap() {
        let data = pts(&[(0, 1.0), (200, 2.0)]);
        // A point may appear in more than one sliding window.
        let result = WindowFunction::apply(
            &data,
            &WindowType::Sliding {
                size_ms: 400,
                step_ms: 100,
            },
        );
        // With overlap, total_points ≥ actual unique points.
        assert!(result.total_points >= 2);
    }

    // --- Window helper methods ---

    #[test]
    fn test_window_is_empty_true() {
        let w = Window {
            start_ms: 0,
            end_ms: 1000,
            points: Vec::new(),
        };
        assert!(w.is_empty());
    }

    #[test]
    fn test_window_is_empty_false() {
        let w = Window {
            start_ms: 0,
            end_ms: 1000,
            points: vec![DataPoint {
                timestamp_ms: 0,
                value: 1.0,
                key: None,
            }],
        };
        assert!(!w.is_empty());
    }

    #[test]
    fn test_window_duration_ms() {
        let w = Window {
            start_ms: 1000,
            end_ms: 3000,
            points: Vec::new(),
        };
        assert_eq!(w.duration_ms(), 2000);
    }

    #[test]
    fn test_window_point_count() {
        let data = pts(&[(0, 1.0), (100, 2.0)]);
        let w = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        assert_eq!(w.point_count(), 2);
    }

    // --- apply() dispatch ---

    #[test]
    fn test_apply_tumbling() {
        let data = pts(&[(0, 1.0)]);
        let result = WindowFunction::apply(&data, &WindowType::Tumbling(1000));
        assert_eq!(result.windows.len(), 1);
    }

    #[test]
    fn test_apply_sliding() {
        let data = pts(&[(0, 1.0)]);
        let result = WindowFunction::apply(
            &data,
            &WindowType::Sliding {
                size_ms: 1000,
                step_ms: 500,
            },
        );
        assert!(!result.windows.is_empty());
    }

    #[test]
    fn test_apply_session() {
        let data = pts(&[(0, 1.0)]);
        let result = WindowFunction::apply(&data, &WindowType::Session { gap_ms: 500 });
        assert_eq!(result.windows.len(), 1);
    }

    #[test]
    fn test_tumbling_zero_size_returns_empty() {
        let data = pts(&[(0, 1.0)]);
        let windows = WindowFunction::tumbling(&data, 0);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_sliding_zero_step_returns_empty() {
        let data = pts(&[(0, 1.0)]);
        let windows = WindowFunction::sliding(&data, 1000, 0);
        assert!(windows.is_empty());
    }

    #[test]
    fn test_aggregate_single_point() {
        let data = pts(&[(500, 7.7)]);
        let win = Window {
            start_ms: 0,
            end_ms: 1000,
            points: data,
        };
        let agg = WindowFunction::aggregate(&win);
        assert_eq!(agg.count, 1);
        assert!((agg.min - 7.7).abs() < 1e-6);
        assert!((agg.max - 7.7).abs() < 1e-6);
        assert!((agg.mean - 7.7).abs() < 1e-6);
    }

    #[test]
    fn test_tumbling_many_points() {
        let data: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint {
                timestamp_ms: i * 100,
                value: i as f64,
                key: None,
            })
            .collect();
        let windows = WindowFunction::tumbling(&data, 1000);
        // 100 points × 100ms = 10 seconds → 10 windows of 1000ms each
        assert_eq!(windows.len(), 10);
        for w in &windows {
            assert_eq!(w.point_count(), 10);
        }
    }

    #[test]
    fn test_session_unsorted_input() {
        // session() must sort by timestamp internally
        let data = vec![
            DataPoint {
                timestamp_ms: 5000,
                value: 3.0,
                key: None,
            },
            DataPoint {
                timestamp_ms: 0,
                value: 1.0,
                key: None,
            },
            DataPoint {
                timestamp_ms: 100,
                value: 2.0,
                key: None,
            },
        ];
        let windows = WindowFunction::session(&data, 500);
        assert_eq!(windows.len(), 2); // [0,100] and [5000]
    }

    #[test]
    fn test_by_key_empty_data() {
        let groups = WindowFunction::by_key(&[], &WindowType::Tumbling(1000));
        assert!(groups.is_empty());
    }

    #[test]
    fn test_window_result_windows_count() {
        let data = pts(&[(0, 1.0), (1000, 2.0), (2000, 3.0)]);
        let result = WindowFunction::apply(&data, &WindowType::Tumbling(1000));
        assert_eq!(result.windows.len(), 3);
    }
}
