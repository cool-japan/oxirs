/// Forward/backward time series iterator with windowing support.
///
/// Provides efficient iteration over ordered `DataPoint` slices in
/// forward or backward direction, with optional time-range filtering
/// and sliding-window iteration.
use chrono::{DateTime, Duration, Utc};

use crate::series::DataPoint;

/// Direction of iteration over a series.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IterDirection {
    /// Iterate from oldest to newest (ascending timestamp).
    Forward,
    /// Iterate from newest to oldest (descending timestamp).
    Backward,
}

/// An iterator over a contiguous slice of `DataPoint` values within a
/// time range, in either forward or backward direction.
pub struct SeriesSlice<'a> {
    data: &'a [DataPoint],
    pos: usize,
    direction: IterDirection,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
}

impl<'a> SeriesSlice<'a> {
    /// Create a new slice iterator.
    ///
    /// `data` must be sorted in ascending order of timestamp.
    /// `start` and `end` are inclusive bounds.
    pub fn new(
        data: &'a [DataPoint],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        direction: IterDirection,
    ) -> Self {
        // Find the sub-slice within [start, end].
        let first = data.partition_point(|dp| dp.timestamp < start);
        let last = data.partition_point(|dp| dp.timestamp <= end);
        let data = &data[first..last];

        let pos = match direction {
            IterDirection::Forward => 0,
            IterDirection::Backward => data.len(),
        };

        Self {
            data,
            pos,
            direction,
            start,
            end,
        }
    }

    /// Peek at the next element without consuming it.
    pub fn peek(&self) -> Option<&'a DataPoint> {
        match self.direction {
            IterDirection::Forward => {
                if self.pos < self.data.len() {
                    Some(&self.data[self.pos])
                } else {
                    None
                }
            }
            IterDirection::Backward => {
                if self.pos > 0 {
                    Some(&self.data[self.pos - 1])
                } else {
                    None
                }
            }
        }
    }

    /// Return the number of elements remaining in the iteration.
    pub fn remaining(&self) -> usize {
        match self.direction {
            IterDirection::Forward => self.data.len().saturating_sub(self.pos),
            IterDirection::Backward => self.pos,
        }
    }

    /// Return the start of the time range.
    pub fn range_start(&self) -> DateTime<Utc> {
        self.start
    }

    /// Return the end of the time range.
    pub fn range_end(&self) -> DateTime<Utc> {
        self.end
    }

    /// Return the iteration direction.
    pub fn direction(&self) -> &IterDirection {
        &self.direction
    }

    /// Return true if the iterator is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }
}

impl<'a> Iterator for SeriesSlice<'a> {
    type Item = &'a DataPoint;

    fn next(&mut self) -> Option<Self::Item> {
        match self.direction {
            IterDirection::Forward => {
                if self.pos < self.data.len() {
                    let item = &self.data[self.pos];
                    self.pos += 1;
                    Some(item)
                } else {
                    None
                }
            }
            IterDirection::Backward => {
                if self.pos > 0 {
                    self.pos -= 1;
                    Some(&self.data[self.pos])
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for SeriesSlice<'a> {}

/// A sliding-window iterator over a sorted `DataPoint` slice.
///
/// Returns overlapping or non-overlapping windows of data points
/// grouped by a fixed time-window duration and step.
pub struct WindowIterator<'a> {
    slice: &'a [DataPoint],
    window_size: Duration,
    step: Duration,
    pos: usize,
}

impl<'a> WindowIterator<'a> {
    /// Create a new window iterator.
    ///
    /// `window_size` is the duration of each window.
    /// `step` is how far to advance the window on each iteration.
    ///
    /// # Panics
    /// Panics if `window_size` or `step` is non-positive.
    pub fn new(data: &'a [DataPoint], window_size: Duration, step: Duration) -> Self {
        assert!(
            window_size > Duration::zero(),
            "window_size must be positive"
        );
        assert!(step > Duration::zero(), "step must be positive");
        Self {
            slice: data,
            window_size,
            step,
            pos: 0,
        }
    }

    /// Return the window size.
    pub fn window_size(&self) -> Duration {
        self.window_size
    }

    /// Return the step size.
    pub fn step(&self) -> Duration {
        self.step
    }
}

impl<'a> Iterator for WindowIterator<'a> {
    type Item = &'a [DataPoint];

    fn next(&mut self) -> Option<Self::Item> {
        if self.slice.is_empty() || self.pos >= self.slice.len() {
            return None;
        }

        let window_start = self.slice[self.pos].timestamp;
        let window_end = window_start + self.window_size;

        // Find elements within [window_start, window_end).
        let end_idx =
            self.slice[self.pos..].partition_point(|dp| dp.timestamp < window_end) + self.pos;

        let window = &self.slice[self.pos..end_idx];

        // Advance by step: find first point with timestamp >= window_start + step.
        let next_start = window_start + self.step;
        let advance = self.slice[self.pos..].partition_point(|dp| dp.timestamp < next_start);
        if advance == 0 {
            // Avoid infinite loop on zero advance.
            self.pos += 1;
        } else {
            self.pos += advance;
        }

        Some(window)
    }
}

/// Factory and utility functions for time series iteration.
pub struct SeriesIterator;

impl SeriesIterator {
    /// Create a forward iterator over the given time range.
    pub fn forward<'a>(
        data: &'a [DataPoint],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> SeriesSlice<'a> {
        SeriesSlice::new(data, start, end, IterDirection::Forward)
    }

    /// Create a backward iterator over the given time range.
    pub fn backward<'a>(
        data: &'a [DataPoint],
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> SeriesSlice<'a> {
        SeriesSlice::new(data, start, end, IterDirection::Backward)
    }

    /// Create a sliding window iterator.
    pub fn windows<'a>(
        data: &'a [DataPoint],
        window_size: Duration,
        step: Duration,
    ) -> WindowIterator<'a> {
        WindowIterator::new(data, window_size, step)
    }

    /// Collect all data points in `[start, end]` into a Vec.
    pub fn range(data: &[DataPoint], start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<&DataPoint> {
        Self::forward(data, start, end).collect()
    }

    /// Return the last `n` data points (most recent by timestamp).
    ///
    /// If `n` exceeds the slice length, returns the whole slice.
    pub fn latest_n(data: &[DataPoint], n: usize) -> &[DataPoint] {
        if n >= data.len() {
            data
        } else {
            &data[data.len() - n..]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    /// Build a sorted `DataPoint` slice from second-offsets.
    fn make_data(seconds: &[i64]) -> Vec<DataPoint> {
        seconds
            .iter()
            .enumerate()
            .map(|(i, &s)| DataPoint {
                timestamp: Utc.timestamp_opt(s, 0).unwrap(),
                value: i as f64,
            })
            .collect()
    }

    fn ts(s: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(s, 0).unwrap()
    }

    // --- IterDirection ---

    #[test]
    fn test_iter_direction_eq() {
        assert_eq!(IterDirection::Forward, IterDirection::Forward);
        assert_ne!(IterDirection::Forward, IterDirection::Backward);
    }

    #[test]
    fn test_iter_direction_clone() {
        let d = IterDirection::Backward;
        assert_eq!(d.clone(), IterDirection::Backward);
    }

    #[test]
    fn test_iter_direction_debug() {
        let s = format!("{:?}", IterDirection::Forward);
        assert_eq!(s, "Forward");
    }

    // --- SeriesSlice forward ---

    #[test]
    fn test_forward_full_range() {
        let data = make_data(&[10, 20, 30, 40, 50]);
        let items: Vec<_> = SeriesIterator::forward(&data, ts(10), ts(50)).collect();
        assert_eq!(items.len(), 5);
        assert_eq!(items[0].timestamp, ts(10));
        assert_eq!(items[4].timestamp, ts(50));
    }

    #[test]
    fn test_forward_partial_range() {
        let data = make_data(&[10, 20, 30, 40, 50]);
        let items: Vec<_> = SeriesIterator::forward(&data, ts(20), ts(40)).collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].timestamp, ts(20));
        assert_eq!(items[2].timestamp, ts(40));
    }

    #[test]
    fn test_forward_empty_range() {
        let data = make_data(&[10, 20, 30]);
        let items: Vec<_> = SeriesIterator::forward(&data, ts(100), ts(200)).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_forward_single_element() {
        let data = make_data(&[42]);
        let items: Vec<_> = SeriesIterator::forward(&data, ts(42), ts(42)).collect();
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn test_forward_empty_data() {
        let items: Vec<_> = SeriesIterator::forward(&[], ts(0), ts(100)).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_forward_timestamps_ascending() {
        let data = make_data(&[1, 2, 3, 4, 5]);
        let timestamps: Vec<DateTime<Utc>> = SeriesIterator::forward(&data, ts(1), ts(5))
            .map(|d| d.timestamp)
            .collect();
        let mut sorted = timestamps.clone();
        sorted.sort_unstable();
        assert_eq!(timestamps, sorted);
    }

    // --- SeriesSlice backward ---

    #[test]
    fn test_backward_full_range() {
        let data = make_data(&[10, 20, 30, 40, 50]);
        let items: Vec<_> = SeriesIterator::backward(&data, ts(10), ts(50)).collect();
        assert_eq!(items.len(), 5);
        assert_eq!(items[0].timestamp, ts(50)); // Most recent first.
        assert_eq!(items[4].timestamp, ts(10));
    }

    #[test]
    fn test_backward_partial_range() {
        let data = make_data(&[10, 20, 30, 40, 50]);
        let items: Vec<_> = SeriesIterator::backward(&data, ts(20), ts(40)).collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].timestamp, ts(40));
        assert_eq!(items[2].timestamp, ts(20));
    }

    #[test]
    fn test_backward_empty_data() {
        let items: Vec<_> = SeriesIterator::backward(&[], ts(0), ts(100)).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_backward_timestamps_descending() {
        let data = make_data(&[1, 2, 3, 4, 5]);
        let timestamps: Vec<DateTime<Utc>> = SeriesIterator::backward(&data, ts(1), ts(5))
            .map(|d| d.timestamp)
            .collect();
        let mut sorted = timestamps.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(timestamps, sorted);
    }

    // --- peek / remaining ---

    #[test]
    fn test_peek_forward_first() {
        let data = make_data(&[10, 20, 30]);
        let slice = SeriesIterator::forward(&data, ts(10), ts(30));
        assert_eq!(slice.peek().map(|d| d.timestamp), Some(ts(10)));
    }

    #[test]
    fn test_peek_backward_first() {
        let data = make_data(&[10, 20, 30]);
        let slice = SeriesIterator::backward(&data, ts(10), ts(30));
        assert_eq!(slice.peek().map(|d| d.timestamp), Some(ts(30)));
    }

    #[test]
    fn test_peek_empty() {
        let slice = SeriesIterator::forward(&[], ts(0), ts(100));
        assert!(slice.peek().is_none());
    }

    #[test]
    fn test_remaining_decreases() {
        let data = make_data(&[1, 2, 3, 4, 5]);
        let mut slice = SeriesIterator::forward(&data, ts(1), ts(5));
        assert_eq!(slice.remaining(), 5);
        let _ = slice.next();
        assert_eq!(slice.remaining(), 4);
    }

    #[test]
    fn test_remaining_zero_when_exhausted() {
        let data = make_data(&[1]);
        let mut slice = SeriesIterator::forward(&data, ts(1), ts(1));
        slice.next();
        assert_eq!(slice.remaining(), 0);
        assert!(slice.is_exhausted());
    }

    #[test]
    fn test_exact_size_iterator() {
        let data = make_data(&[1, 2, 3]);
        let slice = SeriesIterator::forward(&data, ts(1), ts(3));
        assert_eq!(slice.len(), 3);
    }

    // --- range_start / range_end / direction ---

    #[test]
    fn test_range_accessors() {
        let data = make_data(&[1, 2, 3]);
        let slice = SeriesIterator::forward(&data, ts(5), ts(15));
        assert_eq!(slice.range_start(), ts(5));
        assert_eq!(slice.range_end(), ts(15));
        assert_eq!(slice.direction(), &IterDirection::Forward);
    }

    // --- range ---

    #[test]
    fn test_range_collects() {
        let data = make_data(&[10, 20, 30, 40]);
        let pts = SeriesIterator::range(&data, ts(20), ts(30));
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn test_range_returns_references() {
        let data = make_data(&[1, 2, 3]);
        let pts = SeriesIterator::range(&data, ts(1), ts(3));
        assert_eq!(pts[0].timestamp, ts(1));
    }

    // --- latest_n ---

    #[test]
    fn test_latest_n_returns_last_n() {
        let data = make_data(&[1, 2, 3, 4, 5]);
        let last = SeriesIterator::latest_n(&data, 3);
        assert_eq!(last.len(), 3);
        assert_eq!(last[0].timestamp, ts(3));
        assert_eq!(last[2].timestamp, ts(5));
    }

    #[test]
    fn test_latest_n_exceeds_length() {
        let data = make_data(&[1, 2]);
        let last = SeriesIterator::latest_n(&data, 10);
        assert_eq!(last.len(), 2);
    }

    #[test]
    fn test_latest_n_zero() {
        let data = make_data(&[1, 2, 3]);
        let last = SeriesIterator::latest_n(&data, 0);
        assert_eq!(last.len(), 0);
    }

    #[test]
    fn test_latest_n_empty_data() {
        let last = SeriesIterator::latest_n(&[], 5);
        assert!(last.is_empty());
    }

    // --- WindowIterator ---

    #[test]
    fn test_window_iterator_basic() {
        let data = make_data(&[0, 5, 10, 15, 20]);
        let windows: Vec<_> =
            SeriesIterator::windows(&data, Duration::seconds(10), Duration::seconds(10)).collect();
        assert!(!windows.is_empty());
        // First window: [0, 10) → points at 0 and 5.
        assert_eq!(windows[0].len(), 2);
    }

    #[test]
    fn test_window_iterator_step_equal_window() {
        let data = make_data(&[0, 10, 20, 30]);
        let windows: Vec<_> =
            SeriesIterator::windows(&data, Duration::seconds(10), Duration::seconds(10)).collect();
        // Each window covers exactly one point with spacing 10s.
        for w in &windows {
            assert_eq!(w.len(), 1);
        }
    }

    #[test]
    fn test_window_iterator_empty_data() {
        let windows: Vec<_> =
            SeriesIterator::windows(&[], Duration::seconds(10), Duration::seconds(5)).collect();
        assert!(windows.is_empty());
    }

    #[test]
    fn test_window_size_accessor() {
        let data = make_data(&[1]);
        let wi = WindowIterator::new(&data, Duration::seconds(100), Duration::seconds(50));
        assert_eq!(wi.window_size(), Duration::seconds(100));
        assert_eq!(wi.step(), Duration::seconds(50));
    }

    #[test]
    #[should_panic(expected = "window_size must be positive")]
    fn test_window_size_zero_panics() {
        let data = make_data(&[1]);
        let _ = WindowIterator::new(&data, Duration::zero(), Duration::seconds(5));
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn test_window_step_zero_panics() {
        let data = make_data(&[1]);
        let _ = WindowIterator::new(&data, Duration::seconds(10), Duration::zero());
    }

    #[test]
    fn test_window_overlapping() {
        let data = make_data(&[0, 5, 10, 15, 20]);
        // Window 10s, step 5s → overlapping windows.
        let windows: Vec<_> =
            SeriesIterator::windows(&data, Duration::seconds(10), Duration::seconds(5)).collect();
        assert!(windows.len() > 1);
    }

    #[test]
    fn test_window_single_point() {
        let data = make_data(&[42]);
        let windows: Vec<_> =
            SeriesIterator::windows(&data, Duration::seconds(10), Duration::seconds(10)).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].len(), 1);
    }

    // --- Integration ---

    #[test]
    fn test_forward_then_backward_same_elements() {
        let data = make_data(&[1, 2, 3, 4, 5]);
        let forward: Vec<DateTime<Utc>> = SeriesIterator::forward(&data, ts(1), ts(5))
            .map(|d| d.timestamp)
            .collect();
        let mut backward: Vec<DateTime<Utc>> = SeriesIterator::backward(&data, ts(1), ts(5))
            .map(|d| d.timestamp)
            .collect();
        backward.reverse();
        assert_eq!(forward, backward);
    }

    #[test]
    fn test_range_and_forward_same_results() {
        let data = make_data(&[10, 20, 30, 40]);
        let range_pts = SeriesIterator::range(&data, ts(20), ts(30));
        let forward_pts: Vec<_> = SeriesIterator::forward(&data, ts(20), ts(30)).collect();
        assert_eq!(range_pts.len(), forward_pts.len());
        for (a, b) in range_pts.iter().zip(forward_pts.iter()) {
            assert_eq!(a.timestamp, b.timestamp);
        }
    }

    #[test]
    fn test_large_dataset_iteration() {
        let data: Vec<DataPoint> = (0..1000)
            .map(|i| DataPoint {
                timestamp: Utc.timestamp_opt(i, 0).unwrap(),
                value: i as f64,
            })
            .collect();
        let count = SeriesIterator::forward(&data, ts(0), ts(999)).count();
        assert_eq!(count, 1000);
    }
}
