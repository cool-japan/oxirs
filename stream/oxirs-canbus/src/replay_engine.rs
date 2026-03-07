//! CAN bus frame replay engine with time scaling and filtering.
//!
//! Replays a sequence of `CanFrame` values at a configurable speed factor,
//! optionally filtering by CAN identifier and supporting looping playback.

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single CAN bus frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFrame {
    /// Original capture timestamp in microseconds
    pub timestamp_us: u64,
    /// CAN identifier (11-bit standard or 29-bit extended)
    pub can_id: u32,
    /// Payload bytes (0–8 for CAN 2.0, up to 64 for CAN FD)
    pub data: Vec<u8>,
    /// `true` when using a 29-bit extended identifier
    pub is_extended: bool,
}

impl CanFrame {
    /// Construct a standard-frame (11-bit) CAN frame.
    pub fn standard(timestamp_us: u64, can_id: u32, data: Vec<u8>) -> Self {
        Self {
            timestamp_us,
            can_id,
            data,
            is_extended: false,
        }
    }

    /// Construct an extended-frame (29-bit) CAN frame.
    pub fn extended(timestamp_us: u64, can_id: u32, data: Vec<u8>) -> Self {
        Self {
            timestamp_us,
            can_id,
            data,
            is_extended: true,
        }
    }
}

/// Configuration controlling replay behaviour.
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Speed multiplier: 2.0 = twice as fast, 0.5 = half speed
    pub speed_factor: f64,
    /// When `true`, the replay restarts from the beginning after the last frame
    pub loop_replay: bool,
    /// When `Some`, only frames whose `can_id` is in this list are emitted
    pub filter_ids: Option<Vec<u32>>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            speed_factor: 1.0,
            loop_replay: false,
            filter_ids: None,
        }
    }
}

/// Accumulated replay statistics.
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    /// Number of frames actually emitted
    pub frames_replayed: usize,
    /// Number of frames skipped because of `filter_ids`
    pub frames_skipped: usize,
    /// Total duration of the original recording in microseconds
    pub total_duration_us: u64,
}

/// Event produced by [`ReplayEngine::next`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayEvent {
    /// A CAN frame ready to be processed
    Frame(CanFrame),
    /// All frames have been replayed and loop is disabled
    End,
    /// The replay looped back to the beginning
    Loop,
}

// ─────────────────────────────────────────────────────────────────────────────
// Engine
// ─────────────────────────────────────────────────────────────────────────────

/// CAN bus replay engine.
pub struct ReplayEngine {
    frames: Vec<CanFrame>,
    config: ReplayConfig,
    position: usize,
    /// Timestamp of the first frame in the current pass (for relative-time display)
    base_time_us: u64,
    stats: ReplayStats,
}

impl ReplayEngine {
    /// Create a new engine with the given frames and configuration.
    pub fn new(frames: Vec<CanFrame>, config: ReplayConfig) -> Self {
        let base_time_us = frames.first().map(|f| f.timestamp_us).unwrap_or(0);
        let total_duration_us = compute_total_duration(&frames);
        let mut engine = Self {
            frames,
            config,
            position: 0,
            base_time_us,
            stats: ReplayStats {
                total_duration_us,
                ..Default::default()
            },
        };
        engine.skip_filtered_at_current_position();
        engine
    }

    /// Replace the current frame list and reset replay state.
    pub fn load(&mut self, frames: Vec<CanFrame>) {
        self.stats.total_duration_us = compute_total_duration(&frames);
        self.frames = frames;
        self.position = 0;
        self.base_time_us = self.frames.first().map(|f| f.timestamp_us).unwrap_or(0);
        self.stats.frames_replayed = 0;
        self.stats.frames_skipped = 0;
        self.skip_filtered_at_current_position();
    }

    /// Advance to the next frame and return the corresponding `ReplayEvent`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> ReplayEvent {
        if self.frames.is_empty() {
            return ReplayEvent::End;
        }

        if self.position >= self.frames.len() {
            if self.config.loop_replay {
                self.position = 0;
                self.base_time_us = self.frames.first().map(|f| f.timestamp_us).unwrap_or(0);
                self.skip_filtered_at_current_position();
                return ReplayEvent::Loop;
            } else {
                return ReplayEvent::End;
            }
        }

        let frame = self.frames[self.position].clone();
        self.position += 1;

        // Advance past filtered frames, counting skips
        self.skip_filtered_at_current_position();

        self.stats.frames_replayed += 1;
        ReplayEvent::Frame(frame)
    }

    /// Return the scaled timestamp of the next frame (if any).
    ///
    /// The timestamp is scaled relative to the start of the current pass.
    pub fn peek_timestamp(&self) -> Option<u64> {
        if self.position >= self.frames.len() {
            return None;
        }
        Some(scale_timestamp(
            self.frames[self.position].timestamp_us,
            self.base_time_us,
            self.config.speed_factor,
        ))
    }

    /// Reset the replay to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
        self.base_time_us = self.frames.first().map(|f| f.timestamp_us).unwrap_or(0);
        self.stats.frames_replayed = 0;
        self.stats.frames_skipped = 0;
        self.skip_filtered_at_current_position();
    }

    /// Seek to the first frame at or after `timestamp_us` (using original timestamps).
    ///
    /// Returns the new position index.
    pub fn seek(&mut self, timestamp_us: u64) -> usize {
        // Find the first frame whose original timestamp >= timestamp_us,
        // taking the filter into account.
        let effective_ids = self.config.filter_ids.clone();
        let pos = self
            .frames
            .iter()
            .position(|f| {
                f.timestamp_us >= timestamp_us
                    && effective_ids
                        .as_ref()
                        .map(|ids| ids.contains(&f.can_id))
                        .unwrap_or(true)
            })
            .unwrap_or(self.frames.len());

        self.position = pos;
        self.skip_filtered_at_current_position();
        self.position
    }

    /// Return `true` when there are no more frames to replay (and looping is off).
    pub fn is_done(&self) -> bool {
        !self.config.loop_replay && self.position >= self.frames.len()
    }

    /// Total number of frames in the engine.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Current read position (index into the frame list).
    pub fn position(&self) -> usize {
        self.position
    }

    /// Accumulated replay statistics.
    pub fn stats(&self) -> &ReplayStats {
        &self.stats
    }

    /// Total duration of the recording divided by `speed_factor`.
    pub fn scaled_duration(&self) -> u64 {
        let sf = self.config.speed_factor;
        if sf <= 0.0 {
            return 0;
        }
        (self.stats.total_duration_us as f64 / sf) as u64
    }

    // ── private ──────────────────────────────────────────────────────────────

    /// Advance `position` past any frames that are excluded by the filter,
    /// counting them as skipped.
    fn skip_filtered_at_current_position(&mut self) {
        if let Some(ids) = self.config.filter_ids.clone() {
            while self.position < self.frames.len()
                && !ids.contains(&self.frames[self.position].can_id)
            {
                self.position += 1;
                self.stats.frames_skipped += 1;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Free functions
// ─────────────────────────────────────────────────────────────────────────────

/// Scale a frame timestamp relative to `base_us` by `speed_factor`.
///
/// Returns the number of microseconds since the start of the recording,
/// divided by the speed factor.
pub fn scale_timestamp(original_us: u64, base_us: u64, speed_factor: f64) -> u64 {
    if speed_factor <= 0.0 {
        return 0;
    }
    let relative = original_us.saturating_sub(base_us);
    (relative as f64 / speed_factor) as u64
}

fn compute_total_duration(frames: &[CanFrame]) -> u64 {
    if frames.len() < 2 {
        return 0;
    }
    let first = frames.first().map(|f| f.timestamp_us).unwrap_or(0);
    let last = frames.last().map(|f| f.timestamp_us).unwrap_or(0);
    last.saturating_sub(first)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frames() -> Vec<CanFrame> {
        vec![
            CanFrame::standard(0, 0x100, vec![0x01]),
            CanFrame::standard(1_000, 0x200, vec![0x02]),
            CanFrame::standard(2_000, 0x300, vec![0x03]),
            CanFrame::standard(3_000, 0x100, vec![0x04]),
            CanFrame::standard(4_000, 0x200, vec![0x05]),
        ]
    }

    fn default_config() -> ReplayConfig {
        ReplayConfig::default()
    }

    // 1. Replay all frames without filter
    #[test]
    fn test_replay_all_frames() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        let mut count = 0;
        while let ReplayEvent::Frame(_) = engine.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    // 2. is_done after last frame
    #[test]
    fn test_is_done_after_last_frame() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        for _ in 0..5 {
            engine.next();
        }
        assert!(engine.is_done());
    }

    // 3. is_done false while frames remain
    #[test]
    fn test_is_done_false_while_remaining() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        engine.next();
        assert!(!engine.is_done());
    }

    // 4. 2x speed halves timestamps
    #[test]
    fn test_time_scaling_2x() {
        let ts = scale_timestamp(2_000, 0, 2.0);
        assert_eq!(ts, 1_000);
    }

    // 5. 0.5x speed doubles timestamps
    #[test]
    fn test_time_scaling_half() {
        let ts = scale_timestamp(2_000, 0, 0.5);
        assert_eq!(ts, 4_000);
    }

    // 6. 1x speed keeps timestamps unchanged
    #[test]
    fn test_time_scaling_1x() {
        let ts = scale_timestamp(3_000, 0, 1.0);
        assert_eq!(ts, 3_000);
    }

    // 7. filter_ids skips non-matching frames
    #[test]
    fn test_filter_ids_skips_frames() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0x100]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        let mut replayed = vec![];
        while let ReplayEvent::Frame(f) = engine.next() {
            replayed.push(f.can_id);
        }
        assert_eq!(replayed, vec![0x100, 0x100]);
    }

    // 8. Stats track skipped and replayed frames
    #[test]
    fn test_stats_skipped_replayed() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0x100]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        while engine.next() != ReplayEvent::End {}
        assert_eq!(engine.stats().frames_replayed, 2);
        assert_eq!(engine.stats().frames_skipped, 3);
    }

    // 9. loop_replay resets and emits Loop event
    #[test]
    fn test_loop_replay() {
        let config = ReplayConfig {
            loop_replay: true,
            ..Default::default()
        };
        let frames = vec![
            CanFrame::standard(0, 0x1, vec![]),
            CanFrame::standard(1_000, 0x2, vec![]),
        ];
        let mut engine = ReplayEngine::new(frames, config);
        engine.next(); // frame 1
        engine.next(); // frame 2
        let event = engine.next(); // should loop
        assert_eq!(event, ReplayEvent::Loop);
    }

    // 10. loop_replay does not set is_done
    #[test]
    fn test_loop_replay_not_done() {
        let config = ReplayConfig {
            loop_replay: true,
            ..Default::default()
        };
        let frames = vec![CanFrame::standard(0, 0x1, vec![])];
        let mut engine = ReplayEngine::new(frames, config);
        engine.next(); // frame
        engine.next(); // Loop event
        assert!(!engine.is_done());
    }

    // 11. seek to timestamp returns correct position
    #[test]
    fn test_seek_to_timestamp() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        let pos = engine.seek(2_000);
        assert_eq!(pos, 2); // frames[2].timestamp_us == 2000
    }

    // 12. seek beyond all frames goes to end
    #[test]
    fn test_seek_beyond_end() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        let pos = engine.seek(1_000_000);
        assert_eq!(pos, engine.frame_count());
        assert!(engine.is_done());
    }

    // 13. peek_timestamp returns scaled ts of next frame
    #[test]
    fn test_peek_timestamp() {
        let config = ReplayConfig {
            speed_factor: 2.0,
            ..Default::default()
        };
        let frames = vec![
            CanFrame::standard(0, 0x1, vec![]),
            CanFrame::standard(2_000, 0x2, vec![]),
        ];
        let mut engine = ReplayEngine::new(frames, config);
        engine.next(); // consume first frame
        let ts = engine.peek_timestamp().expect("peek");
        assert_eq!(ts, 1_000); // 2000 / 2.0
    }

    // 14. peek_timestamp None when done
    #[test]
    fn test_peek_timestamp_none_when_done() {
        let frames = vec![CanFrame::standard(0, 0x1, vec![])];
        let mut engine = ReplayEngine::new(frames, default_config());
        engine.next();
        assert!(engine.peek_timestamp().is_none());
    }

    // 15. reset restarts replay
    #[test]
    fn test_reset() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        for _ in 0..5 {
            engine.next();
        }
        engine.reset();
        assert_eq!(engine.position(), 0);
        assert_eq!(engine.stats().frames_replayed, 0);
    }

    // 16. frame_count returns total frames
    #[test]
    fn test_frame_count() {
        let engine = ReplayEngine::new(sample_frames(), default_config());
        assert_eq!(engine.frame_count(), 5);
    }

    // 17. position advances with next
    #[test]
    fn test_position_advances() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        assert_eq!(engine.position(), 0);
        engine.next();
        assert_eq!(engine.position(), 1);
    }

    // 18. scaled_duration uses speed_factor
    #[test]
    fn test_scaled_duration() {
        let config = ReplayConfig {
            speed_factor: 2.0,
            ..Default::default()
        };
        let engine = ReplayEngine::new(sample_frames(), config);
        // total duration = 4000 us; scaled = 2000
        assert_eq!(engine.scaled_duration(), 2_000);
    }

    // 19. scaled_duration at 1x equals total duration
    #[test]
    fn test_scaled_duration_1x() {
        let engine = ReplayEngine::new(sample_frames(), default_config());
        assert_eq!(engine.scaled_duration(), 4_000);
    }

    // 20. empty frame list
    #[test]
    fn test_empty_frame_list() {
        let mut engine = ReplayEngine::new(vec![], default_config());
        assert!(engine.is_done());
        assert_eq!(engine.next(), ReplayEvent::End);
        assert_eq!(engine.frame_count(), 0);
        assert!(engine.peek_timestamp().is_none());
    }

    // 21. load replaces frames and resets
    #[test]
    fn test_load_replaces_frames() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        engine.next();
        engine.next();
        let new_frames = vec![CanFrame::standard(0, 0xAAA, vec![0xFF])];
        engine.load(new_frames);
        assert_eq!(engine.position(), 0);
        assert_eq!(engine.frame_count(), 1);
        if let ReplayEvent::Frame(f) = engine.next() {
            assert_eq!(f.can_id, 0xAAA);
        } else {
            panic!("Expected frame");
        }
    }

    // 22. stats total_duration_us computed correctly
    #[test]
    fn test_stats_total_duration() {
        let engine = ReplayEngine::new(sample_frames(), default_config());
        assert_eq!(engine.stats().total_duration_us, 4_000);
    }

    // 23. single-frame recording has 0 duration
    #[test]
    fn test_single_frame_zero_duration() {
        let frames = vec![CanFrame::standard(5_000, 0x1, vec![])];
        let engine = ReplayEngine::new(frames, default_config());
        assert_eq!(engine.stats().total_duration_us, 0);
    }

    // 24. filter_ids None means all frames emitted
    #[test]
    fn test_filter_ids_none_all_emitted() {
        let config = ReplayConfig {
            filter_ids: None,
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        let mut count = 0;
        while let ReplayEvent::Frame(_) = engine.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    // 25. filter_ids empty vec skips all frames
    #[test]
    fn test_filter_ids_empty_skips_all() {
        let config = ReplayConfig {
            filter_ids: Some(vec![]),
            ..Default::default()
        };
        let engine = ReplayEngine::new(sample_frames(), config);
        // All frames immediately filtered → is_done
        assert!(engine.is_done());
    }

    // 26. seek with filter respects filter
    #[test]
    fn test_seek_with_filter() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0x200]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        let pos = engine.seek(1_000);
        // frames[1] has id 0x200 and ts 1000
        assert_eq!(pos, 1);
    }

    // 27. End event returned when no frames match filter
    #[test]
    fn test_end_event_no_matching_frames() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0xFFFF]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        let event = engine.next();
        assert_eq!(event, ReplayEvent::End);
    }

    // 28. CanFrame standard helper
    #[test]
    fn test_can_frame_standard_helper() {
        let f = CanFrame::standard(1_000, 0x100, vec![0x01, 0x02]);
        assert!(!f.is_extended);
        assert_eq!(f.timestamp_us, 1_000);
        assert_eq!(f.can_id, 0x100);
    }

    // 29. CanFrame extended helper
    #[test]
    fn test_can_frame_extended_helper() {
        let f = CanFrame::extended(2_000, 0x1FFFF, vec![]);
        assert!(f.is_extended);
    }

    // 30. scale_timestamp with zero base
    #[test]
    fn test_scale_timestamp_zero_base() {
        assert_eq!(scale_timestamp(5_000, 0, 1.0), 5_000);
    }

    // 31. scale_timestamp with non-zero base
    #[test]
    fn test_scale_timestamp_nonzero_base() {
        // relative = 5000 - 2000 = 3000; 3000 / 1.0 = 3000
        assert_eq!(scale_timestamp(5_000, 2_000, 1.0), 3_000);
    }

    // 32. scale_timestamp with speed_factor 0 returns 0
    #[test]
    fn test_scale_timestamp_zero_speed() {
        assert_eq!(scale_timestamp(5_000, 0, 0.0), 0);
    }

    // 33. peek_timestamp after seek
    #[test]
    fn test_peek_timestamp_after_seek() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        engine.seek(2_000); // position at index 2
        let ts = engine.peek_timestamp().expect("peek");
        // base_time_us = 0 (first frame ts), speed = 1.0 → scaled_ts = 2000
        assert_eq!(ts, 2_000);
    }

    // 34. replayed count grows with next calls
    #[test]
    fn test_replayed_count_grows() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        engine.next();
        engine.next();
        assert_eq!(engine.stats().frames_replayed, 2);
    }

    // 35. End event is not counted in frames_replayed
    #[test]
    fn test_end_not_counted_in_replayed() {
        let frames = vec![CanFrame::standard(0, 0x1, vec![])];
        let mut engine = ReplayEngine::new(frames, default_config());
        engine.next(); // Frame
        engine.next(); // End
        assert_eq!(engine.stats().frames_replayed, 1);
    }

    // 36. loop_replay: after Loop event, frames can be replayed again
    #[test]
    fn test_loop_replay_continues() {
        let config = ReplayConfig {
            loop_replay: true,
            ..Default::default()
        };
        let frames = vec![
            CanFrame::standard(0, 0x1, vec![]),
            CanFrame::standard(1_000, 0x2, vec![]),
        ];
        let mut engine = ReplayEngine::new(frames, config);
        engine.next(); // Frame 1
        engine.next(); // Frame 2
        let loop_event = engine.next(); // Loop
        assert_eq!(loop_event, ReplayEvent::Loop);
        // Next call should give first frame again
        let first_again = engine.next();
        assert!(matches!(first_again, ReplayEvent::Frame(_)));
    }

    // 37. reset after partial replay
    #[test]
    fn test_reset_partial_replay() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        engine.next();
        engine.next();
        engine.next();
        engine.reset();
        let mut count = 0;
        while let ReplayEvent::Frame(_) = engine.next() {
            count += 1;
        }
        assert_eq!(count, 5);
    }

    // 38. frame_count is 0 for empty list
    #[test]
    fn test_frame_count_empty() {
        let engine = ReplayEngine::new(vec![], default_config());
        assert_eq!(engine.frame_count(), 0);
    }

    // 39. Default ReplayConfig
    #[test]
    fn test_default_replay_config() {
        let cfg = ReplayConfig::default();
        assert_eq!(cfg.speed_factor, 1.0);
        assert!(!cfg.loop_replay);
        assert!(cfg.filter_ids.is_none());
    }

    // 40. seek to 0 returns position 0
    #[test]
    fn test_seek_to_zero() {
        let mut engine = ReplayEngine::new(sample_frames(), default_config());
        engine.next();
        engine.next();
        let pos = engine.seek(0);
        assert_eq!(pos, 0);
    }

    // 41. scaled_duration with zero speed_factor returns 0
    #[test]
    fn test_scaled_duration_zero_speed() {
        let config = ReplayConfig {
            speed_factor: 0.0,
            ..Default::default()
        };
        let engine = ReplayEngine::new(sample_frames(), config);
        assert_eq!(engine.scaled_duration(), 0);
    }

    // 42. Frame data is preserved through replay
    #[test]
    fn test_frame_data_preserved() {
        let frames = vec![CanFrame::standard(0, 0x100, vec![0xDE, 0xAD, 0xBE, 0xEF])];
        let mut engine = ReplayEngine::new(frames, default_config());
        if let ReplayEvent::Frame(f) = engine.next() {
            assert_eq!(f.data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        } else {
            panic!("Expected frame");
        }
    }

    // 43. Multiple filter_ids include multiple IDs
    #[test]
    fn test_filter_multiple_ids() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0x100, 0x200]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        let mut count = 0;
        while let ReplayEvent::Frame(_) = engine.next() {
            count += 1;
        }
        // 0x100 appears twice, 0x200 appears twice = 4 frames
        assert_eq!(count, 4);
    }

    // 44. stats frames_skipped reflects filter exclusions
    #[test]
    fn test_stats_skipped_reflects_filter() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0x300]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config);
        while engine.next() != ReplayEvent::End {}
        assert_eq!(engine.stats().frames_replayed, 1);
        assert_eq!(engine.stats().frames_skipped, 4);
    }

    // 45. load resets skipped count
    #[test]
    fn test_load_resets_stats() {
        let config = ReplayConfig {
            filter_ids: Some(vec![0x100]),
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(sample_frames(), config.clone());
        while engine.next() != ReplayEvent::End {}
        assert!(engine.stats().frames_skipped > 0);
        engine.load(sample_frames());
        assert_eq!(engine.stats().frames_skipped, 0);
        assert_eq!(engine.stats().frames_replayed, 0);
    }
}
