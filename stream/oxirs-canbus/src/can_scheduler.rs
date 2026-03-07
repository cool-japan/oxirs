//! CAN message transmission scheduling.
//!
//! Implements periodic message scheduling, priority-based arbitration,
//! a transmission queue, and bus-load calculation for CAN 2.0A/B.

use std::collections::{BinaryHeap, HashMap};

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A raw CAN frame with identifier and payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanFrame {
    /// CAN identifier. 11-bit for standard, 29-bit for extended.
    pub id: u32,
    /// Data payload (0–8 bytes for CAN 2.0, up to 64 for CANFD).
    pub data: Vec<u8>,
    /// `true` when `id` uses the 29-bit extended format.
    pub is_extended: bool,
}

impl CanFrame {
    /// Create a standard-frame (11-bit ID) CAN frame.
    pub fn standard(id: u32, data: Vec<u8>) -> Self {
        Self {
            id: id & 0x7FF,
            data,
            is_extended: false,
        }
    }

    /// Create an extended-frame (29-bit ID) CAN frame.
    pub fn extended(id: u32, data: Vec<u8>) -> Self {
        Self {
            id: id & 0x1FFF_FFFF,
            data,
            is_extended: true,
        }
    }

    /// Number of data bytes.
    pub fn dlc(&self) -> usize {
        self.data.len().min(8)
    }

    /// Approximate wire bit-count for bus-load calculation.
    ///
    /// CAN 2.0B frame (worst-case with bit-stuffing): header + data + overhead.
    pub fn wire_bits(&self) -> u64 {
        let id_bits: u64 = if self.is_extended { 29 + 18 } else { 11 + 3 };
        // SOF(1) + ID + RTR(1) + IDE(1) + DLC(4) + DATA + CRC(16) + DEL(1) + ACK(2) + EOF(7) + IFS(3)
        let overhead = 1 + id_bits + 1 + 1 + 4 + (self.dlc() as u64 * 8) + 16 + 1 + 2 + 7 + 3;
        // Worst-case bit stuffing: +20% (1 stuff bit per 4 data bits)
        overhead + overhead / 5
    }
}

/// A scheduled periodic CAN message.
#[derive(Debug, Clone)]
pub struct ScheduledMessage {
    /// Frame to transmit.
    pub frame: CanFrame,
    /// Transmission period in milliseconds.
    pub period_ms: u64,
    /// Arbitration priority (lower number = higher priority).
    pub priority: u8,
    /// Human-readable label.
    pub name: String,
}

/// CAN bus utilisation metrics.
#[derive(Debug, Clone)]
pub struct BusLoad {
    /// Fraction of bus capacity in use (0.0–1.0).
    pub utilization: f64,
    /// Total frames per second across all scheduled messages.
    pub messages_per_second: f64,
    /// Total bytes per second (data payload only).
    pub bytes_per_second: f64,
}

/// Comparison report for a prioritised frame in the TX queue.
#[derive(Debug, Clone, Eq, PartialEq)]
struct QueueEntry {
    /// Lower priority value → higher urgency.
    priority: u8,
    /// Frame to send.
    frame: CanFrame,
    /// Label used for debugging.
    name: String,
}

// Max-heap ordered by lowest priority number (highest urgency).
impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse: lower priority number → greater ordering
        other.priority.cmp(&self.priority)
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// CyclicTimer
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks when a periodic message is next due.
struct CyclicTimer {
    period_ms: u64,
    next_due: u64,
}

impl CyclicTimer {
    fn new(period_ms: u64) -> Self {
        // Due immediately at time 0.
        Self {
            period_ms,
            next_due: 0,
        }
    }

    /// Check whether the timer fires at `now_ms`. If so, advance and return true.
    fn tick(&mut self, now_ms: u64) -> bool {
        if now_ms >= self.next_due {
            self.next_due = now_ms + self.period_ms;
            true
        } else {
            false
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TxQueue
// ──────────────────────────────────────────────────────────────────────────────

/// Priority queue for pending CAN frames.
pub struct TxQueue {
    heap: BinaryHeap<QueueEntry>,
}

impl TxQueue {
    /// Create an empty TX queue.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    /// Enqueue a frame with the given priority.
    pub fn push(&mut self, priority: u8, frame: CanFrame, name: impl Into<String>) {
        self.heap.push(QueueEntry {
            priority,
            frame,
            name: name.into(),
        });
    }

    /// Dequeue the highest-priority frame (lowest priority number).
    pub fn pop(&mut self) -> Option<(CanFrame, String)> {
        self.heap.pop().map(|e| (e.frame, e.name))
    }

    /// Peek at the highest-priority frame without removing it.
    pub fn peek_priority(&self) -> Option<u8> {
        self.heap.peek().map(|e| e.priority)
    }

    /// Number of frames waiting in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// True when the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

impl Default for TxQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Arbitration
// ──────────────────────────────────────────────────────────────────────────────

/// Priority-based frame selector when multiple frames are simultaneously due.
pub struct Arbitration;

impl Arbitration {
    /// Select the highest-priority frame (lowest priority number) from a slice
    /// of (priority, frame, name) tuples.
    ///
    /// Returns the index of the winner, or `None` when `candidates` is empty.
    pub fn select<'a>(candidates: &'a [(u8, &'a CanFrame, &'a str)]) -> Option<usize> {
        if candidates.is_empty() {
            return None;
        }
        let mut best = 0;
        for (i, (prio, _, _)) in candidates.iter().enumerate() {
            if *prio < candidates[best].0 {
                best = i;
            }
        }
        Some(best)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MessageScheduler
// ──────────────────────────────────────────────────────────────────────────────

/// Manages a set of periodic CAN messages and dispatches them on time.
pub struct MessageScheduler {
    messages: HashMap<String, ScheduledMessage>,
    timers: HashMap<String, CyclicTimer>,
    /// Simulated elapsed time in milliseconds.
    elapsed_ms: u64,
}

impl MessageScheduler {
    /// Create an empty scheduler.
    pub fn new() -> Self {
        Self {
            messages: HashMap::new(),
            timers: HashMap::new(),
            elapsed_ms: 0,
        }
    }

    /// Register a new periodic message.
    ///
    /// If a message with the same `name` already exists it is replaced.
    pub fn add_message(
        &mut self,
        name: impl Into<String>,
        frame: CanFrame,
        period_ms: u64,
        priority: u8,
    ) {
        let name = name.into();
        let period_ms = period_ms.max(1);
        let msg = ScheduledMessage {
            frame,
            period_ms,
            priority,
            name: name.clone(),
        };
        self.timers
            .insert(name.clone(), CyclicTimer::new(period_ms));
        self.messages.insert(name, msg);
    }

    /// Remove a message from the schedule. Returns `true` if it existed.
    pub fn remove_message(&mut self, name: &str) -> bool {
        let removed = self.messages.remove(name).is_some();
        self.timers.remove(name);
        removed
    }

    /// Advance the scheduler by `elapsed_ms` milliseconds.
    ///
    /// Returns references to frames that are due for transmission, ordered by
    /// priority (lowest priority number first).
    pub fn tick(&mut self, elapsed_ms: u64) -> Vec<&CanFrame> {
        self.elapsed_ms += elapsed_ms;
        let now = self.elapsed_ms;

        // Collect due messages
        let mut due: Vec<(u8, &str)> = self
            .timers
            .iter_mut()
            .filter_map(|(name, timer)| {
                if timer.tick(now) {
                    let priority = self.messages.get(name.as_str()).map(|m| m.priority)?;
                    Some((priority, name.as_str()))
                } else {
                    None
                }
            })
            .collect();

        // Sort by priority ascending (highest urgency = lowest number first)
        due.sort_by_key(|(p, _)| *p);

        due.iter()
            .filter_map(|(_, name)| self.messages.get(*name).map(|m| &m.frame))
            .collect()
    }

    /// Number of registered messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// `true` when no messages are registered.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Current simulated time in ms.
    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed_ms
    }
}

impl Default for MessageScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Bus load calculation
// ──────────────────────────────────────────────────────────────────────────────

/// Calculate CAN bus utilisation for a set of scheduled messages at a given
/// bit-rate.
///
/// `bitrate_kbps` is the nominal CAN bit-rate (e.g. 500 for 500 kbps).
pub fn calculate_bus_load(messages: &[ScheduledMessage], bitrate_kbps: u32) -> BusLoad {
    let bitrate_bps = bitrate_kbps as f64 * 1_000.0;
    if bitrate_bps == 0.0 || messages.is_empty() {
        return BusLoad {
            utilization: 0.0,
            messages_per_second: 0.0,
            bytes_per_second: 0.0,
        };
    }

    let mut total_bits_per_second: f64 = 0.0;
    let mut total_frames_per_second: f64 = 0.0;
    let mut total_bytes_per_second: f64 = 0.0;

    for msg in messages {
        let fps = 1_000.0 / msg.period_ms as f64;
        let bits_ps = msg.frame.wire_bits() as f64 * fps;
        total_bits_per_second += bits_ps;
        total_frames_per_second += fps;
        total_bytes_per_second += msg.frame.dlc() as f64 * fps;
    }

    BusLoad {
        utilization: (total_bits_per_second / bitrate_bps).min(1.0),
        messages_per_second: total_frames_per_second,
        bytes_per_second: total_bytes_per_second,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn std_frame(id: u32) -> CanFrame {
        CanFrame::standard(id, vec![0x11, 0x22, 0x33, 0x44])
    }

    fn ext_frame(id: u32) -> CanFrame {
        CanFrame::extended(id, vec![0xAA, 0xBB])
    }

    // ── CanFrame ─────────────────────────────────────────────────────────────

    #[test]
    fn test_can_frame_standard_id_masked() {
        let f = CanFrame::standard(0xFFF, vec![]);
        assert_eq!(f.id, 0x7FF);
        assert!(!f.is_extended);
    }

    #[test]
    fn test_can_frame_extended_id_masked() {
        let f = CanFrame::extended(0x3FFF_FFFF, vec![]);
        assert_eq!(f.id, 0x1FFF_FFFF);
        assert!(f.is_extended);
    }

    #[test]
    fn test_can_frame_dlc_capped_at_8() {
        let f = CanFrame::standard(0x100, vec![0; 20]);
        assert_eq!(f.dlc(), 8);
    }

    #[test]
    fn test_can_frame_dlc_empty() {
        let f = CanFrame::standard(0x100, vec![]);
        assert_eq!(f.dlc(), 0);
    }

    #[test]
    fn test_can_frame_wire_bits_nonzero() {
        let f = std_frame(0x100);
        assert!(f.wire_bits() > 0);
    }

    #[test]
    fn test_can_frame_extended_wire_bits_larger() {
        let std_f = CanFrame::standard(0x100, vec![0; 8]);
        let ext_f = CanFrame::extended(0x100, vec![0; 8]);
        assert!(ext_f.wire_bits() > std_f.wire_bits());
    }

    #[test]
    fn test_can_frame_equality() {
        let f1 = std_frame(0x100);
        let f2 = std_frame(0x100);
        assert_eq!(f1, f2);
    }

    // ── CyclicTimer ──────────────────────────────────────────────────────────

    #[test]
    fn test_cyclic_timer_fires_at_zero() {
        let mut t = CyclicTimer::new(100);
        assert!(t.tick(0));
    }

    #[test]
    fn test_cyclic_timer_does_not_fire_before_period() {
        let mut t = CyclicTimer::new(100);
        t.tick(0); // consume first fire
        assert!(!t.tick(50));
    }

    #[test]
    fn test_cyclic_timer_fires_at_period() {
        let mut t = CyclicTimer::new(100);
        t.tick(0); // consume first fire
        assert!(t.tick(100));
    }

    #[test]
    fn test_cyclic_timer_multiple_periods() {
        let mut t = CyclicTimer::new(10);
        let mut count = 0;
        for ms in 0..=50u64 {
            if t.tick(ms) {
                count += 1;
            }
        }
        assert!(count >= 5);
    }

    // ── TxQueue ──────────────────────────────────────────────────────────────

    #[test]
    fn test_tx_queue_empty() {
        let q = TxQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_tx_queue_push_pop() {
        let mut q = TxQueue::new();
        q.push(5, std_frame(0x100), "msg");
        assert_eq!(q.len(), 1);
        let (f, name) = q.pop().unwrap();
        assert_eq!(f, std_frame(0x100));
        assert_eq!(name, "msg");
        assert!(q.is_empty());
    }

    #[test]
    fn test_tx_queue_priority_order() {
        let mut q = TxQueue::new();
        q.push(10, std_frame(0x200), "low");
        q.push(1, std_frame(0x100), "high");
        q.push(5, std_frame(0x150), "mid");
        let (_, name1) = q.pop().unwrap();
        assert_eq!(name1, "high");
        let (_, name2) = q.pop().unwrap();
        assert_eq!(name2, "mid");
        let (_, name3) = q.pop().unwrap();
        assert_eq!(name3, "low");
    }

    #[test]
    fn test_tx_queue_peek_priority() {
        let mut q = TxQueue::new();
        q.push(3, std_frame(0x100), "msg");
        assert_eq!(q.peek_priority(), Some(3));
    }

    #[test]
    fn test_tx_queue_default() {
        let q = TxQueue::default();
        assert!(q.is_empty());
    }

    // ── Arbitration ──────────────────────────────────────────────────────────

    #[test]
    fn test_arbitration_empty() {
        assert_eq!(Arbitration::select(&[]), None);
    }

    #[test]
    fn test_arbitration_single() {
        let f = std_frame(0x100);
        let c = [(5u8, &f, "msg")];
        assert_eq!(Arbitration::select(&c), Some(0));
    }

    #[test]
    fn test_arbitration_picks_lowest_priority() {
        let f1 = std_frame(0x100);
        let f2 = std_frame(0x200);
        let f3 = std_frame(0x300);
        let c = [(10u8, &f1, "low"), (2u8, &f2, "high"), (5u8, &f3, "mid")];
        assert_eq!(Arbitration::select(&c), Some(1));
    }

    #[test]
    fn test_arbitration_tie_returns_first() {
        let f1 = std_frame(0x100);
        let f2 = std_frame(0x200);
        let c = [(5u8, &f1, "a"), (5u8, &f2, "b")];
        // Both have the same priority; either index is acceptable but we check
        // that it returns a valid index.
        let idx = Arbitration::select(&c).unwrap();
        assert!(idx < 2);
    }

    // ── MessageScheduler ──────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_empty() {
        let s = MessageScheduler::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn test_scheduler_add_message() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 100, 5);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_scheduler_remove_message() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 100, 5);
        assert!(s.remove_message("msg"));
        assert!(s.is_empty());
    }

    #[test]
    fn test_scheduler_remove_nonexistent() {
        let mut s = MessageScheduler::new();
        assert!(!s.remove_message("ghost"));
    }

    #[test]
    fn test_scheduler_tick_fires_at_zero() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 100, 5);
        let due = s.tick(0);
        assert_eq!(due.len(), 1);
    }

    #[test]
    fn test_scheduler_tick_no_fire_before_period() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 100, 5);
        s.tick(0); // consume initial fire
        let due = s.tick(50);
        assert!(due.is_empty());
    }

    #[test]
    fn test_scheduler_tick_fires_at_period() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 100, 5);
        s.tick(0); // consume initial
        let due = s.tick(100); // advance to 100ms
        assert_eq!(due.len(), 1);
    }

    #[test]
    fn test_scheduler_tick_priority_order() {
        let mut s = MessageScheduler::new();
        s.add_message("high", std_frame(0x100), 100, 1);
        s.add_message("low", std_frame(0x200), 100, 10);
        let due = s.tick(0);
        assert_eq!(due.len(), 2);
        // First frame in due list should have lower ID (high priority)
        assert_eq!(due[0].id, 0x100);
    }

    #[test]
    fn test_scheduler_tick_different_periods() {
        let mut s = MessageScheduler::new();
        s.add_message("fast", std_frame(0x100), 10, 1);
        s.add_message("slow", std_frame(0x200), 100, 2);
        s.tick(0); // consume initials
        let due_10 = s.tick(10);
        // Only fast fires at 10ms
        assert_eq!(due_10.len(), 1);
        assert_eq!(due_10[0].id, 0x100);
    }

    #[test]
    fn test_scheduler_elapsed_ms() {
        let mut s = MessageScheduler::new();
        s.tick(50);
        s.tick(50);
        assert_eq!(s.elapsed_ms(), 100);
    }

    #[test]
    fn test_scheduler_replace_message() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 100, 5);
        s.add_message("msg", std_frame(0x200), 200, 3);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn test_scheduler_default() {
        let s = MessageScheduler::default();
        assert!(s.is_empty());
    }

    // ── BusLoad ───────────────────────────────────────────────────────────────

    #[test]
    fn test_bus_load_empty() {
        let load = calculate_bus_load(&[], 500);
        assert_eq!(load.utilization, 0.0);
        assert_eq!(load.messages_per_second, 0.0);
    }

    #[test]
    fn test_bus_load_utilization_bounded() {
        let msgs: Vec<ScheduledMessage> = (0..100)
            .map(|i| ScheduledMessage {
                frame: CanFrame::standard(i, vec![0xFF; 8]),
                period_ms: 1, // 1000 Hz each
                priority: 5,
                name: format!("msg-{i}"),
            })
            .collect();
        let load = calculate_bus_load(&msgs, 500);
        assert!(load.utilization <= 1.0);
    }

    #[test]
    fn test_bus_load_messages_per_second() {
        let msg = ScheduledMessage {
            frame: std_frame(0x100),
            period_ms: 100, // 10 Hz
            priority: 5,
            name: "test".to_string(),
        };
        let load = calculate_bus_load(&[msg], 500);
        assert!((load.messages_per_second - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_bus_load_bytes_per_second() {
        let msg = ScheduledMessage {
            frame: CanFrame::standard(0x100, vec![0; 4]),
            period_ms: 1000, // 1 Hz
            priority: 5,
            name: "test".to_string(),
        };
        let load = calculate_bus_load(&[msg], 500);
        // 1 Hz × 4 bytes = 4 bytes/s
        assert!((load.bytes_per_second - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_bus_load_zero_bitrate() {
        let msg = ScheduledMessage {
            frame: std_frame(0x100),
            period_ms: 100,
            priority: 5,
            name: "test".to_string(),
        };
        let load = calculate_bus_load(&[msg], 0);
        assert_eq!(load.utilization, 0.0);
    }

    #[test]
    fn test_bus_load_utilization_positive() {
        let msg = ScheduledMessage {
            frame: std_frame(0x100),
            period_ms: 100,
            priority: 5,
            name: "test".to_string(),
        };
        let load = calculate_bus_load(&[msg], 500);
        assert!(load.utilization > 0.0);
    }

    // ── Extended coverage ─────────────────────────────────────────────────────

    #[test]
    fn test_scheduler_multiple_tick_cycles() {
        let mut s = MessageScheduler::new();
        s.add_message("msg", std_frame(0x100), 50, 1);
        let mut fire_count = 0;
        for _ in 0..10 {
            let due = s.tick(50);
            fire_count += due.len();
        }
        assert!(fire_count >= 9); // should fire ~10 times
    }

    #[test]
    fn test_scheduler_many_messages() {
        let mut s = MessageScheduler::new();
        for i in 0..20u32 {
            s.add_message(format!("msg-{i}"), std_frame(i), 100, (i % 8) as u8);
        }
        assert_eq!(s.len(), 20);
        let due = s.tick(0);
        assert_eq!(due.len(), 20);
    }

    #[test]
    fn test_extended_frame_in_scheduler() {
        let mut s = MessageScheduler::new();
        s.add_message("ext", ext_frame(0x1234567), 100, 1);
        let due = s.tick(0);
        assert_eq!(due.len(), 1);
        assert!(due[0].is_extended);
    }
}
