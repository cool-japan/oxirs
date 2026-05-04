# OxiRS Stream — Engine Overview

This document describes the watermark / window / join / SLA contract that the
oxirs-stream engine offers to operator authors and pipeline builders.

It complements the rustdoc API references for the following modules:

- `oxirs_stream::watermark`
- `oxirs_stream::window::joins`
- `oxirs_stream::aggregation::exactly_once`
- `oxirs_stream::sla`
- `oxirs_stream::adaptive_load_shedding`
- `oxirs_stream::stream_fusion`

## 1. Event time vs. processing time

The engine distinguishes two clocks:

| Concept | Definition | Source |
|---------|------------|--------|
| **event time** | time the event was *produced* by its source | `event.metadata.timestamp` (`i64` ms since epoch) |
| **processing time** | wall clock at the operator | `chrono::Utc::now()` / `std::time::Instant::now()` |

All windowing, watermarking and join semantics in this document are
event-time based.  Wall-clock-driven helpers exist (e.g.
`processing::window::Watermark`) but they live alongside the event-time API
and are clearly marked as such.

## 2. Watermarks

A **watermark** of `T` asserts: *"all subsequent events from this source will
have timestamps `≥ T`"*.

Two watermark APIs are exposed:

- `watermark::WatermarkGenerator` — produces watermarks from an event stream.
  At every `advance_threshold`-th event, emits
  `max_observed_ts − max_out_of_order_ms`.
- `watermark::WatermarkAligner` — combines per-source watermarks using the
  **minimum rule**.  An aligner with N sources reports the global watermark
  as `min(W₁, …, Wₙ)`.  An aligner with zero sources returns `i64::MIN`.

### 2.1 Operator-graph propagation

`watermark::propagation::WatermarkPropagator` runs the minimum rule across
a topology of operators:

```text
     src ─→ map ─→ filter ─→ join ─→ sink
              ↑
       another src
```

Contract:

1. **Per-edge non-decreasing**: each operator's *output* watermark is `≥`
   any previously emitted output watermark on the same edge.  Violations
   surface as `StreamError::WatermarkViolation { operator_id, reason }`.
2. **Aggregation = min**: an operator with N upstreams emits
   `min` over their last-emitted watermarks.
3. **Acyclic**: cycles are unsupported.
4. **Sink global**: the propagator's global watermark is `min` over all
   sinks' last-emitted watermarks.

### 2.2 Late events

`watermark::LateDataHandler` decides what to do with events whose
event-time is below the current watermark:

| `LateDataPolicy` | `LateDataDecision` |
|------------------|--------------------|
| `Drop` | `Drop` |
| `Reassign { max_lateness_ms }` | `Reassign(watermark)` if `lateness ≤ budget`, else `Drop` |
| `SideOutput { channel }` | `SideOutput` |

The `late_handler` submodule additionally provides:

- `AllowedLatenessTracker` — per-window allowed-lateness budget.  A window
  remains "open" until `now ≥ window_end + allowed_lateness`.
- `SideOutputRouter` — accumulates events tagged for side-output channels
  so callers can drain per channel.

## 3. Windowing

Two complementary window implementations exist:

- `processing::window` — wall-clock-driven, used by legacy operators.
- `window` — event-time, watermark-driven (this document).

### 3.1 Tumbling × Tumbling join

`window::joins::TumblingTumblingJoin<L, R>`

- Both streams are bucketed into the same fixed pane size `size_ms`.
- Two events join *iff* they share both join key and pane index.
- A pane `[s, s+size_ms)` is purged when
  `watermark ≥ s + size_ms + allowed_lateness_ms`.
- Late events for already-purged panes are dropped and counted in
  `WindowJoinStats::late_events_dropped`.

### 3.2 Tumbling × Sliding join

`window::joins::TumblingSlidingJoin<L, R>`

- Left stream uses tumbling windows of size `left_size_ms`.
- Right stream uses sliding windows defined by `right_size_ms` /
  `right_slide_ms`.
- An event joins when the right event's timestamp falls within any
  sliding pane that overlaps the left tumbling pane that contains it.
- Right events that occupy multiple sliding panes are de-duplicated at
  emit time so each (left, right) pair appears exactly once.

### 3.3 Session × Session join

`window::joins::SessionSessionJoin<L, R>`

- Both streams use *session* windows defined by an inactivity gap `gap_ms`.
- For each key, sessions accumulate events whose timestamps are within the
  gap of the session's first or last event.
- Two sessions overlap when their gap-extended intervals intersect:
  `[a.first, a.last + gap]  ∩  [b.first, b.last + gap]  ≠  ∅`.
- Emission happens at watermark advance: when **both** sides for a key have
  closed (`end + allowed_lateness ≤ watermark`) the cross-product of
  overlapping sessions is emitted and state is purged.

## 4. Aggregation under operator parallelism

`aggregation::ExactlyOnceAggregator`

A per-partition aggregator that wraps `state::exactly_once::ExactlyOnceProcessor`
to provide:

1. **Idempotent fold** — re-deliveries (same `MessageId`) are filtered.
2. **Checkpointable state** — `checkpoint()` writes the encoded state to the
   backing `StateBackend`; `restore()` reloads it.
3. **Recovery semantics** — after a crash, `restore()` brings back the same
   aggregate values that were committed pre-crash.

Supported aggregate values: `Count`, `Sum`, `Min`, `Max`, `Mean { sum, count }`.

The encoding format is documented in `PartitionAggregateState::encode` and
is deterministic with respect to key ordering.

## 5. SLA admission control

`sla::StreamAdmissionController`

A per-stream wrapper around `oxirs_core::sla::AdmissionController`
(token-bucket).  Configuration per stream:

```rust
struct StreamSlaConfig {
    class: SlaClass,                      // Bronze … Platinum
    max_events_per_sec: f64,              // sustained rate cap
    max_lag: Option<Duration>,            // ingestion-lag budget
    jitter_budget_ms: Option<u64>,        // inter-arrival jitter cap
    token_cost: f64,                      // per-event cost
}
```

Order of checks on every `try_admit`:

1. Lag — `now − event_ts > max_lag`?  Reject with
   `StreamError::SlaExceeded { reason: "lag … exceeds max_lag …" }`.
2. Jitter — `now − last_admit > jitter_budget_ms`?  Reject.
3. Rate — token-bucket consume `token_cost`.  Reject when bucket is empty.

Successful admissions return:

```rust
StreamAdmissionDecision::Admit { tokens_left, lag_ms }
```

### 5.1 SLA × backpressure interplay

`sla::SlaBackpressureCoordinator` fuses SLA admission with the adaptive
load shedder (`adaptive_load_shedding::LoadSheddingManager`).

Per the W2-S6 plan, **SLA reject takes precedence over load shedding**:

```text
                                    ┌─→ Reject (SlaExceeded)
                                    │
SLA admit? ────────► No ────────────┤
                                    │
                  Yes ─→ shedder?  ─┴─→ Drop      (Shed)
                            │       └─→ Throttle  (PreferThrottle policy)
                            │
                          Pass ─────────► Admit { tokens_left, lag_ms }
```

Three policies (`SlaBackpressurePolicy`):

| Policy | Behaviour |
|--------|-----------|
| `Strict` (default) | Shedder Drop ⇒ `Decision::Shed`. |
| `PreferThrottle` | Shedder Drop ⇒ `Decision::Throttle` (caller throttles producer). |
| `BypassShedder` | Shedder consultation skipped. |

## 6. Operator fusion + adaptive batching

The engine offers two performance multipliers for large-scale deployments:

- **Operator fusion** — `stream_fusion::FusionOptimizer` automatically fuses
  adjacent map / filter / map-filter operators where the type system permits,
  eliminating intermediate allocations and function-call overhead.  Configured
  via `FusionConfig { enable_map_fusion, enable_filter_fusion,
  enable_cross_fusion, max_fusion_depth, … }`.
- **Adaptive batching** — `performance_optimizer::batching::AdaptiveBatcher`
  scales batch size based on observed throughput / latency and feeds back a
  `BatchSizePredictor` running over a sliding window of
  `BatchPerformancePoint` samples.

Both optimisations are *transparent*: an operator pipeline opts in via the
`PerformanceConfig` knobs and the fusion + batcher run alongside the
operator's normal execution.  No API changes are required for downstream
operators.

## 7. Putting it all together

A typical Wave-2-S6 pipeline:

```text
  [source]
     │
     ▼
[admission] ── SLA reject ──► reject channel
     │
     ▼
[load shedder] ── drop ──► metrics
     │
     ▼
[event-time fusion + adaptive batching]
     │
     ▼
[watermark generator] ── propagated via WatermarkPropagator
     │
     ├──► [tumbling-tumbling join]
     ├──► [session × session join]
     └──► [exactly-once aggregator]
                  │
                  ▼
             [checkpoint]
                  │
                  ▼
                [sink]
```

Watermarks propagate through every operator on the right-hand path,
maintaining the contract laid out in §2.  Late events are routed per
`LateDataPolicy`.  Aggregator commits flow through the
`ExactlyOnceProcessor` so that re-delivery never double-counts.  All ingress
events first pass admission control, which short-circuits SLA-violating
streams before they reach the load shedder.
