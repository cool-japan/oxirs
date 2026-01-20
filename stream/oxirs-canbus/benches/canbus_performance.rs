//! CANbus performance benchmarks
//!
//! Benchmarks CANbus protocol operations including frame parsing,
//! J1939 processing, DBC signal decoding, and RDF triple generation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_canbus::{CanFrame, CanId, J1939Processor, Pgn};

/// Generate test CAN frames
fn generate_can_frames(count: usize) -> Vec<CanFrame> {
    let mut frames = Vec::with_capacity(count);
    for i in 0..count {
        let id = CanId::standard((0x100 + (i % 100)) as u16).unwrap();
        let data: Vec<u8> = (0..8).map(|j| ((i + j) % 256) as u8).collect();
        frames.push(CanFrame::new(id, data).unwrap());
    }
    frames
}

/// Generate J1939 CAN frames
fn generate_j1939_frames(count: usize) -> Vec<CanFrame> {
    let mut frames = Vec::with_capacity(count);
    // EEC1 PGN (61444) frames
    for i in 0..count {
        // 29-bit extended ID for J1939: priority(3) + reserved(1) + DP(1) + PF(8) + PS(8) + SA(8)
        // PGN 61444 = 0xF004 (EEC1)
        // Full ID: 0x0CF00400 (Priority 3, PGN 61444, SA 0)
        let id = CanId::extended(0x0CF00400 + (i % 256) as u32).unwrap();
        let data = vec![
            0xFF,                               // Torque mode
            ((i % 256) as u8),                  // Driver demand
            ((i % 256) as u8),                  // Actual torque
            (((i * 100) % 65536) & 0xFF) as u8, // Engine speed LSB
            (((i * 100) % 65536) >> 8) as u8,   // Engine speed MSB
            0xFF,
            0xFF,
            0xFF,
        ];
        frames.push(CanFrame::new(id, data).unwrap());
    }
    frames
}

/// Benchmark CAN frame creation and parsing
fn bench_frame_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("can_frame");

    for count in [100, 1000, 10000] {
        group.throughput(Throughput::Elements(count as u64));

        // Frame creation
        group.bench_function(BenchmarkId::new("create", count), |b| {
            b.iter(|| {
                let mut frames = Vec::with_capacity(count);
                for i in 0..count {
                    let id = CanId::standard((0x100 + (i % 100)) as u16).unwrap();
                    let data: Vec<u8> = vec![0; 8];
                    frames.push(CanFrame::new(id, data).unwrap());
                }
                black_box(frames)
            });
        });

        // Frame data extraction
        let frames = generate_can_frames(count);
        group.bench_with_input(
            BenchmarkId::new("extract_data", count),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let data: Vec<&[u8]> = frames.iter().map(|f| f.data.as_slice()).collect();
                    black_box(data)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark J1939 protocol processing
fn bench_j1939_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("j1939");

    for count in [100, 1000, 10000] {
        let frames = generate_j1939_frames(count);

        group.throughput(Throughput::Elements(count as u64));

        // PGN extraction from 29-bit CAN ID
        group.bench_with_input(
            BenchmarkId::new("pgn_extract", count),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let pgns: Vec<Option<Pgn>> = frames
                        .iter()
                        .map(|f| {
                            if let CanId::Extended(id) = f.id {
                                let pgn_value = (id >> 8) & 0x3FFFF;
                                Some(Pgn::new(pgn_value))
                            } else {
                                None
                            }
                        })
                        .collect();
                    black_box(pgns)
                });
            },
        );

        // Full J1939 message processing
        group.bench_with_input(
            BenchmarkId::new("process_messages", count),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let mut processor = J1939Processor::new();
                    let mut messages = Vec::with_capacity(count);

                    for frame in frames {
                        if let Some(msg) = processor.process(frame) {
                            messages.push(msg);
                        }
                    }

                    black_box(messages)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark signal extraction from CAN data
fn bench_signal_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_extraction");

    // Test data: 8 bytes of CAN frame data
    let frame_data: [u8; 8] = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];

    group.throughput(Throughput::Elements(1));

    // Extract 16-bit little-endian signal
    group.bench_function("16bit_le", |b| {
        b.iter(|| {
            let start_bit = 0;
            let _length = 16;
            let byte_pos = start_bit / 8;
            let value = u16::from_le_bytes([frame_data[byte_pos], frame_data[byte_pos + 1]]);
            black_box(value)
        });
    });

    // Extract 16-bit big-endian signal
    group.bench_function("16bit_be", |b| {
        b.iter(|| {
            let start_bit = 0;
            let _length = 16;
            let byte_pos = start_bit / 8;
            let value = u16::from_be_bytes([frame_data[byte_pos], frame_data[byte_pos + 1]]);
            black_box(value)
        });
    });

    // Extract arbitrary bit-aligned signal (11 bits starting at bit 5)
    group.bench_function("11bit_unaligned", |b| {
        b.iter(|| {
            let start_bit = 5;
            let length = 11;

            // Manual bit extraction
            let mut value: u32 = 0;
            for i in 0..length {
                let bit_pos = start_bit + i;
                let byte_pos = bit_pos / 8;
                let bit_in_byte = bit_pos % 8;

                if byte_pos < 8 {
                    let bit = (frame_data[byte_pos] >> bit_in_byte) & 1;
                    value |= (bit as u32) << i;
                }
            }
            black_box(value)
        });
    });

    // Extract and scale multiple signals (simulating typical message decode)
    let frame_count = 1000;
    let frames = generate_can_frames(frame_count);

    group.throughput(Throughput::Elements(frame_count as u64));

    group.bench_with_input(
        BenchmarkId::new("decode_full_message", frame_count),
        &frames,
        |b, frames| {
            b.iter(|| {
                let mut decoded = Vec::with_capacity(frame_count * 4);

                for frame in frames {
                    let data = &frame.data;
                    if data.len() >= 8 {
                        // Signal 1: bytes 0-1, factor 0.125, offset 0
                        let raw1 = u16::from_le_bytes([data[0], data[1]]);
                        let phys1 = raw1 as f64 * 0.125;

                        // Signal 2: bytes 2-3, factor 1.0, offset -40
                        let raw2 = u16::from_le_bytes([data[2], data[3]]);
                        let phys2 = raw2 as f64 * 1.0 - 40.0;

                        // Signal 3: bytes 4-5, factor 0.01, offset 0
                        let raw3 = u16::from_le_bytes([data[4], data[5]]);
                        let phys3 = raw3 as f64 * 0.01;

                        // Signal 4: bytes 6-7, factor 0.5, offset 0
                        let raw4 = u16::from_le_bytes([data[6], data[7]]);
                        let phys4 = raw4 as f64 * 0.5;

                        decoded.push((phys1, phys2, phys3, phys4));
                    }
                }

                black_box(decoded)
            });
        },
    );

    group.finish();
}

/// Benchmark CAN ID operations
fn bench_can_id_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("can_id");

    let count = 10000;
    group.throughput(Throughput::Elements(count as u64));

    // Standard ID creation
    group.bench_function(BenchmarkId::new("standard_create", count), |b| {
        b.iter(|| {
            let mut ids = Vec::with_capacity(count);
            for i in 0..count {
                ids.push(CanId::standard((i % 2048) as u16).unwrap());
            }
            black_box(ids)
        });
    });

    // Extended ID creation
    group.bench_function(BenchmarkId::new("extended_create", count), |b| {
        b.iter(|| {
            let mut ids = Vec::with_capacity(count);
            for i in 0..count {
                ids.push(CanId::extended(i as u32 % 0x1FFFFFFF).unwrap());
            }
            black_box(ids)
        });
    });

    // ID comparison
    let ids: Vec<CanId> = (0..count)
        .map(|i| CanId::standard((i % 2048) as u16).unwrap())
        .collect();

    group.bench_with_input(BenchmarkId::new("compare", count), &ids, |b, ids| {
        b.iter(|| {
            let target = CanId::standard(0x100).unwrap();
            let matches: Vec<bool> = ids.iter().map(|id| *id == target).collect();
            black_box(matches)
        });
    });

    group.finish();
}

/// Benchmark message filtering
fn bench_message_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtering");

    for count in [1000, 10000] {
        let frames = generate_can_frames(count);

        group.throughput(Throughput::Elements(count as u64));

        // Single ID filter
        group.bench_with_input(
            BenchmarkId::new("single_id", count),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let target_id = CanId::standard(0x100).unwrap();
                    let filtered: Vec<&CanFrame> =
                        frames.iter().filter(|f| f.id == target_id).collect();
                    black_box(filtered)
                });
            },
        );

        // ID range filter (0x100-0x1FF)
        group.bench_with_input(BenchmarkId::new("id_range", count), &frames, |b, frames| {
            b.iter(|| {
                let filtered: Vec<&CanFrame> = frames
                    .iter()
                    .filter(|f| {
                        if let CanId::Standard(id) = f.id {
                            (0x100..=0x1FF).contains(&id)
                        } else {
                            false
                        }
                    })
                    .collect();
                black_box(filtered)
            });
        });

        // Mask-based filter (bits 8-10 must be 001)
        group.bench_with_input(
            BenchmarkId::new("mask_filter", count),
            &frames,
            |b, frames| {
                b.iter(|| {
                    let mask = 0x700u16;
                    let expected = 0x100u16;

                    let filtered: Vec<&CanFrame> = frames
                        .iter()
                        .filter(|f| {
                            if let CanId::Standard(id) = f.id {
                                (id & mask) == expected
                            } else {
                                false
                            }
                        })
                        .collect();
                    black_box(filtered)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_frame_operations,
    bench_j1939_processing,
    bench_signal_extraction,
    bench_can_id_operations,
    bench_message_filtering
);
criterion_main!(benches);
