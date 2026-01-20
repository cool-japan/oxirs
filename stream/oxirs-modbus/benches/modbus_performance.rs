//! Modbus performance benchmarks
//!
//! Benchmarks Modbus protocol operations including frame parsing,
//! CRC calculation, register mapping, and RDF triple generation.

use chrono::Utc;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use oxirs_modbus::mapping::{ModbusDataType, RegisterMap, RegisterMapping};
use oxirs_modbus::protocol::{calculate_crc, verify_crc};
use oxirs_modbus::rdf::{ModbusTripleGenerator, QudtUnit};
use std::collections::HashMap;

/// Generate test register values
fn generate_register_values(count: usize) -> Vec<u16> {
    (0..count).map(|i| (i as u16).wrapping_mul(7)).collect()
}

/// Benchmark CRC-16 calculation
fn bench_crc_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("crc16");

    for size in [8, 64, 256] {
        let data: Vec<u8> = (0..size).map(|i| i as u8).collect();

        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("calculate", size), &data, |b, data| {
            b.iter(|| {
                let crc = calculate_crc(black_box(data));
                black_box(crc)
            });
        });

        // CRC verification
        let mut data_with_crc = data.clone();
        let crc = calculate_crc(&data);
        data_with_crc.push((crc & 0xFF) as u8);
        data_with_crc.push((crc >> 8) as u8);

        group.bench_with_input(
            BenchmarkId::new("verify", size),
            &data_with_crc,
            |b, data| {
                b.iter(|| {
                    let valid = verify_crc(black_box(data));
                    black_box(valid)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark register value decoding
fn bench_register_decoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("register_decoding");

    // UINT16 decoding
    let uint16_regs = generate_register_values(1000);
    group.throughput(Throughput::Elements(1000));

    group.bench_with_input(BenchmarkId::new("uint16", 1000), &uint16_regs, |b, regs| {
        b.iter(|| {
            let values: Vec<u16> = regs.to_vec();
            black_box(values)
        });
    });

    // INT16 decoding (interpret as signed)
    group.bench_with_input(BenchmarkId::new("int16", 1000), &uint16_regs, |b, regs| {
        b.iter(|| {
            let values: Vec<i16> = regs.iter().map(|&r| r as i16).collect();
            black_box(values)
        });
    });

    // FLOAT32 decoding (pairs of registers)
    let float32_regs: Vec<[u16; 2]> = uint16_regs
        .chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| [c[0], c[1]])
        .collect();

    group.bench_with_input(
        BenchmarkId::new("float32", float32_regs.len()),
        &float32_regs,
        |b, regs| {
            b.iter(|| {
                let values: Vec<f32> = regs
                    .iter()
                    .map(|pair| {
                        let bytes = [
                            (pair[0] >> 8) as u8,
                            (pair[0] & 0xFF) as u8,
                            (pair[1] >> 8) as u8,
                            (pair[1] & 0xFF) as u8,
                        ];
                        f32::from_be_bytes(bytes)
                    })
                    .collect();
                black_box(values)
            });
        },
    );

    group.finish();
}

/// Benchmark register map creation and lookup
fn bench_register_map(c: &mut Criterion) {
    let mut group = c.benchmark_group("register_map");

    for num_registers in [10, 100, 500] {
        // Create register map
        let mut map = RegisterMap::new("device001", "http://example.com/device");
        for i in 0..num_registers {
            map.add_register(
                RegisterMapping::new(
                    i as u16,
                    ModbusDataType::Uint16,
                    format!("http://example.com/property/reg{}", i),
                )
                .with_name(format!("Register {}", i)),
            );
        }

        group.throughput(Throughput::Elements(num_registers as u64));

        // Benchmark batch read optimization
        group.bench_with_input(
            BenchmarkId::new("batch_optimization", num_registers),
            &map,
            |b, map| {
                b.iter(|| {
                    let batches =
                        map.batch_reads(oxirs_modbus::mapping::RegisterType::Holding, 125);
                    black_box(batches)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark RDF triple generation
fn bench_triple_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("triple_generation");

    for num_registers in [10, 50, 100] {
        // Create register map
        let mut map = RegisterMap::new("device001", "http://example.com/device");
        for i in 0..num_registers {
            map.add_register(
                RegisterMapping::new(
                    i as u16,
                    ModbusDataType::Uint16,
                    format!("http://example.com/property/reg{}", i),
                )
                .with_name(format!("Register {}", i))
                .with_unit(QudtUnit::celsius()),
            );
        }

        // Create register values
        let mut values: HashMap<u16, Vec<u16>> = HashMap::new();
        for i in 0..num_registers {
            values.insert(i as u16, vec![(i as u16 * 10) + 200]);
        }

        let mut generator = ModbusTripleGenerator::new(map);

        group.throughput(Throughput::Elements(num_registers as u64));

        group.bench_with_input(
            BenchmarkId::new("generate", num_registers),
            &values,
            |b, vals| {
                b.iter(|| {
                    let triples = generator
                        .generate_triples(
                            black_box(vals),
                            oxirs_modbus::mapping::RegisterType::Holding,
                            Utc::now(),
                        )
                        .unwrap();
                    black_box(triples)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scaling and deadband operations
fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");

    let raw_values: Vec<u16> = (0..1000).map(|i| i as u16).collect();

    group.throughput(Throughput::Elements(1000));

    // Linear scaling: physical = raw * 0.1 + 10.0
    group.bench_with_input(
        BenchmarkId::new("linear_scale", 1000),
        &raw_values,
        |b, values| {
            b.iter(|| {
                let scaled: Vec<f64> = values.iter().map(|&v| (v as f64) * 0.1 + 10.0).collect();
                black_box(scaled)
            });
        },
    );

    // Deadband filtering (only emit if change > 5.0)
    let mut prev_values: Vec<f64> = raw_values.iter().map(|&v| v as f64 * 0.1).collect();
    // Slightly modify some values
    for (i, v) in prev_values.iter_mut().enumerate() {
        if i % 3 == 0 {
            *v += 10.0; // Outside deadband
        } else {
            *v += 2.0; // Within deadband
        }
    }

    group.bench_with_input(
        BenchmarkId::new("deadband_filter", 1000),
        &raw_values,
        |b, values| {
            b.iter(|| {
                let deadband = 5.0;
                let mut emit_count = 0;

                for (i, &raw) in values.iter().enumerate() {
                    let current = raw as f64 * 0.1;
                    let prev = prev_values[i];
                    if (current - prev).abs() > deadband {
                        emit_count += 1;
                    }
                }

                black_box(emit_count)
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_crc_calculation,
    bench_register_decoding,
    bench_register_map,
    bench_triple_generation,
    bench_scaling
);
criterion_main!(benches);
