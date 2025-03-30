extern crate criterion;
extern crate num_modular;

use criterion::{black_box, Criterion, criterion_group, criterion_main};
use num_modular::{ModularInteger, MontgomeryInt};

fn montgomery_reduction(number: u128, modulus: u128) -> u128 {
    let mont_number = MontgomeryInt::new(number, &modulus);
    mont_number.residue()
}

fn regular_modulus(number: u128, modulus: u128) -> u128 {
    number % modulus
}

fn criterion_benchmark(c: &mut Criterion) {
    let number: u128 = 1_234_567_890_123_456_789_012_345_678;
    let modulus: u128 = 1_000_000_007;

    c.bench_function("Montgomery Reduction", |b| {
        b.iter(|| montgomery_reduction(black_box(number), black_box(modulus)))
    });

    c.bench_function("Regular Modulus", |b| {
        b.iter(|| regular_modulus(black_box(number), black_box(modulus)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
