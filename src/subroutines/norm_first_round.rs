use num_traits::ToPrimitive;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use crate::arithmetic::{compose_with_radix, compose_with_radix_mod, decompose_matrix_by_chunks, first_n_columns, join_matrices_horizontally, last_n_columns, parallel_dot_matrix_matrix, parallel_dot_series_matrix, PowerSeries, sample_random_mat, transpose};
use std::time::Instant;
use crate::helpers::println_with_timestamp;
use crate::custom_ring::ring::{DPrimeRingElement, Ring, RingElement};
use crate::slow_ntt::convolution::convolution;
use crate::subroutines::crs::CRS;

/// Struct representing the output of the norm computation.
pub struct Norm1Output {
    pub radix: u64,
    pub new_rhs: Vec<Vec<DPrimeRingElement>>,
}
/// Computes the norm of multiple row witness by convolving each row independently.
///
/// # Arguments
///
/// * `power_series` - Reference to the power series matrix.
/// * `witness` - Reference to the witness matrix.
///
/// # Returns
///
/// A `Norm1Output` containing the convolved witness, the new joined witness, the computed radix,
/// and the new right-hand side matrix.

pub fn norm_1(
    power_series: &Vec<PowerSeries>,
    witness: &Vec<Vec<DPrimeRingElement>>
) -> (Vec<Vec<DPrimeRingElement>>, Norm1Output) {

    let now = Instant::now();
    let witness_transposed = transpose(&witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to transpose witness: {:.2?}", elapsed);

    // Convolve each row independently
    let now = Instant::now();
    let convoluted_witness_transposed: Vec<Vec<DPrimeRingElement>> =
        witness_transposed
            .par_iter()
            .map(convolution)
            .collect();
    let convoluted_witness = transpose(&convoluted_witness_transposed);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to convolve rows: {:.2?}", elapsed);

    // Compute new right-hand side matrix
    let now = Instant::now();
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to get power_series_sub: {:.2?}", elapsed);

    let now = Instant::now();
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute new RHS: {:.2?}", elapsed);

    let (decomposed_witness, radix) = decompose_matrix_by_chunks(&convoluted_witness, 2);
    let new_rhs = parallel_dot_series_matrix(&power_series, &decomposed_witness);
    // Return the result along with the newly computed right-hand side matrix
    (
        join_matrices_horizontally(&witness, &decomposed_witness),
        Norm1Output {
            radix,
            new_rhs,
        }
    )
}

#[test]
fn test_norm_1_multiple_rows() {
    let ck = CRS::gen_crs(3, 2).ck;
    let witness_transposed: Vec<DPrimeRingElement> = vec![Ring::random_short(), Ring::random_short(), Ring::random_short()];
    let witness = transpose(&vec![witness_transposed]);
    let (new_witness,  norm_output) = norm_1(&ck, &witness);
    let decomposed_witness = last_n_columns(&new_witness, 2);
    let convoluted_witness = compose_with_radix(&decomposed_witness, norm_output.radix, 2);
    assert_eq!(convoluted_witness[0].len(), 1);
    assert_eq!(convoluted_witness.len(), witness.len());
}
#[test]
fn test_norm_1() {
    let ck = CRS::gen_crs(3, 2).ck;
    let a = Ring::random_short();
    let b = Ring::random_short();
    let c = Ring::random_short();

    let d = Ring::random_short();
    let e = Ring::random_short();
    let f = Ring::random_short();
    let witness_transposed = vec![a, b, c];
    let witness_transposed_2 = vec![d, e, f];

    let witness = transpose(&vec![witness_transposed, witness_transposed_2]);

    let (new_witness,  norm_output)  = norm_1(&ck, &witness);
    let decomposed_witness = last_n_columns(&new_witness, 4);
    let convoluted_witness = compose_with_radix_mod(&decomposed_witness, norm_output.radix, 2);
    let transposed = transpose(&convoluted_witness);
    assert_eq!(transposed[0], vec![
        a * a.conjugate() + b * b.conjugate() + c * c.conjugate(),
        b.conjugate() * c + a.conjugate() * b,
        a.conjugate() * c,
    ]);

    assert_eq!(transposed[1], vec![
        d * d.conjugate() + e * e.conjugate() + f * f.conjugate(),
        e.conjugate() * f + d.conjugate() * e,
        d.conjugate() * f,
    ]);
}
