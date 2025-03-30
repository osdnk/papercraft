use num_traits::ToPrimitive;
use rayon::iter::IntoParallelRefIterator;
use rayon::prelude::*;
use crate::arithmetic::{first_n_columns, join_matrices_horizontally, last_n_columns, parallel_dot_matrix_matrix, sample_random_mat, split_into_submatrices_by_columns, zip_columns_horizontally};
use crate::crs::gen_crs;
use crate::subroutines::convolution::convolution;
use crate::r#static::BASE_INT;
use crate::ring_helpers::transpose;
use crate::ring_i128::{Ring, ring_inner_product, RingElement};
use crate::static_rings::static_generated::PHI;

/// Struct representing the output of the norm computation.
pub struct Norm1Output {
    pub convoluted_witness: Vec<Vec<RingElement>>,
    pub radix: BASE_INT,
    pub new_rhs: Vec<Vec<RingElement>>,
}

/// Performs square root decomposition on the transposed witness matrix to balance the norm.
///
/// # Arguments
///
/// * `witness_transposed` - The transposed witness matrix.
///
/// # Returns
///
/// A tuple containing the decomposed witness matrix and the computed radix.
fn sqrt_decomp(witness_transposed: &Vec<Vec<RingElement>>) -> (Vec<Vec<RingElement>>, BASE_INT) {
    // Calculate the maximum infinity norm of the elements in the transposed witness matrix.
    let max_inf_norm_witness = witness_transposed.par_iter()
        .flat_map(|row| row.par_iter())
        .map(|element| element.inf_norm())
        .max()
        .unwrap();

    // Compute the radix for decomposition.
    let radix = (max_inf_norm_witness + 1).to_f64().unwrap().sqrt().ceil().to_i128().unwrap();

    let parts = 2;

    // Decompose the columns based on the radix.
    let columns: Vec<Vec<Vec<RingElement>>> = witness_transposed
        .par_iter()
        .map(|col| col.par_iter().map(|m| m.g_decompose_coeffs_base(parts, radix)).collect())
        .collect();

    // Zip the decomposed columns horizontally into a new witness matrix.
    let decomposed_witness = zip_columns_horizontally(&columns);
    (decomposed_witness, radix)
}

#[test]
fn test_sqrt_decomp() {
    let nrows = 10;
    let ncols = 10;
    let wit = sample_random_mat(nrows, ncols);
    let (decomposed_witness, radix) = sqrt_decomp(&transpose(&wit));

    let decomposed = split_into_submatrices_by_columns(&decomposed_witness, wit[0].len());
    let g = vec![Ring::constant(radix), Ring::constant(1)];

    for i in 0..nrows {
        for j in 0..ncols {
            let entries = (0..2).map(|k| decomposed[k][i][j]).collect();
            assert_eq!(ring_inner_product(&entries, &g), wit[i][j]);
        }
    }
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
use std::time::Instant;
use crate::helpers::println_with_timestamp;

pub fn norm_1(
    power_series: &Vec<Vec<RingElement>>,
    witness: &Vec<Vec<RingElement>>
) -> (Vec<Vec<RingElement>>, Norm1Output) {

    let now = Instant::now();
    let witness_transposed = transpose(&witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to transpose witness: {:.2?}", elapsed);

    // Convolve each row independently
    let now = Instant::now();
    let convoluted_witness_transposed: Vec<Vec<RingElement>> = last_n_columns(
        &witness_transposed
            .par_iter()
            .map(|witness| {
                let convoluted = convolution(&witness);

                convoluted.iter().map(|w| {
                    RingElement {
                        coeffs: <[BASE_INT; PHI]>::try_from(w.clone()).unwrap(),
                    }
                }).collect::<Vec<RingElement>>()
            })
            .collect::<Vec<Vec<RingElement>>>(),
        witness_transposed[0].len()
    );
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to convolve rows: {:.2?}", elapsed);

    // Perform square root decomposition
    let now = Instant::now();
    let (decomposed_witness, radix) = sqrt_decomp(&convoluted_witness_transposed);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time for square root decomposition: {:.2?}", elapsed);

    // Compute new right-hand side matrix
    let now = Instant::now();
    let power_series_sub = first_n_columns(power_series, decomposed_witness.len());
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to get power_series_sub: {:.2?}", elapsed);

    let now = Instant::now();
    let new_rhs = parallel_dot_matrix_matrix(&power_series_sub, &decomposed_witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute new RHS: {:.2?}", elapsed);

    // Return the result along with the newly computed right-hand side matrix
    (
        join_matrices_horizontally(&witness, &decomposed_witness),
        Norm1Output {
            convoluted_witness: transpose(&convoluted_witness_transposed),
            radix,
            new_rhs,
        }
    )
}

#[test]
fn test_norm_1_single_row() {
    let ck = gen_crs().ck;
    let witness: Vec<RingElement> = vec![Ring::random_bin(), Ring::random_bin(), Ring::random_bin()];
    let witness = transpose(&vec![witness.clone()]);
    let (_,  norm_output) = norm_1(&ck, &witness);
    assert_eq!(norm_output.convoluted_witness[0].len(), 1);
    assert_eq!(norm_output.convoluted_witness.len(), witness.len());
}

#[test]
fn test_norm_1_multiple_rows() {
    let ck = gen_crs().ck;
    let witness1: Vec<RingElement> = vec![Ring::random_bin(), Ring::random_bin(), Ring::random_bin()];
    let witness2: Vec<RingElement> = vec![Ring::random_bin(), Ring::random_bin(), Ring::random_bin()];
    let witness = transpose(&vec![witness1.clone(), witness2.clone()]);
    let (_,  norm_output)  = norm_1(&ck, &witness);
    assert_eq!(norm_output.convoluted_witness[0].len(), 2);
    assert_eq!(norm_output.convoluted_witness.len(), witness1.len());
    assert_eq!(norm_output.convoluted_witness.len(), witness2.len());
}

#[test]
fn test_norm_1() {
    let ck = gen_crs().ck;
    let wit = vec![Ring::random(), Ring::random()];
    let (_,  norm_output)  = norm_1(&ck, &transpose(&vec![wit.clone()]));
    let transposed = transpose(&norm_output.convoluted_witness);
    assert_eq!(transposed[0], vec![&wit[1] * &wit[1].conjugate() + &wit[0] * &wit[0].conjugate(), &wit[1] * &wit[0].conjugate()]);
}

#[test]
fn test_norm_decomp() {
    let ck = gen_crs().ck;
    let nrows = 10;
    let ncols = 10;
    let wit = (0..nrows).map(|_| {
        (0..ncols).map(|_| Ring::random()).collect()
    }).collect();

    let (new_witness,  output)  = norm_1(&ck, &wit);

    let decomposed = split_into_submatrices_by_columns(&last_n_columns(&new_witness, 2 * ncols), wit[0].len());
    let g = vec![Ring::constant(output.radix), Ring::constant(1)];

    for i in 0..nrows {
        for j in 0..ncols {
            // Create a matrix of size n x m
            let entries = (0..2).map(|k| decomposed[k][i][j]).collect();
            assert_eq!(ring_inner_product(&entries, &g), output.convoluted_witness[i][j]);
        }
    }

    let power_series_sub = first_n_columns(&ck, nrows);
    let new_rhs = parallel_dot_matrix_matrix(&power_series_sub, &last_n_columns(&new_witness, 2 * ncols));
    assert_eq!(new_rhs, output.new_rhs);
}
