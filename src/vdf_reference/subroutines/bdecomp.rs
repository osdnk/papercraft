use num_traits::ToPrimitive;
use rayon::prelude::*;

use crate::arithmetic::{first_n_columns, parallel_dot_matrix_matrix, ring_very_parallel_dot_matrix_matrix, sample_random_mat, split_into_submatrices_by_columns, zip_columns_horizontally};
use crate::crs::gen_crs;
use crate::helpers::println_with_timestamp;
use crate::r#static::BASE_INT;
use crate::ring_helpers::transpose;
use crate::ring_i128::{get_g, Ring, ring_inner_product, RingElement};

/// Struct representing the decomposition output.
pub struct DecompOutput {
    /// The number of parts the original witness matrix is decomposed into.
    pub(crate) parts: usize,

    /// The resulting right-hand side (RHS) matrix.
    pub(crate) rhs: Vec<Vec<RingElement>>,
}

/// Decomposes a witness matrix based on a given power series matrix, using the maximal infinity norm
/// to determine the number of parts for decomposition.
///
/// # Arguments
///
/// * `power_series` - A reference to the power series matrix, represented as `Vec<Vec<RingElement>>`.
/// * `witness` - A reference to the witness matrix, represented as `Vec<Vec<RingElement>>`.
///
/// # Returns
///
/// A `DecompOutput` struct containing the new witness matrix, the number of parts, and the resulting RHS matrix.
/// ```
pub fn b_decomp(power_series: &Vec<Vec<RingElement>>, witness: &Vec<Vec<RingElement>>, radix: BASE_INT) -> (Vec<Vec<RingElement>>, DecompOutput) {
    use std::time::Instant;

    // Calculate the maximum infinity norm for the witness matrix
    let now = Instant::now();
    let max_inf_norm_witness = witness.par_iter()
        .flat_map(|row| row.par_iter())
        .map(|element| element.inf_norm())
        .max()
        .unwrap();
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to calculate max_inf_norm_witness: {:.2?}", elapsed);

    // Determine the number of parts based on the maximum infinity norm using log2
    let now = Instant::now();
    let parts = (max_inf_norm_witness + 1).to_f64().unwrap().log(radix as f64).ceil().to_usize().unwrap();
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to calculate parts: {:.2?}", elapsed);

    // Transpose the witness matrix for column-wise operations
    let now = Instant::now();
    let witness_transposed = transpose(witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to transpose witness: {:.2?}", elapsed);

    // Decompose each column of the transposed witness matrix
    let now = Instant::now();
    let columns: Vec<Vec<Vec<RingElement>>> = witness_transposed
        .par_iter()
        .map(|col|
        col.par_iter().map(|m| m.g_decompose_coeffs_base(parts, radix)).collect()
        )
        .collect();
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to decompose columns: {:.2?}", elapsed);

    // Zip the decomposed columns horizontally into a new witness matrix
    let now = Instant::now();
    let decomposed_witness = zip_columns_horizontally(&columns);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to zip columns horizontally: {:.2?}", elapsed);

    // Extract relevant columns from the power series matrix to form a submatrix
    let now = Instant::now();
    let power_series_sub = first_n_columns(power_series, witness.len());
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to extract relevant columns from power series: {:.2?}", elapsed);

    // Compute the resulting RHS matrix
    let now = Instant::now();
    let rhs = ring_very_parallel_dot_matrix_matrix(&power_series_sub, &decomposed_witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute RHS matrix: {:.2?}", elapsed);

    (decomposed_witness, DecompOutput {
        parts,
        rhs,
    })
}


#[cfg(test)]
mod b_decomposition_tests {
    use crate::ring_i128::get_g_custom;
    use super::*;
    #[test]
    fn test_b_decomp() {
        let radix = 4;
        let ck = gen_crs().ck;
        let ell = Ring::random();
        let wit = vec![vec![ell]];
        let (new_witness, output) = b_decomp(&ck, &wit, radix);
        let g = get_g_custom(output.parts, radix);
        assert_eq!(ring_inner_product(&new_witness[0], &g), ell);
    }


    #[test]
    fn test_b_decomp_comm() {
        let radix = 2;
        let ck = gen_crs().ck;
        let nrows = 10;
        let ncols = 10;
        let wit = sample_random_mat(nrows, ncols);
        let (new_witness, output) = b_decomp(&ck, &wit, radix);

        let power_series_sub = first_n_columns(&ck, wit.len());

        assert_eq!(output.rhs, parallel_dot_matrix_matrix(&power_series_sub, &new_witness));

        let decomposed = split_into_submatrices_by_columns(&new_witness, wit[0].len());
        let g = get_g_custom(output.parts, radix);


        for i in 0..nrows {
            for j in 0..ncols {
                let entries = (0..output.parts).map(|k| {
                    decomposed[k][i][j]
                }).collect();
                assert_eq!(ring_inner_product(&entries, &g), wit[i][j]);
            }
        }
    }

    #[test]
    fn test_b_decomp_comm_bin() {
        let radix = 2;
        let ck = gen_crs().ck;
        let nrows = 5;
        let ncols = 7;
        let wit = (0..nrows).map(|_| {
            (0..ncols).map(|_| Ring::random_bin()).collect()
        }).collect();

        let (new_witness, output) = b_decomp(&ck, &wit, radix);

        let power_series_sub = first_n_columns(&ck, wit.len());

        assert_eq!(output.rhs, parallel_dot_matrix_matrix(&power_series_sub, &new_witness));

        let decomposed = split_into_submatrices_by_columns(&new_witness, wit[0].len());
        let g = get_g_custom(output.parts, radix);

        println_with_timestamp!("{:?}", output.parts);

        for i in 0..nrows {
            for j in 0..ncols {
                // Create a matrix of size n x m
                let entries = (0..output.parts).map(|k| {
                    decomposed[k][i][j]
                }).collect();

                assert_eq!(ring_inner_product(&entries, &g), wit[i][j]);
            }
        }
    }
}
