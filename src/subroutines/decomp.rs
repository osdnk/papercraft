use num_traits::ToPrimitive;
use rayon::prelude::*;
use crate::arithmetic::{decompose_matrix_by_chunks, decompose_matrix_by_radix, parallel_dot_matrix_matrix, parallel_dot_series_matrix, PowerSeries};
use crate::helpers::println_with_timestamp;
use crate::custom_ring::ring::{DPrimeRingElement, RingElement};

/// Struct representing the decomposition output.
pub struct DecompOutput {
    /// The number of parts the original witness matrix is decomposed into.
    pub(crate) parts: usize,

    /// The resulting right-hand side (RHS) matrix.
    pub(crate) rhs: Vec<Vec<DPrimeRingElement>>,
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
pub fn decomp(power_series: &Vec<PowerSeries>, witness: &Vec<Vec<DPrimeRingElement>>, radix: u64) -> (Vec<Vec<DPrimeRingElement>>, DecompOutput) {
    use std::time::Instant;
    // Decompose each column of the decomposition of the witness matrix
    let now = Instant::now();
    let (new_witness, parts) = decompose_matrix_by_radix(&witness, radix);
    let elapsed = now.elapsed();

    println_with_timestamp!("  Time to decompose witness: {:.2?}", elapsed);

    // Extract relevant columns from the power series matrix to form a submatrix
    let now = Instant::now();
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to extract relevant columns from power series: {:.2?}", elapsed);

    // Compute the resulting RHS matrix
    let now = Instant::now();
    let rhs = parallel_dot_series_matrix(&power_series, &new_witness);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute RHS matrix: {:.2?}", elapsed);

    (new_witness, DecompOutput {
        parts,
        rhs,
    })
}


#[cfg(test)]
mod b_decomposition_tests {
    use crate::arithmetic::{compose_with_radix, compose_with_radix_mod, sample_random_mat, sample_short_random_mat};
    use crate::custom_ring::r#static::MOD_Q;
    use crate::custom_ring::ring::Ring;
    use crate::subroutines::crs::CRS;
    use super::*;
    #[test]
    fn test_decomp() {
        let radix = 4;
        let ck = CRS::gen_crs(8, 2).ck;
        let wit = sample_short_random_mat(8, 2);
        let (new_witness, output) = decomp(&ck, &wit, radix);
        let composed_witness = compose_with_radix(&new_witness, radix, output.parts);
        assert_eq!(composed_witness, wit);


        let rhs = parallel_dot_series_matrix(&ck, &wit);

        let composed_rhs = compose_with_radix_mod(&output.rhs, radix, output.parts);

        assert_eq!(composed_rhs, rhs);
    }
}
