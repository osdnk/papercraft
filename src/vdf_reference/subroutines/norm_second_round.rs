use rayon::prelude::*;
use crate::arithmetic::{
    add_matrices, compute_one_prefixed_power_series, compute_one_prefixed_zero_series,
    conjugate_vector, first_n_columns, last_n_columns, multiply_matrix_constant_in_place,
    parallel_dot_matrix_matrix, sample_random_bin_mat, sample_random_mat,
    split_into_submatrices_by_columns
};
use crate::crs::gen_crs;
use crate::helpers::println_with_timestamp;
use crate::subroutines::convolution::convolution;
use crate::subroutines::norm_first_round::norm_1;
use crate::ring_helpers::transpose;
use crate::ring_i128::{Ring, ring_inner_product, RingElement};

/// Struct to store the output of the `norm_2` function.
pub struct Norm2Output {
    pub(crate) new_rhs: Vec<Vec<RingElement>>,
}

/// Computes the second round normalization in a zero-knowledge proof system.
///
/// # Arguments
///
/// * `power_series` - A reference to a vector of vectors containing `RingElement` power series.
/// * `witness` - A reference to a vector of vectors containing `RingElement` witness matrix.
/// * `challenges` - A reference to a `RingElement` containing the challenges.
/// * `inverse_challenge` - A reference to a `RingElement` containing the inverse challenge.
/// * `exact_binariness` - A boolean indicating whether exact binariness is enforced.
///
/// # Returns
///
/// * `Norm2Output` - Contains the `new_rhs` matrix computed in this round.
pub fn norm_2(
    power_series: &Vec<Vec<RingElement>>,
    witness: &Vec<Vec<RingElement>>,
    challenges: &RingElement,
    inverse_challenge: &RingElement,
    exact_binariness: bool
) -> (Vec<Vec<RingElement>>, Norm2Output) {
    let challenge_power_series = compute_one_prefixed_power_series(challenges, power_series[0].len());
    let challenge_power_series_conjugate = compute_one_prefixed_power_series(inverse_challenge, power_series[0].len());
    let challenge_one_zero_series = compute_one_prefixed_zero_series(power_series[0].len());
    let challenge_one_series = vec![Ring::all(1).conjugate(); power_series[0].len()];

    let mut new_power_series = vec![
        challenge_power_series,
        challenge_power_series_conjugate,
        challenge_one_zero_series,
    ];

    if exact_binariness {
        new_power_series.push(challenge_one_series);
    }

    let nrows = new_power_series[0].len();
    let ncols = witness.len();
    println_with_timestamp!("{:?}, {:?}", nrows, ncols);

    let new_rhs = parallel_dot_matrix_matrix(&new_power_series, witness);
    (vec![power_series.clone(), new_power_series].concat(), Norm2Output { new_rhs }) //TODO
}

// Unit tests

#[cfg(test)]
mod tests_norm_2 {
    use super::*;
    use crate::arithmetic::{sample_random_mat, sample_random_bin_mat};
    use crate::crs::gen_crs;
    use crate::subroutines::norm_first_round::norm_1;

    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_norm_2() {
        let ck = first_n_columns(&gen_crs().ck, 10);
        let witness = sample_random_mat(11, 20);
        let challenge = Ring::random();
        let inverse_challenge = challenge.inverse().conjugate();
        norm_2(&ck, &witness, &challenge, &inverse_challenge, false);
    }

    #[test]
    fn test_norm_2_inner_product() {
        let nrows = 10;
        let ncols = 2;
        let ck = first_n_columns(&gen_crs().ck, nrows);
        let witness = sample_random_mat(nrows, ncols);
        let (new_witness, norm_output) = norm_1(&ck, &witness);
        let new_witness = new_witness;
        let g = vec![Ring::constant(norm_output.radix), Ring::constant(1)];
        let vec_witness = transpose(&witness);
        let convoluted = convolution(&vec_witness[0]);
        let t1 = RingElement {
            coeffs: convoluted[witness.len() - 1].clone().try_into().unwrap(),
        };

        let ip = ring_inner_product(&vec_witness[0], &conjugate_vector(&vec_witness[0]));
        assert_eq!(ip, t1);
        let challenge = Ring::random();
        let inverse_challenge = challenge.inverse().conjugate();
        let (_, output_2) = norm_2(&ck, &new_witness, &challenge, &inverse_challenge, false);
        let new_evaluations = last_n_columns(&output_2.new_rhs, ncols * 2);
        let mut new_evaluations_per_witness_column = split_into_submatrices_by_columns(&new_evaluations, new_evaluations[0].len() / 2);
        multiply_matrix_constant_in_place(&mut new_evaluations_per_witness_column[0], &g[0]);
        let new_evaluations_comped =
            add_matrices(&new_evaluations_per_witness_column[0], &new_evaluations_per_witness_column[1]);

        for i in 0..ncols {
            let ip = ring_inner_product(&vec_witness[i], &conjugate_vector(&vec_witness[i]));
            assert_eq!(ip, new_evaluations_comped[2][i]);
            assert_eq!(
                new_evaluations_comped[0][i] + new_evaluations_comped[1][i].conjugate() - ip,
                output_2.new_rhs[0][i] * output_2.new_rhs[1][i].conjugate()
            );
        }
    }

    #[test]
    fn test_norm_2_inner_product_binary() {
        let nrows = 10;
        let ncols = 5;
        let ck = first_n_columns(&gen_crs().ck, nrows);
        let witness = sample_random_bin_mat(nrows, ncols);
        let (new_witness, norm_output) = norm_1(&ck, &witness);
        let g = vec![Ring::constant(norm_output.radix), Ring::constant(1)];
        let vec_witness = transpose(&witness);
        let challenge = Ring::random();
        let inverse_challenge = challenge.inverse().conjugate();
        let (_, output_2) = norm_2(&ck, &new_witness, &challenge, &inverse_challenge, true);
        let new_evaluations = last_n_columns(&output_2.new_rhs, ncols * 2);
        let mut new_evaluations_per_witness_column = split_into_submatrices_by_columns(&new_evaluations, new_evaluations[0].len() / 2);
        multiply_matrix_constant_in_place(&mut new_evaluations_per_witness_column[0], &g[0]);
        let new_evaluations_comped =
            add_matrices(&new_evaluations_per_witness_column[0], &new_evaluations_per_witness_column[1]);

        for i in 0..ncols {
            assert_eq!(
                new_evaluations_comped[0][i] + new_evaluations_comped[1][i].conjugate() - new_evaluations_comped[2][i],
                output_2.new_rhs[0][i] * output_2.new_rhs[1][i].conjugate()
            );
            assert_eq!(
                (output_2.new_rhs[3][i] - new_evaluations_comped[2][i]).twisted_trace(),
                0
            );
        }
    }
}
