use num_traits::One;
use rayon::prelude::*;
use crate::arithmetic::{add_matrices, compute_one_prefixed_power_series, first_n_columns, last_n_columns, parallel_dot_matrix_matrix, parallel_dot_series_matrix, PowerSeries, sample_random_mat, compute_one_series};
use crate::helpers::println_with_timestamp;
use crate::custom_ring::ring::{DPrimeRingElement, Ring, RingElement};
use crate::subroutines::norm_first_round::norm_1;

/// Struct to store the output of the `norm_2` function.
pub struct Norm2Output {
    pub(crate) new_rhs: Vec<Vec<DPrimeRingElement>>,
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
    power_series: &Vec<PowerSeries>, // TODO mut instead of cloning!!
    witness: &Vec<Vec<DPrimeRingElement>>,
    challenges: &DPrimeRingElement,
    inverse_challenge: &DPrimeRingElement,
    exact_binariness: bool
) -> (Vec<PowerSeries>, Norm2Output) {
    let challenge_power_series = compute_one_prefixed_power_series(challenges, witness.len());
    let challenge_power_series_conjugate = compute_one_prefixed_power_series(inverse_challenge, witness.len());
    let challenge_one_zero_series = compute_one_prefixed_power_series(&Ring::constant(0), witness.len());
    // let mut challenge_one_series =  PowerSeries {
    //     expanded_layers: vec![],
    //     tensors: vec![],
    // };
    //
    // let challenge_one_series_row = vec![PrimeRing::all(1).conjugate(); witness.len()];
    //
    // let mut current_dim = witness.len();
    // while current_dim % 2 == 0 {
    //     challenge_one_series.expanded_layers.push(challenge_one_series_row[0..current_dim].to_vec());
    //     current_dim /= 2;
    //     challenge_one_series.tensors.push(vec![PrimeRingElement::one(), PrimeRingElement::one()]);
    // }
    // challenge_one_series.expanded_layers.push(challenge_one_series_row[0..current_dim].to_vec());

    let mut new_power_series = vec![
        challenge_power_series,
        challenge_power_series_conjugate,
        challenge_one_zero_series,
    ];

    if exact_binariness {
        let challenge_one_series = compute_one_series(witness.len());
        new_power_series.push(challenge_one_series);
    }

    let ncols = witness.len();
    println_with_timestamp!("{:?}", ncols);

    let new_rhs = parallel_dot_series_matrix(
        &new_power_series,
        witness
    );

    (vec![power_series.clone(), new_power_series].concat(), Norm2Output { new_rhs }) //TODO
}

// Unit tests

#[cfg(test)]
mod tests_norm_2 {
    use super::*;
    use crate::arithmetic::{sample_random_mat, transpose, ring_inner_product, conjugate_vector, compose_with_radix, compose_with_radix_mod, sample_bin_random_mat_subring, zero_mat};
    use crate::custom_ring::r#static::MOD_Q;
    use crate::slow_ntt::convolution::convolution;
    use crate::subroutines::crs::CRS;
    use crate::subroutines::norm_first_round::norm_1;

    #[test]
    fn test_norm_2_inner_product() {
        let a = Ring::random_subring_bin();
        // println!("WWWW {:?}", ((PrimeRing::all_subring(1) - a) * a.conjugate()).twisted_trace());

        let nrows = 10;
        let ncols = 2;
        let ck = CRS::gen_crs(10, 2).ck;
        let witness = sample_bin_random_mat_subring(nrows, ncols);
        let (new_witness, norm_output) = norm_1(&ck, &witness);
        let decomposed_witness = last_n_columns(&new_witness, 4);
        let convoluted_witness = compose_with_radix(&decomposed_witness, norm_output.radix, 2);

        // let g = vec![PrimeRing::constant(norm_output.radix), PrimeRing::constant(1)];
        let vec_witness = transpose(&witness);
        let convoluted = convolution(&vec_witness[0]);
        // println!("{:?}", vec_witness[0]);
        // println!("{:?}", convoluted);
        assert_eq!(convoluted, transpose(&convoluted_witness)[0]);
        let t1 = convoluted[0];

        let ip = ring_inner_product(&vec_witness[0], &conjugate_vector(&vec_witness[0]));
        assert_eq!(ip, t1);
        let challenge = Ring::random();
        let inverse_challenge = challenge.inverse().conjugate();
        assert_eq!(challenge * inverse_challenge.conjugate(), Ring::constant(1));

        let (_, output_2) = norm_2(&ck, &new_witness, &challenge, &inverse_challenge, true);
        let new_evaluations = compose_with_radix_mod(
            &last_n_columns(&output_2.new_rhs, 2 * ncols),
            norm_output.radix,
            2
        );

        for i in 0..ncols {
            let ip = ring_inner_product(&vec_witness[i], &conjugate_vector(&vec_witness[i]));
            assert_eq!(ip, new_evaluations[2][i]);
            assert_eq!(
                new_evaluations[0][i] + new_evaluations[1][i].conjugate(),
                output_2.new_rhs[0][i] * output_2.new_rhs[1][i].conjugate() + ip
            );
            assert_eq!(
                (output_2.new_rhs[3][i] - new_evaluations[2][i]).twisted_trace(),
                0
            );
        }
    }
}
