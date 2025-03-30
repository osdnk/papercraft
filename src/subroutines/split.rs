use crate::arithmetic::{add_matrices, first_n_columns, last_n_columns, map_matrix_to_prime_ring, map_vector_to_prime_ring, parallel_dot_matrix_matrix, parallel_dot_series_matrix, PowerSeries, row_wise_tensor, transpose};
use crate::helpers::println_with_timestamp;
use crate::custom_ring::r#static::{LOG_Q, MODULE_SIZE};
use crate::custom_ring::ring::{DPrimeRingElement, Ring, RingElement};
use crate::subroutines::crs::CRS;

/// Splits the provided vector into three parts: L (left), C (center), and R (right).
///
/// # Arguments
///
/// * `vec` - The vector to be split.
/// * `chunk_size` - The size of each chunk.
///
/// # Returns
///
/// A tuple containing three vectors: `vec_L`, `vec_C`, and `vec_R`.
fn split_vec(vec: &Vec<DPrimeRingElement>, chunk_size: usize) -> (Vec<DPrimeRingElement>, Vec<DPrimeRingElement>, Vec<DPrimeRingElement>) {
    let n = vec.len();
    let len_C = if (n / chunk_size) % 2 == 0 { 0 * chunk_size } else { chunk_size };
    let len_L_R_adjusted = (n - len_C) / 2;

    let vec_L = vec[0..len_L_R_adjusted].to_vec();
    let vec_C = vec[len_L_R_adjusted..len_L_R_adjusted + len_C].to_vec();
    let vec_R = vec[len_L_R_adjusted + len_C..].to_vec();

    (vec_L, vec_C, vec_R)
}

/// The output of the split operation, containing the new RHS, the witness center.
pub struct SplitOutput {
    pub(crate) rhs: Vec<Vec<DPrimeRingElement>>,
    pub(crate) witness_center: Vec<Vec<DPrimeRingElement>>,
}

/// Splits the given power series and witness into components and computes the necessary matrices.
///
/// # Arguments
///
/// * `power_series` - The reference to the power series matrix.
/// * `witness` - The witness matrix to be split.
///
/// # Returns
///
/// A `SplitOutput` containing the new RHS, witness center, and new witness matrices.
pub fn split(power_series: &Vec<PowerSeries>, witness: &Vec<Vec<DPrimeRingElement>>) -> (Vec<Vec<DPrimeRingElement>>, SplitOutput) {
    let mut witness_split_transposed_l = Vec::new();
    let mut witness_split_transposed_r = Vec::new();
    let mut witness_center_transposed = Vec::new();


    // Transpose the witness matrix
    let witness_transposed = transpose(&witness);

    println_with_timestamp!(" Splitting {:?}", witness_transposed[0].len());

    // Split each column of the transposed witness matrix
    for witness in witness_transposed {
        let (l, c, r) = split_vec(&witness, 1);
        witness_split_transposed_l.push(l);
        witness_split_transposed_r.push(r);
        witness_center_transposed.push(c);
    }

    println_with_timestamp!(" into {:?} {:?} {:?}", witness_split_transposed_l[0].len(), witness_center_transposed[0].len(), witness_split_transposed_r[0].len());

    if witness_center_transposed[0].len() != 0 {
        println_with_timestamp!("   SPLIT NOT OPTIMAL");
    }


    // Concatenate left and right splits
    let witness_split_transposed = [witness_split_transposed_l, witness_split_transposed_r].concat();

    // Compute new witness length and transpose
    let new_witness_len = witness_split_transposed[0].len();
    let witness_split = transpose(&witness_split_transposed);

    // Compute the new RHS
    let new_rhs = parallel_dot_series_matrix(
        &power_series,
        &witness_split
    );

    (witness_split, SplitOutput {
        rhs: new_rhs,
        witness_center: transpose(&witness_center_transposed),
    })
}

#[test]
// fn test_split_vec() {
//     let series = PowerSeries {
//         coeffs: map_vector_to_prime_ring(vec![1, 2, 4, 8, 16, 32, 64, 128]),
//         factor: PrimeRing::constant(2),
//         chunks: 8,
//         expanded_layers: vec![],
//         tensors: vec![],
//     };
//
//     let expected_L = map_vector_to_prime_ring(vec![1, 2, 4, 8]);
//     let expected_C: Vec<DPrimeRingElement> = Vec::new();
//     let expected_R = map_vector_to_prime_ring(vec![16, 32, 64, 128]);
//
//     let (vec_L, vec_C, vec_R) = split_power_series(&series);
//
//     assert_eq!(vec_L, expected_L);
//     assert_eq!(vec_C, expected_C);
//     assert_eq!(vec_R, expected_R);
// }

#[test]
fn test_split() {
    let mut series = vec![PowerSeries {
        expanded_layers: map_matrix_to_prime_ring(vec![
            vec![1, 2, 4, 8, 16, 32, 64, 128],
            vec![1, 2, 4, 8],
            vec![1, 2],
            vec![1],
        ]),
        tensors: map_matrix_to_prime_ring(vec![
            vec![1, 16],
            vec![1, 4],
            vec![1, 2],
        ]),
    }];

    let witness = transpose(&vec![map_vector_to_prime_ring(vec![1, 2, 3, 4, 5, 6, 7, 8])]);

    let rhs = vec![map_vector_to_prime_ring(vec![1793])];

    assert_eq!(parallel_dot_series_matrix(
        &series,
        &witness,
    ), rhs);

    let (witness_split, split_output) = split(&mut series, &witness);

    let new_rhs = vec![map_vector_to_prime_ring(vec![49, 109])];

    assert_eq!(split_output.rhs, new_rhs);

    let new_witness = transpose(&vec![
        map_vector_to_prime_ring(vec![1, 2, 3, 4]),
        map_vector_to_prime_ring(vec![5, 6, 7, 8])
    ]);

    assert_eq!(witness_split, new_witness);
}

#[test]
fn test_split_2() {
    let mut series = CRS::gen_crs(2, 1).ck;

    let multiplier = series[0].expanded_layers[0][0];
    let witness = transpose(&vec![map_vector_to_prime_ring(vec![1, 1])]);

    let rhs = parallel_dot_series_matrix(
        &series,
        &witness);

    let (witness_split, split_output) = split(&mut series, &witness);

    let rhs_l = first_n_columns(&split_output.rhs, 1);
    let rhs_r = last_n_columns(&split_output.rhs, 1);

    let rhs_r_multiplied = row_wise_tensor(&rhs_r, &transpose(&vec![vec![multiplier]]));

    println!("{:?} {:?}", rhs_l, rhs_r_multiplied);
    assert_eq!(add_matrices(&rhs_r_multiplied, &rhs_l), rhs);

}

