use num_traits::One;
use rayon::prelude::*;

use crate::arithmetic::{add_matrices, columns, fast_power, first_n_columns, last_n_columns, parallel_dot_matrix_matrix, row_wise_tensor, sample_random_mat};
use crate::crs::gen_crs;
use crate::helpers::println_with_timestamp;
use crate::r#static::{CHUNK_SIZE, LOG_Q, MODULE_SIZE};
use crate::ring_helpers::transpose;
use crate::ring_i128::{Ring, RingElement};

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
fn split_vec(vec: &Vec<RingElement>, chunk_size: usize) -> (Vec<RingElement>, Vec<RingElement>, Vec<RingElement>) {
    let n = vec.len();
    let len_C = if (n / chunk_size) % 2 == 0 { 0 * chunk_size } else { chunk_size };
    let len_L_R_adjusted = (n - len_C) / 2;

    let vec_L = vec[0..len_L_R_adjusted].to_vec();
    let vec_C = vec[len_L_R_adjusted..len_L_R_adjusted + len_C].to_vec();
    let vec_R = vec[len_L_R_adjusted + len_C..].to_vec();

    (vec_L, vec_C, vec_R)
}

/// The output of the split operation, containing the new RHS, the witness center, and the new witness.
pub struct SplitOutput {
    pub(crate) rhs: Vec<Vec<RingElement>>,
    pub(crate) witness_center: Vec<Vec<RingElement>>,
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
pub fn split(power_series: &Vec<Vec<RingElement>>, witness: &Vec<Vec<RingElement>>) -> (Vec<Vec<RingElement>>, Vec<Vec<RingElement>>, SplitOutput) {
    let mut witness_split_transposed_l = Vec::new();
    let mut witness_split_transposed_r = Vec::new();
    let mut witness_center_transposed = Vec::new();


    // Transpose the witness matrix
    let witness_transposed = transpose(&witness);

    println_with_timestamp!(" Splitting {:?} chunks", witness_transposed[0].len() / (LOG_Q * MODULE_SIZE));
    println_with_timestamp!(" Precisely {:?} ", witness_transposed[0].len());

    // Split each column of the transposed witness matrix
    for witness in witness_transposed {
        let (l, c, r) = split_vec(&witness, LOG_Q * MODULE_SIZE);
        witness_split_transposed_l.push(l);
        witness_split_transposed_r.push(r);
        witness_center_transposed.push(c);
    }

    println_with_timestamp!(" into {:?} {:?} {:?} chunks", witness_split_transposed_l[0].len() / (LOG_Q * MODULE_SIZE), witness_center_transposed[0].len() / (LOG_Q * MODULE_SIZE), witness_split_transposed_r[0].len() / (LOG_Q * MODULE_SIZE));
    println_with_timestamp!(" precisely {:?} {:?} {:?} chunks", witness_split_transposed_l[0].len(), witness_center_transposed[0].len(), witness_split_transposed_r[0].len());

    if witness_center_transposed[0].len() != 0 {
        println_with_timestamp!("   SPLIT NOT OPTIMAL");
    }


    // Concatenate left and right splits
    let witness_split_transposed = [witness_split_transposed_l, witness_split_transposed_r].concat();

    // Compute new witness length and transpose
    let new_witness_len = witness_split_transposed[0].len();
    let witness_split = transpose(&witness_split_transposed);

    // Extract the relevant columns from the power series
    let power_series_sub = first_n_columns(&power_series, new_witness_len);

    // Compute the new RHS
    let new_rhs = parallel_dot_matrix_matrix(&power_series_sub, &witness_split);

    (witness_split, power_series_sub, SplitOutput {
        rhs: new_rhs,
        witness_center: transpose(&witness_center_transposed),
    })
}

// we support many types of power series
pub fn get_power_series_multiplier(power_series: &Vec<Vec<RingElement>>, len_L_R_adjusted: usize, len_C: usize, first_multiplier: &RingElement) -> Vec<RingElement> {
    let power_series_sub = columns(&power_series, len_L_R_adjusted + len_C - 1, len_L_R_adjusted + len_C);
    let multiplier1 = transpose(&power_series_sub).first().unwrap().clone();
    let multiplier2 = transpose(&power_series_sub).last().unwrap().clone();
    let power = fast_power(first_multiplier.clone(), ((len_L_R_adjusted + len_C) / (LOG_Q * MODULE_SIZE)) as u32);

    let mut is_first = true;
    multiplier1.iter()
        .zip(multiplier2.iter())
        .zip(power_series)
        .map(|((&m1, &m2), series)| if is_first {
            is_first = false;
            power // TODO
        } else if series[0].is_one() {
            m2
        } else {
            if series[0] == series[1] {
                Ring::constant(1)
            } else {
                m1
            }
        } )
        .collect()
}
#[test]
fn test_split() {
    // Generate the CRS and sample a random witness
    let ck = gen_crs().ck;
    let another_power_series = vec![Ring::constant(1); ck[0].len()];
    let mut yet_another_power_series = vec![Ring::constant(0); ck[0].len()];
    yet_another_power_series[0] = Ring::constant(1);
    let one_more_power_series = vec![vec![Ring::constant(1)], ck[0][0..ck[0].len() - 1].to_vec()].concat();

    let yet_yet_another_power_series = vec![Ring::all(1); ck[0].len()];
    let zero_series = vec![Ring::zero(); ck[0].len()];
    assert_eq!(yet_yet_another_power_series[0], yet_yet_another_power_series[1]);

    let power_series = vec![vec![zero_series], ck, vec![another_power_series, yet_another_power_series, one_more_power_series, yet_yet_another_power_series]].concat();
    let witness = sample_random_mat(power_series[0].len(), 2);
    // Compute the commitment
    let commitment = parallel_dot_matrix_matrix(&power_series, &witness);

    // Split the witness matrix
    let (_, _, output) = split(&power_series, &witness);

    // Length calculations for splitting
    let chunk_size = LOG_Q * MODULE_SIZE;
    let n = power_series[0].len();
    let len_C = if n % 2 == 0 { 0 * chunk_size } else { chunk_size };
    let len_L_R_adjusted = (n - len_C) / 2;

    // Extract the power series submatrix

    // Extract left and right columns from RHS
    let ck_l = first_n_columns(&output.rhs, 2);
    let ck_r = last_n_columns(&output.rhs, 2);

    println_with_timestamp!("{:?} {:?}", ck_l.len(), ck_l[0].len());

    // Compute the center commitment
    let power_series_sub = columns(&power_series, len_L_R_adjusted, len_L_R_adjusted + len_C - 1);
    // let ck_c = parallel_dot_matrix_matrix(&power_series_sub, &output.witness_center);

    let multiplier = get_power_series_multiplier(&power_series, len_L_R_adjusted, len_C, &Ring::zero());

    let ck_r_multiplied = row_wise_tensor(&ck_r, &transpose(&vec![multiplier]));

    assert_eq!(add_matrices(&ck_r_multiplied, &ck_l), commitment);
}


