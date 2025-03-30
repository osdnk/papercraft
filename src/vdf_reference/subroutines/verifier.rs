use crate::arithmetic::{add_matrices, add_vectors, columns, first_n_columns, join_matrices_horizontally, last_n_columns, multiply_matrix_constant_in_place, parallel_dot_matrix_matrix, parallel_dot_matrix_vector, row_wise_tensor, sample_random_bin_mat, sample_random_constant_bin_mat, split_into_submatrices_by_columns};
use crate::r#static::{BASE_INT, CHUNK_SIZE, CHUNKS, LOG_Q, MODULE_SIZE};
use crate::ring_helpers::transpose;
use crate::ring_i128::{get_g_custom, Ring, ring_inner_product, RingElement};
use crate::subroutines::bdecomp::DecompOutput;
use crate::subroutines::norm_first_round::Norm1Output;
use crate::subroutines::norm_second_round::Norm2Output;
use crate::subroutines::split::{get_power_series_multiplier, SplitOutput};
use crate::protocol::*;
use std::time::Instant;
use crate::crs::CRS;
use crate::helpers::println_with_timestamp;
use crate::vdf::{flat_vdf, VDFOutputMat};

pub fn norm_challenge(norm_1_output: &Norm1Output, verifier_state: &VerifierState) -> (VerifierState, RingElement, RingElement) {
    let challenge = Ring::random();
    let inverse_challenge = challenge.inverse().conjugate();
    let state = VerifierState {
        wit_cols: verifier_state.wit_cols * 3,
        wit_rows: verifier_state.wit_rows,
        commitment: join_matrices_horizontally(&verifier_state.commitment, &norm_1_output.new_rhs.clone())
    };
    (state, challenge, inverse_challenge)
}

pub fn verify_norm_2(norm_1_output: &Norm1Output, norm_2_output: &Norm2Output, state: &VerifierState, exact_binariness: bool) -> VerifierState {
    let new_evaluations = last_n_columns(&norm_2_output.new_rhs, state.wit_cols / 3 * 2);
    let mut new_evaluations_per_witness_column = split_into_submatrices_by_columns(&new_evaluations, new_evaluations[0].len() / 2);
    multiply_matrix_constant_in_place(&mut new_evaluations_per_witness_column[0], &Ring::constant(norm_1_output.radix));
    let new_evaluations_comped =
        add_matrices(&new_evaluations_per_witness_column[0], &new_evaluations_per_witness_column[1]);

    for i in 0..state.wit_cols / 3 {
        assert_eq!(
            new_evaluations_comped[0][i] + new_evaluations_comped[1][i].conjugate() - new_evaluations_comped[2][i],
            norm_2_output.new_rhs[0][i] * norm_2_output.new_rhs[1][i].conjugate()
        );
        if exact_binariness {
            assert_eq!(
                (norm_2_output.new_rhs[3][i] - new_evaluations_comped[2][i]).twisted_trace(),
                0
            );
        }
    }

    println_with_timestamp!("{:?} {:?} {:?} {:?}", state.commitment.len(),state.commitment[0].len(), norm_2_output.new_rhs.len(), norm_2_output.new_rhs[0].len());
    VerifierState {
        wit_cols: state.wit_cols,
        wit_rows: state.wit_rows,
        commitment: vec![state.commitment.clone(), norm_2_output.new_rhs.clone()].concat()
    }

}


pub fn verify_split(
    power_series: &Vec<Vec<RingElement>>,
    output_split: &SplitOutput,
    verifier_state: &VerifierState,
    power: &RingElement
) -> VerifierState {

    let now = Instant::now();
    let chunk_size = LOG_Q * MODULE_SIZE;
    let n = verifier_state.wit_rows;
    let len_C = if (n / chunk_size) % 2 == 0 { 0 * chunk_size } else { chunk_size };
    let len_L_R_adjusted = (n - len_C) / 2;
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute initial parameters: {:.2?}", elapsed);

    // Extract left and right columns from RHS
    let now = Instant::now();
    let ck_l = first_n_columns(&output_split.rhs, verifier_state.wit_cols);
    let ck_r = last_n_columns(&output_split.rhs, verifier_state.wit_cols);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to extract left and right columns from RHS: {:.2?}", elapsed);

    // Compute the center commitment
    let now = Instant::now();
    let ck_c = if len_C == 0 {
        Vec::new()
    } else {
        let power_series_sub = columns(&power_series, len_L_R_adjusted, len_L_R_adjusted + len_C - 1);
        parallel_dot_matrix_matrix(&power_series_sub, &output_split.witness_center)
    };
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute the center commitment: {:.2?}", elapsed);

    // Compute the multiplier and row-wise tensor product
    let now = Instant::now();
    let multiplier = get_power_series_multiplier(&power_series, len_L_R_adjusted, len_C, power);
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to compute the multiplier: {:.2?}", elapsed);

    let now = Instant::now();
    let ck_r_multiplied = row_wise_tensor(&ck_r, &transpose(&vec![multiplier]));
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time for row-wise tensor product: {:.2?}", elapsed);

    // Combine results and verify the commitment
    let now = Instant::now();
    let ck_lr = add_matrices(&ck_r_multiplied, &ck_l);
    if len_C == 0 {
        assert_eq!(ck_lr, verifier_state.commitment);
    } else {
        assert_eq!(add_matrices(&ck_lr, &ck_c), verifier_state.commitment);
    }
    let elapsed = now.elapsed();
    println_with_timestamp!("  Time to combine results and verify the commitment: {:.2?}", elapsed);

    VerifierState {
        wit_cols: verifier_state.wit_cols * 2,
        wit_rows: len_L_R_adjusted,
        commitment: join_matrices_horizontally(&ck_l, &ck_r)
    }
}

pub fn verify_bdecomp(bdecomp_output: &DecompOutput, power_series: &Vec<Vec<RingElement>>, verifier_state: &VerifierState, radix: BASE_INT) -> VerifierState {
    let decomposed = split_into_submatrices_by_columns(&bdecomp_output.rhs, verifier_state.wit_cols);

    let g = get_g_custom(bdecomp_output.parts, radix);


    for i in 0..power_series.len() {
        for j in 0..verifier_state.wit_cols {
            let entries = (0..bdecomp_output.parts).map(|k| {
                decomposed[k][i][j]
            }).collect();
            assert_eq!(ring_inner_product(&entries, &g), verifier_state.commitment[i][j]);
        }
    }

    VerifierState {
        wit_rows: verifier_state.wit_rows,
        wit_cols: verifier_state.wit_cols * bdecomp_output.parts,
        commitment: bdecomp_output.rhs.clone(), //TODO
    }

}

pub fn challenge_for_fold(verifier_state: &VerifierState) -> Vec<Vec<RingElement>> {
    sample_random_constant_bin_mat(verifier_state.wit_cols, CHUNKS) // TODO
}

pub fn verifier_fold(verifier_state: &VerifierState, challenge: &Vec<Vec<RingElement>>) -> VerifierState {
    VerifierState {
        wit_cols: CHUNKS,
        wit_rows: verifier_state.wit_rows,
        commitment: parallel_dot_matrix_matrix(&verifier_state.commitment, &challenge)
    }
}

pub fn verifier_squeeze(crs: &CRS, output: &VDFOutputMat, y_a: Vec<RingElement>, chunk_size: usize) -> (Vec<RingElement>, Vec<RingElement>, RingElement) {
    let challenge = vdf_flatten_challenge();
    let negative_images: Vec<Vec<RingElement>> = output.intermediate_images.iter().map(|image| { image.iter().map(RingElement::minus).collect::<Vec<_>>() }).collect();
    let top_row = transpose(&vec![vec![y_a], output.intermediate_images.clone()].concat());
    let bottom_row = transpose(&vec![negative_images, vec![output.output_image.clone()]].concat());
    let  (result, squeeze_vector_0, squeeze_vector) = flat_vdf(&challenge, &crs.a, chunk_size);
    let r1 = parallel_dot_matrix_vector(&transpose(&top_row), &squeeze_vector_0);
    let r2 = parallel_dot_matrix_vector(&transpose(&bottom_row), &squeeze_vector);
    let r = add_vectors(&r1, &r2);
    let power = squeeze_vector_0.last().unwrap().clone();

    (result, r, power)
}
