use std::time::Instant;
use crate::arithmetic::{parallel_dot_matrix_matrix, parallel_dot_series_matrix, reshape, sample_random_vector, sample_random_vector_subring, sample_short_random_mat, transpose};
use crate::helpers::println_with_timestamp;
use crate::custom_ring::r#static::{CHUNKS, CHUNK_SIZE, COMMITMENT_MODULE_SIZE, LOG_Q, MODULE_SIZE, NOF_ROUNDS, RADICES, RADIX, REP, SHOULD_REPEAT_SP_IN_LAST, TIME, WIT_DIM};
use crate::custom_ring::ring::{DPrimeRingElement};
use crate::subroutines::crs::CRS;
use crate::subroutines::decomp::decomp;
use crate::subroutines::fold::fold;
use crate::subroutines::norm_first_round::norm_1;
use crate::subroutines::norm_second_round::norm_2;
use crate::subroutines::split::split;
use crate::subroutines::verifier::{VerifierState, challenge_for_fold, norm_challenge, verifier_fold, verifier_split, verify_decomp, verify_norm_2, verifier_squeeze};
use crate::vdf::execute_vdf;

pub fn protocol() {
    let crs = CRS::gen_crs(CHUNK_SIZE * MODULE_SIZE * LOG_Q, COMMITMENT_MODULE_SIZE);
    let y_a = sample_random_vector_subring(MODULE_SIZE);

    let now = Instant::now();
    let output = execute_vdf(&y_a, &crs.a, CHUNKS, TIME);
    let output_witness = output.witness.clone(); // TODO!!
    // let output_witness = sample_random_bin_vec(MODULE_SIZE * LOG_Q * TIME);
    let vdf_elapsed = now.elapsed();
    let mut verifier_runtime = Instant::now().elapsed();
    let mut prover_runtime = Instant::now().elapsed();
    println_with_timestamp!("Time for execute_vdf: {:.2?}", vdf_elapsed);

    let now = Instant::now();
    let mut witness = transpose(&reshape(&output_witness, CHUNK_SIZE * MODULE_SIZE * LOG_Q));
    let elapsed = now.elapsed();
    println_with_timestamp!("Time for transpose & reshape: {:.2?}", elapsed);
    prover_runtime = prover_runtime + elapsed;
    let now = Instant::now();
    let (result, r)= verifier_squeeze(&crs, &output, y_a, CHUNK_SIZE);

    let elapsed = now.elapsed();
    println_with_timestamp!("Time for VDF squeeze challenge: {:.2?}", elapsed);
    prover_runtime = prover_runtime + elapsed;

    let now = Instant::now();
    let mut commitment = parallel_dot_series_matrix(&crs.ck, &witness);
    let elapsed = now.elapsed();

    let mut verifier_state = VerifierState {
        wit_cols: CHUNKS,
        wit_rows: CHUNK_SIZE * MODULE_SIZE * LOG_Q,
        rhs: vec![vec![r], commitment].concat(),
    };

    println_with_timestamp!("Time for parallel_dot_matrix_matrix (commitment): {:.2?}", elapsed);
    prover_runtime = prover_runtime + elapsed;


    let mut statement = vec![vec![result], crs.ck].concat();

    for i in 0..NOF_ROUNDS {
        // if SKIP_OPENER && i == 1 {
        //     println_with_timestamp!("skipping opener...");
        //     continue;
        // }

        // cfg_if::cfg_if! {
        //     if #[cfg(feature = "a0")] {} else {
        //        if SKIP_OPENER && i == 4 {
        //            println_with_timestamp!("skipping opener...");
        //            continue;
        //        }
        //     }
        // }
        if RADICES[i as usize] != 0 {
            let now = Instant::now();
            let radix = RADICES[i as usize];
            let (new_witness, bdecomp_output) = decomp(&statement, &witness, radix);
            witness = new_witness;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for b_decomp: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;


            let now = Instant::now();
            let new_verifier_state = verify_decomp(bdecomp_output, &verifier_state, radix);
            verifier_state = new_verifier_state;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for verify_bdecomp: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;

            // assert_eq!(parallel_dot_matrix_matrix(&statement, &witness), verifier_state.rhs);
        }


        let now = Instant::now();
        let (new_witness, norm_1_output) = norm_1(&statement, &witness);
        witness = new_witness;
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for norm_1: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;

        let now = Instant::now();
        let (new_verifier_state, challenge, inverse_challenge) = norm_challenge(&norm_1_output, &verifier_state);
        verifier_state = new_verifier_state;
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for norm_challenge: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;
        //
        let now = Instant::now();
        let (new_power_series, norm_2_output) = norm_2(&statement, &witness, &challenge, &inverse_challenge, i == 0);
        statement = new_power_series;
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for norm_2: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;

        let now = Instant::now();
        verifier_state = verify_norm_2(&norm_1_output, &norm_2_output, &verifier_state, i == 0);
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for verify_norm_2: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;


        for u in 0..2 {
            if u == 0 && i == NOF_ROUNDS -  1 && !SHOULD_REPEAT_SP_IN_LAST {
                continue;
            }
            let now = Instant::now();
            let (new_witness, split_output) = split(&mut statement, &witness);
            witness = new_witness;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for split: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;


            let now = Instant::now();
            verifier_state = verifier_split(&statement, split_output, &verifier_state);
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for split verifier: {:.2?}", elapsed);

            let now = Instant::now();
            let challenge = challenge_for_fold(&verifier_state);
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for challenge fold: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;

            let now = Instant::now();
            let new_witness = fold(&witness, &challenge);
            witness = new_witness;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for fold: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;

            let now = Instant::now();
            verifier_state = verifier_fold(&verifier_state, &challenge);
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for fold verifier: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;
        }
    }



    let now = Instant::now();
    assert_eq!(parallel_dot_series_matrix(&statement, &witness), verifier_state.rhs);
    let elapsed = now.elapsed();
    println_with_timestamp!("Time for final assert_eq: {:.2?}", elapsed);
    verifier_runtime = verifier_runtime + elapsed;


    println_with_timestamp!("VDF: {:.2?}", vdf_elapsed);
    println_with_timestamp!("PRV: {:.2?}", prover_runtime);
    println_with_timestamp!("VER: {:.2?}", verifier_runtime);

}
