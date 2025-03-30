use std::time::Instant;

use crate::arithmetic::{parallel_dot_matrix_matrix, reshape, sample_random_vector};
use crate::crs::gen_crs;
use crate::helpers::println_with_timestamp;
use crate::r#static::{CHUNK_SIZE, CHUNKS, LOG_Q, MODULE_SIZE, RADIX, SKIP_OPENER, TIME};
use crate::ring_helpers::transpose;
use crate::ring_i128::{Ring, RingElement};
use crate::subroutines::bdecomp::b_decomp;
use crate::subroutines::fold::fold;
use crate::subroutines::norm_first_round::norm_1;
use crate::subroutines::norm_second_round::norm_2;
use crate::subroutines::split::split;
use crate::subroutines::verifier::{challenge_for_fold, norm_challenge, verifier_fold, verifier_squeeze, verify_bdecomp, verify_norm_2, verify_split};
use crate::vdf::execute_vdf;

pub(crate) struct VerifierState {
    pub(crate) wit_cols: usize,
    pub(crate) wit_rows: usize,
    pub(crate) commitment: Vec<Vec<RingElement>>
}


pub fn vdf_flatten_challenge() -> RingElement {
     Ring::random()
}




pub fn protocol() {
    let crs = gen_crs();
    let y_a = sample_random_vector(MODULE_SIZE);

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
    let (result, r, power) = verifier_squeeze(&crs, &output, y_a, CHUNK_SIZE);
    let elapsed = now.elapsed();
    println_with_timestamp!("Time for VDF squeeze challenge: {:.2?}", elapsed);
    verifier_runtime = verifier_runtime + elapsed;

    let now = Instant::now();
    let mut commitment = parallel_dot_matrix_matrix(&crs.ck, &witness);
    let elapsed = now.elapsed();

    println_with_timestamp!("Time for parallel_dot_matrix_matrix (commitment): {:.2?}", elapsed);
    prover_runtime = prover_runtime + elapsed;

    let mut verifier_state = VerifierState {
        wit_cols: CHUNKS,
        wit_rows: CHUNK_SIZE * MODULE_SIZE * LOG_Q,
        commitment: vec![vec![r], commitment].concat(),
    };

    let mut statement = vec![vec![result], crs.ck].concat();

   for i in 0..7 {
       if SKIP_OPENER && i == 1 {
           println_with_timestamp!("skipping opener...");
           continue;
       }

       cfg_if::cfg_if! {
            if #[cfg(feature = "a0")] {} else {
               if SKIP_OPENER && i == 4 {
                   println_with_timestamp!("skipping opener...");
                   continue;
               }
            }
        }
        if i == 3 || i == 6 {
            let now = Instant::now();
            let radix = RADIX;
            let (new_witness, bdecomp_output) = b_decomp(&statement, &witness, radix);
            witness = new_witness;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for b_decomp: {:.2?}", elapsed);
            prover_runtime = prover_runtime + elapsed;


            let now = Instant::now();
            let new_verifier_state = verify_bdecomp(&bdecomp_output, &statement, &verifier_state, radix);
            verifier_state = new_verifier_state;
            let elapsed = now.elapsed();
            println_with_timestamp!("Time for verify_bdecomp: {:.2?}", elapsed);
            verifier_runtime = verifier_runtime + elapsed;

            assert_eq!(parallel_dot_matrix_matrix(&statement, &witness), verifier_state.commitment);
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

        // assert_eq!(parallel_dot_matrix_matrix(&statement, &witness), verifier_state.commitment);
        let now = Instant::now();
        let (new_witness, new_power_series, split_output) = split(&statement, &witness);
        witness = new_witness;
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for split: {:.2?}", elapsed);
        prover_runtime = prover_runtime + elapsed;

        let now = Instant::now();
        verifier_state = verify_split(&statement, &split_output, &verifier_state, &power);
        let elapsed = now.elapsed();
        println_with_timestamp!("Time for verify_split: {:.2?}", elapsed);
        verifier_runtime = verifier_runtime + elapsed;

        // we need to keep the old statement for the verifier.
        statement = new_power_series;

    }



    let now = Instant::now();
    assert_eq!(parallel_dot_matrix_matrix(&statement, &witness), verifier_state.commitment);
    let elapsed = now.elapsed();
    println_with_timestamp!("Time for final assert_eq: {:.2?}", elapsed);
    verifier_runtime = verifier_runtime + elapsed;


    println_with_timestamp!("VDF: {:.2?}", vdf_elapsed);
    println_with_timestamp!("PRV: {:.2?}", prover_runtime);
    println_with_timestamp!("VER: {:.2?}", verifier_runtime);

}
