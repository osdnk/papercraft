use std::sync::Mutex;
use std::time::Instant;

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator};
use rayon::prelude::*;
use crate::arithmetic::{last_n_columns, sample_random_bin_mat, sample_random_mat};
use crate::helpers::println_with_timestamp;
use crate::poly_arithmetic_i128::{hadamard, hadamard_64, inverse_ntt_pow_of_2, inverse_ntt_pow_of_2_fast, ntt_pow_of_2, ntt_pow_of_2_fast, reduce_mod, reduce_mod_imbalanced, reduce_mod_imbalanced_u64, reduce_quotient};
use crate::r#static::{BASE_INT, CHUNK_SIZE, CHUNKS, LOG_Q, MOD_1, MODULE_SIZE, TIME};
use crate::ring_helpers::transpose;
use crate::ring_i128::{Ring, RingElement};
use crate::static_rings::static_generated::{MIN_POLY, PHI};

/// Computes the convolution of a vector `a` such that it maps `a` to `b`.
/// This ensures that poly(a) * negative_poly(conjugate(a)) = poly(b),
/// where poly^(-1) are the coefficients of the polynomial.
/// Uses double-CRT representation for computing the convolution.
///
/// # Arguments
///
/// * `a_ring_el` - The input vector of `RingElement`.
///
/// # Returns
///
/// A vector of vectors containing elements of type `BASE_INT`.
pub fn convolution(a_ring_el: &Vec<RingElement>) -> Vec<Vec<BASE_INT>> {
    // let now = Instant::now();
    let b_ring_el: Vec<RingElement> = a_ring_el.iter().rev().map(|w| w.clone().conjugate()).collect();
    let mut a_ring_el_copy = a_ring_el.clone();
    let mut b_ring_el_copy = b_ring_el.clone();

    let a: Vec<[BASE_INT; PHI]> = a_ring_el_copy
        .iter_mut()
        .map(|w| {
            reduce_mod_imbalanced(&mut w.coeffs, MOD_1);
            w.coeffs
        })
        .collect();
    let b: Vec<[BASE_INT; PHI]> = b_ring_el_copy
        .iter_mut()
        .map(|w| {
            reduce_mod_imbalanced(&mut w.coeffs, MOD_1);
            w.coeffs
        })
        .collect();
    // let elapsed = now.elapsed();
    // println_with_timestamp!("Time 1: {:.2?}", elapsed);
    let now = Instant::now();

    let n = a[0].len().next_power_of_two() * 2;
    let root_unity = crate::root_of_unity::choose_root_unity(n, MOD_1).unwrap();

    let mut extended_a: Vec<Vec<u64>> = a
        .par_iter()
        .map(|t| {
            let mut v = t.to_vec();
            let mut v_64: Vec<u64> = v.iter().map(|t| *t as u64).collect();
            v_64.resize(n, 0);
            ntt_pow_of_2_fast(&mut v_64, MOD_1 as u64, root_unity as u64);
            v_64
        })
        .collect();

    let mut extended_b: Vec<Vec<u64>> = b
        .par_iter()
        .map(|t| {
            let mut v = t.to_vec();
            let mut v_64: Vec<u64> = v.iter().map(|t| *t as u64).collect();
            v_64.resize(n, 0);
            ntt_pow_of_2_fast(&mut v_64, MOD_1 as u64, root_unity as u64);
            v_64
        })
        .collect();


    // let elapsed = now.elapsed();
    // println_with_timestamp!("Time 2: {:.2?}", elapsed);
    let now =   Instant::now();

    let n2 = extended_a.len() * 2;
    let n2_pow = n2.next_power_of_two();
    // println_with_timestamp!("n2_pow, {:?}", n2_pow);
    let root_unity_2 = crate::root_of_unity::choose_root_unity(n2_pow, MOD_1).unwrap();

    let extended_c_transposed: Vec<Vec<u64>> = (0..extended_a[0].len()).into_par_iter().map(|t| {
        let mut c_row = vec![0; n2 - 1];
        let mut at: Vec<u64> = extended_a.iter().map(|ai| ai[t] as u64).collect();
        let mut bt: Vec<u64> = extended_b.iter().map(|bi| bi[t] as u64).collect();
        at.resize(n2_pow, 0);
        bt.resize(n2_pow, 0);

        rayon::scope(|s| {
            s.spawn(|_| {
                ntt_pow_of_2_fast(&mut at, MOD_1 as u64, root_unity_2 as u64);
            });
            s.spawn(|_| {
                ntt_pow_of_2_fast(&mut bt, MOD_1 as u64, root_unity_2 as u64);
            });
        });


        let mut ct = hadamard_64(&at, &bt, MOD_1 as u64);

        reduce_mod_imbalanced_u64(&mut ct, MOD_1 as u64);


        inverse_ntt_pow_of_2_fast(&mut ct, MOD_1 as u64, root_unity_2 as u64);

        for (j, value) in ct.iter().enumerate().take(n2 - 1) {
            c_row[j] = *value;
        }
        c_row.iter().map(|u| u.clone() as u64).collect()
    }).collect();

    // let elapsed = now.elapsed();
    // println_with_timestamp!("Time 3: {:.2?}", elapsed);

    transpose(&extended_c_transposed)
        .into_par_iter()
        .map(|mut v| {
            inverse_ntt_pow_of_2_fast(&mut v, MOD_1 as u64, root_unity as u64);
            let v_128: Vec<i128> = v.iter().map(|t| *t as i128).collect();
            let mut v_intt = reduce_quotient(&v_128, &MIN_POLY);
            reduce_mod(&mut v_intt, MOD_1);
            v_intt
        })
        .collect()
}

#[test]
fn test_convolution() {
    let len = LOG_Q * CHUNK_SIZE * 4;
    // Generate a random witness vector
    let witness: Vec<RingElement> = vec![0; len].iter().map(|_| {
        Ring::random_bin()
    }).collect();

    // Compute the convolution of the witness vector
    let convoluted = convolution(&witness);

    // Compute the last element of the convolution result
    let t1 = RingElement {
        coeffs: <[BASE_INT; PHI]>::try_from(convoluted[witness.len() - 1].clone()).unwrap(),
    };

    // Compute the sum of the product of witness elements and all one conjugate
    let mut sum = Ring::zero();
    let all_one_conj = Ring::all(1).conjugate();
    for b in witness.iter() {
        sum = &sum + &(&all_one_conj * b);
    }

    let t = sum - t1;
    // Verify the twisted trace
    assert_eq!(t.twisted_trace(), 0);
}


#[test]
fn test_convolution_big() {
    let witness_transposed = sample_random_bin_mat(CHUNKS,  CHUNKS * 1000);
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
    println_with_timestamp!("Time: {:.2?}", elapsed);


}
