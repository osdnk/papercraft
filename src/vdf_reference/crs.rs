use crate::{ring_i128::{Ring, RingElement}, r#static::{LOG_Q}};
use crate::arithmetic::{sample_random_mat, sample_random_vector, sample_random_vector_non_real};
use crate::r#static::{CHUNK_SIZE, COMMITMENT_MODULE_SIZE, MODULE_SIZE};
use crate::ring_helpers::transpose;
use crate::static_rings::static_generated::{V_COEFFS, V_INV_COEFFS, CHALLENGE_SET, PHI};
use rayon::prelude::*;

/// Struct representing the Common Reference String (CRS) for cryptographic operations.
pub struct CRS {
    pub(crate) ck: Vec<Vec<RingElement>>,
    pub(crate) a: Vec<Vec<RingElement>>,
    pub(crate) challenge_set: Vec<RingElement>,
}

/// Generates a Common Reference String (CRS).
///
/// # Returns
///
/// A `CRS` containing commitment keys (`ck`) a randomly sampled vector (`a`), and a challenge set.
///
/// # Panics
///
/// This function will panic if the dimensions of `V_COEFFS` do not match the expected values.
pub fn gen_crs() -> CRS {
    let (v, v_inv) = (
        RingElement { coeffs: V_COEFFS },
        RingElement { coeffs: V_INV_COEFFS }
    );

    // TODO later use different Vs
    let v_module = sample_random_vector_non_real(COMMITMENT_MODULE_SIZE);

    let n_dim_log_q = CHUNK_SIZE * MODULE_SIZE * LOG_Q;

    let ck = compute_commitment_keys(v_module, n_dim_log_q);

    let a = sample_random_mat(MODULE_SIZE, LOG_Q * MODULE_SIZE);

    // Sample challenge set
    let challenge_set = CHALLENGE_SET.iter()
        .map(|e| RingElement { coeffs: <[i128; PHI]>::try_from(e.to_vec()).unwrap() })
        .collect();

    CRS { ck, a, challenge_set }
}




/// Computes commitment keys by raising the given module to successive powers.
///
/// # Arguments
///
/// * `module` - A vector of `RingElement`
/// * `chunk_size` - The chunk size.
/// * `log_q` - The logarithmic size of Q.
///
/// # Returns
///
/// A vector of vectors representing the computed commitment keys.
pub fn compute_commitment_keys(module: Vec<RingElement>, n_dim_log_q: usize) -> Vec<Vec<RingElement>> {
    module.into_par_iter().map(|m| {
        let mut row = Vec::with_capacity(n_dim_log_q);
        let mut power = m.clone();
        row.push(m.clone());
        for _ in 1..n_dim_log_q {
            power = power * m;
            row.push(power.clone());
        }
        row
    }).collect()
}


// #[test]
// fn test_compute_commitment_keys() {
//     let v = RingElement { coeffs: V_COEFFS };
//     let v_inv = RingElement { coeffs: V_INV_COEFFS };
//     let ck_dual = compute_commitment_keys(vec![v_inv], 10);
//     let ck = compute_commitment_keys(vec![v], 10);
//     assert_eq!(v * v_inv, Ring::constant(1));
//     assert_eq!(&ck[7][0] * &ck_dual[7][0], Ring::constant(1));
// }
