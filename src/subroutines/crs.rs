use num_traits::One;
use crate::arithmetic::{PowerSeries, sample_random_vector, sample_random_mat, sample_random_mat_subring};
use crate::custom_ring::r#static::{LOG_Q, MODULE_SIZE, WIT_DIM};
use crate::custom_ring::ring::{DPrimeRingElement, Ring, RingElement};
use rayon::prelude::*;

/// Struct representing the Common Reference String (CRS) for cryptographic operations.
pub struct CRS {
    pub(crate) ck: Vec<PowerSeries>,
    pub(crate) a: Vec<Vec<DPrimeRingElement>>,
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

impl CRS {
    pub fn gen_crs(wit_dim: usize, module_size: usize) -> CRS {
        let v_module = sample_random_vector(module_size);

        let ck = compute_commitment_keys(v_module, wit_dim);

        let a = sample_random_mat_subring(MODULE_SIZE, LOG_Q * MODULE_SIZE);

        CRS { ck, a }
    }
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
pub fn compute_commitment_keys(module: Vec<DPrimeRingElement>, wit_dim: usize) -> Vec<PowerSeries> {
    module.into_par_iter().map(|m| {
        let mut row = Vec::with_capacity(wit_dim);
        let mut power = m.clone();
        row.push(m.clone());
        for _ in 1..wit_dim {
            power = power * m;
            row.push(power.clone());
        }
        let mut ps = PowerSeries {
            expanded_layers: vec![],
            tensors: vec![],
        };
        let mut current_dim = wit_dim;
        while current_dim % 2 == 0 {
            ps.expanded_layers.push(row[0..current_dim].to_vec());
            current_dim /= 2;
            ps.tensors.push(vec![RingElement::one(), row[current_dim - 1]]);
        }
        ps.expanded_layers.push(row[0..current_dim].to_vec());
        ps
    }).collect()
}

