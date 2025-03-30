use rayon::prelude::*;

use crate::arithmetic::{parallel_dot_matrix_matrix, parallel_dot_series_matrix, sample_random_mat};
use crate::custom_ring::ring::DPrimeRingElement;
use crate::subroutines::crs::CRS;

/// The output of the split operation, containing the new RHS, the witness center, and the new witness.

pub struct FoldOutput {
    pub(crate) new_witness: Vec<Vec<DPrimeRingElement>>,
}


/// Performs a fold operation on the given witness and challenge matrices.
///
/// # Arguments
///
/// * `witness` - A reference to a matrix represented as a vector of vectors of `RingElement`.
/// * `challenge` - A reference to a matrix represented as a vector of vectors of `RingElement`.
///
/// # Returns
///
/// A `FoldOutput` instance containing the new witness matrix resulting from the dot product of the witness and challenge matrices.
///
pub fn fold(witness: &Vec<Vec<DPrimeRingElement>>, challenge: &Vec<Vec<DPrimeRingElement>>) -> Vec<Vec<DPrimeRingElement>> {
    parallel_dot_matrix_matrix(&witness, &challenge)
}



#[test]
fn test_fold() {
    // Generate the CRS and sample a random witness
    let ck = CRS::gen_crs(8, 1).ck;
    let witness = sample_random_mat(ck[0].expanded_layers[0].len(), 4);


    let commitment = parallel_dot_series_matrix(&ck, &witness);

    let challenge = sample_random_mat(4 ,2);

    assert_eq!(parallel_dot_series_matrix(
        &ck,
        &witness
    ), commitment);



    let folded_witness = fold(&witness, &challenge);
    let folded_commitment = parallel_dot_matrix_matrix(&commitment, &challenge);

    assert_eq!(parallel_dot_series_matrix(&ck, &folded_witness), folded_commitment);
}
