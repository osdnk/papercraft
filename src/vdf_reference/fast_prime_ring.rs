// use ndarray::{arr1, arr2, Array2};
// use once_cell::sync::Lazy;
// use rand::Rng;
// use crate::arithmetic::{binary_decomposition, binary_decomposition_radix, call_sage_inverse_polynomial};
// use crate::poly_arithmetic_i128;
// use crate::poly_arithmetic_i128::{random, reduce_mod, reduce_mod_imbalanced};
// use crate::r#static::{BASE_INT, LOG_Q, MOD_Q};
// use crate::ring_i128::RingElement;
// /// MOVE TO STATIC FILE
// pub const CONDUCTOR: usize = 1021;
// pub const PHI: usize = 1020;
//
// #[derive(Clone, Debug)]
// pub struct FastPrimeRingElement {
//     pub coeffs: [i64; PHI]
// }
// impl FastPrimeRingElement {
//     pub fn to_vector(&self) -> Vec<i128>{
//         panic!()
//     }
//
//     pub fn one_minus(&self) -> RingElement {
//         panic!()
//     }
//
//     pub fn conj_one_minus(&self) -> RingElement {
//         panic!()
//     }
//
//     pub fn minus(&self) -> RingElement {
//         panic!()
//     }
//
//     // pub fn twisted_trace(&self) -> i128 {
//     //     panic!()
//     // }
//
//     pub fn conjugate(&self) -> RingElement {
//         panic!()
//     }
//
//     pub fn inverse(&self) -> RingElement {
//         panic!()
//         // call_sage_inverse_polynomial(self).unwrap(
//         //
//         // )
//     }
//
//     pub fn g_decompose(&self) -> Vec<RingElement> {
//         panic!()
//         // let mut coeffs = self.to_vector();
//         // // we need imbalanced representation for decomposition
//         // reduce_mod_imbalanced(&mut coeffs, MOD_Q);
//         // let coeffs_decomposed = poly_arithmetic_i128::binary_decomposition(&coeffs, LOG_Q);
//         // coeffs_decomposed.iter().map(|x|
//         // crate::ring_i128::Ring::new(x.clone())
//         // ).collect::<Vec<_>>()
//     }
//
//
//
//     pub fn g_decompose_coeffs(&self, chunks: usize) -> Vec<RingElement> {
//         panic!()
//         //
//         // let mut coeffs = self.coeffs.clone().to_vec();
//         // // we need imbalanced representation for decomposition
//         // reduce_mod_imbalanced(&mut coeffs, MOD_Q);
//         // let coeffs_decomposed = binary_decomposition(&coeffs, chunks);
//         // coeffs_decomposed.iter().map(|x|
//         // RingElement {
//         //     coeffs:  <[BASE_INT; PHI]>::try_from(x.clone()).unwrap(),
//         // }
//         // ).collect::<Vec<_>>()
//     }
//
//     pub fn inf_norm(&self) -> BASE_INT {
//         panic!()
//         // let mut coeffs = self.coeffs.clone().to_vec();
//         // reduce_mod_imbalanced(&mut coeffs, MOD_Q);
//         // *coeffs.iter().max().unwrap()
//     }
// }
//
// pub struct Ring;
//
// impl crate::ring_i128::Ring {
//     pub fn new(coeffs: [i64; PHI]) -> FastPrimeRingElement {
//         FastPrimeRingElement {
//             coeffs: coeffs,
//         }
//     }
//
//     pub fn zero() -> RingElement {
//         crate::ring_i128::Ring::new( vec![0; PHI])
//     }
//
//     pub fn all(v: i128) -> RingElement {
//         crate::ring_i128::Ring::new( vec![v; PHI])
//     }
//
//     pub fn constant(v: i128) -> RingElement {
//         let mut coeffs = vec![0; PHI];
//         coeffs[0] = v;
//         RingElement {
//             coeffs: <[i128; PHI]>::try_from(coeffs).unwrap(),
//         }
//     }
//
//     pub fn random() -> RingElement {
//         crate::ring_i128::Ring::new( random(PHI, MOD_Q))
//     }
//
//     pub fn random_non_real() -> RingElement {
//         RingElement {
//             coeffs: <[BASE_INT; PHI]>::try_from(random(PHI, MOD_Q)).unwrap()
//         }
//     }
//
//     pub fn random_bin() -> RingElement {
//         crate::ring_i128::Ring::new( poly_arithmetic_i128::random(PHI, 2))
//     }
//
//     pub fn random_constant_bin() -> RingElement {
//         let mut rng = rand::thread_rng();
//         let number = rng.gen_range(0..2);
//         crate::ring_i128::Ring::constant(number)
//     }
// }
