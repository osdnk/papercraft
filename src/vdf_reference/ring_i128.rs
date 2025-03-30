use std::ops::{Add, Div, Mul, Sub};
use serde::{Serialize, Deserialize};
use serde_json;
use crate::poly_arithmetic_i128;
use crate::poly_arithmetic_i128::{choose_root_unity, reduce_quotient, reduce_quotient_and_cyclotomic, choose_root_unity_fast, inverse_ntt_slow, karatsuba_mul, polynomial_mul, mod_inverse, random, reciprocal, reduce_mod, reduce_mod_imbalanced, reduce_with_prime_cyclotomic_polynomial};
use crate::r#static::{BASE_INT, LOG_Q, MOD_Q};
use crate::static_rings::static_generated::{DEGREE, MIN_POLY, BASIS, CONDUCTOR_COMPOSITE, INV_BASIS, TWIST, TRACE_COEFFS, PHI, V_COEFFS, V_INV_COEFFS, CONJUGATE_MAP};
use ndarray::{arr2, arr1, Array2, Array1, s, LinalgScalar};
use num_traits::{MulAdd, One, Zero};
use once_cell::sync::Lazy;
use rand::Rng;
use crate::arithmetic::{binary_decomposition, binary_decomposition_radix, call_sage_inverse_polynomial};
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct RingElement {
    pub coeffs: [i128; PHI]
}

impl Copy for RingElement {}

impl Zero for RingElement {
    fn zero() -> Self {
        Ring::zero()
    }

    fn is_zero(&self) -> bool {
        for r in self.coeffs {
            if r != 0 {
                return false
            }
        }
        true
    }
}

impl One for RingElement {
    fn one() -> Self {
        Ring::constant(1)
    }
}
impl Div for RingElement {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        panic!("no division implemented!");
    }
}

impl  PartialEq<Self> for RingElement {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}

impl Default for RingElement {
    fn default() -> Self {
        Ring::random()
    }
}

impl Add<Self> for &RingElement {
    type Output = RingElement;
    fn add(self, other: Self) -> RingElement {
        let mut coeffs = poly_arithmetic_i128::add(&self.coeffs, &other.coeffs);
        reduce_mod(&mut coeffs, MOD_Q);
        RingElement {
            coeffs: <[i128; PHI]>::try_from(coeffs).unwrap(),
        }
    }
}

impl Sub <Self> for &RingElement {
    type Output = RingElement;
    fn sub(self, other: Self) -> RingElement {
        let mut coeffs = poly_arithmetic_i128::sub(&self.coeffs, &other.coeffs);
        poly_arithmetic_i128::reduce_mod(&mut coeffs, MOD_Q);
        RingElement {
            coeffs: <[i128; PHI]>::try_from(coeffs).unwrap(),
        }
    }
}

impl Add for RingElement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut coeffs = poly_arithmetic_i128::add(&self.coeffs, &other.coeffs);
        poly_arithmetic_i128::reduce_mod(&mut coeffs, MOD_Q);
        Self {
            coeffs: <[i128; PHI]>::try_from(coeffs).unwrap(),
        }
    }
}

impl Sub for RingElement {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut coeffs = poly_arithmetic_i128::sub(&self.coeffs, &other.coeffs);
        poly_arithmetic_i128::reduce_mod(&mut coeffs, MOD_Q);
        Self {
            coeffs: <[i128; PHI]>::try_from(coeffs).unwrap(),
        }
    }
}

impl Mul for RingElement {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let product = crate::arithmetic::karatsuba_mul_par(&self.coeffs, &other.coeffs, Some(MOD_Q));
        let mut reduced = reduce_quotient_and_cyclotomic(&product, &MIN_POLY, CONDUCTOR_COMPOSITE);
        reduce_mod(&mut reduced, MOD_Q);
        Self {
            coeffs: <[i128; PHI]>::try_from(reduced).unwrap(),
        }
    }
}

impl Mul<Self> for &RingElement {
    type Output = RingElement;
    fn mul(self, other: Self) -> RingElement {
        let product = crate::arithmetic::karatsuba_mul_par(&self.coeffs, &other.coeffs, Some(MOD_Q));
        let mut reduced = reduce_quotient_and_cyclotomic(&product, &MIN_POLY, CONDUCTOR_COMPOSITE);
        reduce_mod(&mut reduced, MOD_Q);
        RingElement {
            coeffs: <[i128; PHI]>::try_from(reduced).unwrap(),
        }
    }
}

static INV_BASIS_ARR: Lazy<Array2<BASE_INT>> = Lazy::new(|| arr2(&INV_BASIS));
static CONJUGATE_MAP_ARR: Lazy<Array2<BASE_INT>> = Lazy::new(|| arr2(&CONJUGATE_MAP));
static TRACE_COEFFS_ARR: Lazy<Array1<BASE_INT>> = Lazy::new(|| arr1(&TRACE_COEFFS));
impl RingElement {
    pub fn to_vector(&self) -> Vec<i128>{
        let mut coeffs2 = vec![0; DEGREE];
        let coeffs_vec = arr1(&self.coeffs);
        coeffs2 = INV_BASIS_ARR.dot(&coeffs_vec).to_vec();
        coeffs2
    }

    pub fn one_minus(&self) -> RingElement {
        let one = Ring::all(1);
        &one - self
    }

    pub fn conj_one_minus(&self) -> RingElement {
        let one = Ring::all(1);
        &one.conjugate() - self
    }

    pub fn minus(&self) -> RingElement {
        &Ring::zero() - self
    }

    pub fn twisted_trace(&self) -> i128 {
        let twist = RingElement {
            coeffs: TWIST,
        };


        let twisted = self * &twist;
        let trace_res = TRACE_COEFFS_ARR.dot(&arr1(&twisted.coeffs));
        trace_res
    }

    pub fn conjugate(&self) -> RingElement {
        let coeffs_vec = arr1(&self.coeffs);
        let coeffs_conjugated = CONJUGATE_MAP_ARR.dot(&coeffs_vec).to_vec();
        RingElement {
            coeffs: <[BASE_INT; PHI]>::try_from(coeffs_conjugated).unwrap()
        }
    }

    pub fn inverse(&self) -> RingElement {
        call_sage_inverse_polynomial(self).unwrap()
    }

    pub fn g_decompose(&self) -> Vec<RingElement> {
        let mut coeffs = self.to_vector();
        // we need imbalanced representation for decomposition
        reduce_mod_imbalanced(&mut coeffs, MOD_Q);
        let coeffs_decomposed = poly_arithmetic_i128::binary_decomposition(&coeffs, LOG_Q);
        coeffs_decomposed.iter().map(|x|
        Ring::new(x.clone())
        ).collect::<Vec<_>>()
    }



    pub fn g_decompose_coeffs(&self, chunks: usize) -> Vec<RingElement> {
        let mut coeffs = self.coeffs.clone().to_vec();
        // we need imbalanced representation for decomposition
        reduce_mod_imbalanced(&mut coeffs, MOD_Q);
        let coeffs_decomposed = binary_decomposition(&coeffs, chunks);
        coeffs_decomposed.iter().map(|x|
        RingElement {
            coeffs:  <[BASE_INT; PHI]>::try_from(x.clone()).unwrap(),
        }
        ).collect::<Vec<_>>()
    }

    pub fn g_decompose_coeffs_base(&self, chunks: usize, base: BASE_INT) -> Vec<RingElement> {
        let mut coeffs = self.coeffs.clone().to_vec();
        // we need imbalanced representation for decomposition
        reduce_mod_imbalanced(&mut coeffs, MOD_Q);
        let coeffs_decomposed = binary_decomposition_radix(&coeffs, base, chunks);
        coeffs_decomposed.iter().map(|x|
        RingElement {
            coeffs:  <[BASE_INT; PHI]>::try_from(x.clone()).unwrap(),
        }
        ).collect::<Vec<_>>()
    }

    pub fn inf_norm(&self) -> BASE_INT {
        let mut coeffs = self.coeffs.clone().to_vec();
        reduce_mod_imbalanced(&mut coeffs, MOD_Q);
        *coeffs.iter().max().unwrap()
    }
}

pub struct Ring;

static BASIS_ARR: Lazy<Array2<i128>> = Lazy::new(|| arr2(&BASIS));

impl Ring {
    pub fn new(coeffs: Vec<i128>) -> RingElement {
        let mut coeff2 = vec![0; CONDUCTOR_COMPOSITE - 1];

        if coeffs.len() != DEGREE {
            panic!("The input size needs to be equal to the degree!");
        }


        let coeffs_vec = arr1(&coeffs);
        coeff2 = BASIS_ARR.dot(&coeffs_vec).to_vec();

        reduce_mod(&mut coeff2, MOD_Q);
        RingElement {
            coeffs: <[i128; PHI]>::try_from(coeff2).unwrap(),
        }
    }

    pub fn zero() -> RingElement {
        Ring::new( vec![0; DEGREE])
    }

    pub fn all(v: i128) -> RingElement {
        Ring::new( vec![v; DEGREE])
    }

    pub fn constant(v: i128) -> RingElement {
        let mut coeffs = vec![0; PHI];
        coeffs[0] = v;
        RingElement {
            coeffs: <[i128; PHI]>::try_from(coeffs).unwrap(),
        }
    }

    pub fn random() -> RingElement {
        Ring::new( random(DEGREE, MOD_Q))
    }

    pub fn random_non_real() -> RingElement {
        RingElement {
            coeffs: <[BASE_INT; PHI]>::try_from(random(PHI, MOD_Q)).unwrap()
        }
    }

    pub fn random_bin() -> RingElement {
        Ring::new( poly_arithmetic_i128::random(DEGREE, 2))
    }

    pub fn random_constant_bin() -> RingElement {
        let mut rng = rand::thread_rng();
        let number = rng.gen_range(0..2);
        Ring::constant(number)
    }
}

pub fn ring_inner_product(a: &Vec<RingElement>, b: &Vec<RingElement>) -> RingElement {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");

    a.par_iter()
        .zip(b.par_iter())
        .map(|(a_i, b_i)| a_i * b_i)
        .reduce(Ring::zero, |acc, prod| acc + prod)
}

pub fn reshape(vec: Vec<RingElement>, chunks_size: usize) -> Vec<Vec<RingElement>> {
    vec.chunks(chunks_size).map(|c| c.clone().to_vec()).collect()
}

pub fn get_g(size: usize) -> Vec<RingElement> {
    let mut result = Vec::with_capacity(size);
    for i in (0..size).rev() {
        let number = Ring::constant(i128::pow(2, i as u32));
        result.push(number);
    }
    result
}

pub fn get_g_custom(size: usize, radix: i128) -> Vec<RingElement> {
    let mut result = Vec::with_capacity(size);
    for i in (0..size).rev() {
        let number = Ring::constant(i128::pow(radix, i as u32));
        result.push(number);
    }
    result
}


// #[test]
// fn test_inverse() {
//     let v = RingElement {
//         coeffs: <[i128; PHI]>::try_from(V_COEFFS.to_vec()).unwrap()
//     };
//
//     let v_inv = RingElement {
//         coeffs: <[i128; PHI]>::try_from(V_INV_COEFFS.to_vec()).unwrap()
//     };
//
//     assert_eq!(v * v_inv, Ring::constant(1));
// }

#[test]
fn test_basis() {
    let el = Ring::random();
    assert_eq!(Ring::new(el.to_vector()), el);
}
