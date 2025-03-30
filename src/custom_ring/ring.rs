#![feature(generic_const_exprs)]
#![allow(incomplete_features, unused)]

use std::cell::UnsafeCell;
use std::iter::Sum;
use std::ops::{Add, Mul, Sub};
use std::sync::Mutex;
use ndarray::{arr1, arr2, Array2};
use num_traits::{One, Zero};
use once_cell::sync::Lazy;
use crate::custom_ring::r#static::{MOD_Q, PHI, TWO_PHI_MINUS_ONE, TEST_PHI, TEST_TWO_PHI_MINUS_ONE, MODS_Q, NOF_PRIMES, MOD_Q_1, FHE_Q};
use crate::helpers::println_with_timestamp;
use rand::Rng;
use crate::arithmetic::{add_mod, call_sage_inverse_polynomial, decompose, decompose_matrix_by_radix, first_n_columns, multiply_mod, random, reduce_mod, reduce_mod_vec, reduce_quotient_and_cyclotomic, sub_mod};
use crate::custom_ring::static_generated_24::{BASIS, CONDUCTOR_COMPOSITE, CONJUGATE_MAP, DEGREE, INV_BASIS, MIN_POLY, TRACE_COEFFS, TWIST};

static BASIS_ARR: Lazy<Array2<u64>> = Lazy::new(|| arr2(&BASIS));
static BASIS_INV_ARR: Lazy<Array2<u64>> = Lazy::new(|| arr2(&INV_BASIS));

pub type DPrimeRingElement = RingElement<PHI, TWO_PHI_MINUS_ONE, MOD_Q>;

#[derive(Clone, Copy, Debug)]
pub struct RingElement<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> {
    pub coeffs: [u64; phi]
}


static nof_adds: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(0));
static nof_mul: Lazy<Mutex<u64>> = Lazy::new(|| Mutex::new(0));

impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> Add for RingElement<phi, two_phi_minus_one, mod_q> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut result = [0; phi];
        for i in 0..phi {
            result[i] = (self.coeffs[i] + other.coeffs[i]) % mod_q;
            if result[i] < 0u64 { result[i] += mod_q; }
        }
        RingElement { coeffs: result }
    }
}
impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> Sum for RingElement<phi, two_phi_minus_one, mod_q> {
    fn sum<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(RingElement::zero(), |acc, x| acc + x)
    }
}



impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> Sub for RingElement<phi, two_phi_minus_one, mod_q> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let mut result = [0; phi];
        for i in 0..phi {
            result[i] = (mod_q + self.coeffs[i] - other.coeffs[i]) % mod_q;
        }
        RingElement { coeffs: result }
    }
}
pub fn poly_mul_mod<const phi:usize, const two_phi_minus_one:usize>
(a: &[u64; phi], b: &[u64; phi], mod_q: u64) -> [u64; two_phi_minus_one] {
    let mut result = [0u64; two_phi_minus_one];
    for i in 0..phi {
        for j in 0..phi {
            result[i + j] = (((result[i + j] as u128) + (a[i] as u128) * (b[j] as u128)) % (mod_q as u128)) as u64;
        }
    }
    result
}

impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> Mul for RingElement<phi, two_phi_minus_one, mod_q> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // let nof_mul_curr = *nof_mul.lock().unwrap();
        // *nof_mul.lock().unwrap() = nof_mul_curr + 1;

        let mut product = poly_mul_mod::<phi, two_phi_minus_one>(&self.coeffs, &other.coeffs, mod_q);

        let mut reduced = reduce_quotient_and_cyclotomic(&product, &MIN_POLY, CONDUCTOR_COMPOSITE);

        for i in 0..phi {
            reduced[i] = reduced[i] % mod_q;
        }

        RingElement { coeffs: <[u64; phi]>::try_from(reduced).unwrap() }
    }
}

impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> Zero for RingElement<phi, two_phi_minus_one, mod_q> {
    fn zero() -> Self {
        RingElement {
            coeffs: [0; phi],
        }
    }

    fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&coeff| coeff == 0)
    }
}


impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> One for RingElement<phi, two_phi_minus_one, mod_q> {
    fn one() -> Self {
        let mut coeffs = [0; phi];
        coeffs[0] = 1;
        RingElement { coeffs }
    }
}

impl<
    const PHI: usize,
    const TWO_PHI_MINUS_ONE: usize,
    const mod_q: u64
> PartialEq for RingElement<PHI, TWO_PHI_MINUS_ONE, mod_q> {
    fn eq(&self, other: &Self) -> bool {
        self.coeffs == other.coeffs
    }
}



impl<
    const phi: usize,
    const two_phi_minus_one: usize,
    const mod_q: u64
> RingElement<phi, two_phi_minus_one, mod_q> {

    pub fn conjugate(&self) -> RingElement<phi, two_phi_minus_one, mod_q> {
        let mut new_coeffs:[u64; phi] = [0u64; phi];

        for i in 0..CONJUGATE_MAP.len() {
            for j in 0..CONJUGATE_MAP[i].len() {
                new_coeffs[i] = add_mod(new_coeffs[i], multiply_mod(CONJUGATE_MAP[i][j], self.coeffs[j], MOD_Q), MOD_Q);
            }
        }

        RingElement {
            coeffs: new_coeffs
        }
    }

    pub fn trace(&self) -> u64 {
        let mut result = multiply_mod(self.coeffs[0], phi as u64, mod_q);
        for i in 1..PHI {
            result = sub_mod(result, self.coeffs[i], mod_q);
        }
        result
    }


    pub fn scaled_circulant_rep(&self, index: usize) -> [u64; NOF_PRIMES] {
        let mut result = [0u64; NOF_PRIMES];
        for j in 0..NOF_PRIMES {
            result[j] = multiply_mod(self.coeffs[index], phi as u64, mod_q);
            for i in 0..PHI {
                if i == index {
                    continue
                }
                result[j] = sub_mod(result[j], self.coeffs[i], mod_q);
            }
        }
        result
    }



    pub fn negative(&self) -> RingElement<phi, two_phi_minus_one, mod_q> {
        RingElement::<phi, two_phi_minus_one, mod_q>::zero() - *self
    }
}

impl DPrimeRingElement {
    pub fn inverse(&self) -> DPrimeRingElement {
        call_sage_inverse_polynomial(self).unwrap()
    }
    pub fn twisted_trace(&self) -> u64 {
        let twist = DPrimeRingElement {
            coeffs: TWIST,
        };


        let twisted = self.clone() * twist;
        TRACE_COEFFS.iter()
            .zip(twisted.coeffs.iter())
            .map(|(a, b)| multiply_mod(*a, *b, MOD_Q))
            .sum()
    }
}

pub struct Ring<
    const phi: usize = PHI,
    const two_phi_minus_one: usize = TWO_PHI_MINUS_ONE,
    const mod_q: u64 = MOD_Q,
>;

impl Ring {
    // Constructor for PrimeRingElement
    pub fn new(coeffs: [u64; PHI]) -> DPrimeRingElement {
        DPrimeRingElement { coeffs }
    }

    pub fn new_rns(coeffs: [u64; PHI]) -> DPrimeRingElement {
        DPrimeRingElement { coeffs }
    }
    pub fn new_from_larger_vec(coeffs: &Vec<u64>) -> DPrimeRingElement {
        let mut reduced_result = [0; PHI];
        let mut reduced = reduce_quotient_and_cyclotomic(&coeffs, &MIN_POLY, CONDUCTOR_COMPOSITE);

        for i in 0..PHI {
            reduced[i] = reduced[i] % MOD_Q;
        }

        RingElement { coeffs: <[u64; PHI]>::try_from(reduced).unwrap() }
    }

    pub fn new_small_tests_only(coeffs: [u64; TEST_PHI]) -> RingElement<TEST_PHI, TEST_TWO_PHI_MINUS_ONE, MOD_Q> {
        RingElement::<TEST_PHI, TEST_TWO_PHI_MINUS_ONE, MOD_Q> { coeffs: coeffs }
    }

    pub fn constant(v: u64) -> DPrimeRingElement {
        let mut coeffs = [0; PHI];
        coeffs[0] = v;
        RingElement { coeffs }
    }

    pub fn all(v: u64) -> DPrimeRingElement<> {
        Ring::new( [v; PHI])
    }
    pub fn random_all() -> DPrimeRingElement<> {
        // TODO
        Ring::all( random(1, MODS_Q[0])[0])
    }

    pub fn sample_subtractive() -> DPrimeRingElement {
        let mut rng = rand::thread_rng();

        // Randomly select an index i from 0 to PHI
        let i: usize = rng.gen_range(0..3);
        let mut coeffs= [0u64; PHI];
        coeffs[i] = 1;

        RingElement { coeffs }
    }

    pub fn random() -> DPrimeRingElement {
        Ring::new(
            random(PHI, MOD_Q).try_into().unwrap()
        )
    }

    pub fn random_subring() -> DPrimeRingElement {
        let coeffs = random(DEGREE, MOD_Q);
        let mut new_coeffs = [0u64; PHI];
        for i in 0..BASIS.len() {
            for j in 0..BASIS[i].len() {
                new_coeffs[i] = add_mod(new_coeffs[i], multiply_mod(BASIS[i][j], coeffs[j], MOD_Q), MOD_Q);
            }
        }
        DPrimeRingElement {
            coeffs: new_coeffs
        }
    }

    pub fn all_subring(v: u64) -> DPrimeRingElement<> {
        let coeffs = [v; DEGREE];
        let mut new_coeffs = [0u64; PHI];
        for i in 0..BASIS.len() {
            for j in 0..BASIS[i].len() {
                new_coeffs[i] = add_mod(new_coeffs[i], multiply_mod(BASIS[i][j], coeffs[j], MOD_Q), MOD_Q);
            }
        }
        DPrimeRingElement {
            coeffs: new_coeffs
        }
    }

    pub fn random_subring_bin() -> DPrimeRingElement {
        let coeffs = random(DEGREE, 2);
        let mut new_coeffs = [0u64; PHI];
        for i in 0..BASIS.len() {
            for j in 0..BASIS[i].len() {
                new_coeffs[i] = add_mod(new_coeffs[i], multiply_mod(BASIS[i][j], coeffs[j], MOD_Q), MOD_Q);
            }
        }
        DPrimeRingElement {
            coeffs: new_coeffs
        }
    }

    pub fn random_constant() -> DPrimeRingElement {
        // TODO
        Ring::constant(
            random(1, MOD_Q)[0]
        )
    }
    pub fn random_short() -> DPrimeRingElement {
        Ring::new(random(PHI, 10).try_into().unwrap())
    }

    pub fn random_from(l: u64) -> DPrimeRingElement {
        Ring::new(random(PHI, l).try_into().unwrap())
    }

    pub fn random_constant_from(l: u64) -> DPrimeRingElement {
        Ring::constant(
            random(1, l)[0]
        )
    }

}





