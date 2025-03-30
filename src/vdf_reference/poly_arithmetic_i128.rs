use rand::Rng;
use sha3::digest::Output;
use crate::r#static::{MOD_Q, MOD_1, BASE_INT};
use crate::static_rings::static_generated::{MIN_POLY, PHI};
use std::ops::{Add,Sub,Mul,Div};
use std::sync::Mutex;
use std::time::Instant;
use fast_modulo::{mod_u128u64_unchecked, mulmod_u64, powmod_u64};
use num_traits::{zero, Zero};
use rayon::prelude::*;
use crate::ring_i128::RingElement;
// DO NOT USE THIS FILE. Rewrite the function and put to arithmetic.

pub fn binary_decomposition(coeffs: &Vec<i128>, log_q: usize) -> Vec<Vec<i128>> {
    let mut cloned = coeffs.clone();

    cloned.iter_mut().for_each(|x| *x = *x % (1 << log_q));
    let binary_coeffs: Vec<Vec<i128>> = cloned.into_iter().map(|x| {
        let mut binary = Vec::with_capacity(log_q);
        for i in (0..log_q).rev() {
            binary.push((x >> i) & 1);
        }
        assert_eq!(x >> log_q, 0);
        binary
    }).collect();


    let mut result: Vec<Vec<i128>> = vec![vec![0; binary_coeffs.len()]; log_q as usize];
    for (i, row) in binary_coeffs.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            result[j][i] = *value;
        }
    }

    result
}



pub type RingElement32 = [i128];




pub fn reduce_quotient(a: &RingElement32, b: &RingElement32) -> Vec<i128> {
    let mut a = a.to_vec();
    let b_lead = *b.last().unwrap();  // the leading coefficient of b is at the end

    for i in (b.len()-1..a.len()).rev() {
        let factor = a[i] / b_lead;
        if factor != 0 {
            for (j, &b_val) in b.iter().rev().enumerate() {
                a[i - j] -= factor * b_val;
            }
        }
    }

    let (reduced, _) = a.split_at(b.len() - 1);
    reduced.to_vec()
}

pub fn reduce_quotient_generic<A: Zero + Div<Output=A> + Sub<Output=A> + Add<Output=A> + Mul<Output=A> + Copy>(a: &[A], b: &[A]) -> Vec<A> {
    let mut a = a.to_vec();
    let b_lead = *b.last().unwrap();  // the leading coefficient of b is at the end

    for i in (b.len()-1..a.len()).rev() {
        let factor = a[i] / b_lead;
        if !factor.is_zero() {
            for (j, &b_val) in b.iter().rev().enumerate() {
                a[i - j] = a[i - j] - factor * b_val;
            }
        }
    }

    let (reduced, _) = a.split_at(b.len() - 1);
    reduced.to_vec()
}

pub fn  reduce_quotient_and_cyclotomic(a: &RingElement32, b: &RingElement32, conductor: usize)  -> Vec<i128> {
    let mut res = reduce_with_quasiprime_cyclotomic_polynomial(a, conductor);
    res = reduce_quotient(&res, &b);
    res
}

pub fn reduce_quotient_and_cyclotomic_generic<A: Zero + Div<Output=A> + Sub<Output=A> + Add<Output=A> + Mul<Output=A> + Copy>(a: &[A], b: &[A], conductor: usize)  -> Vec<A> {
    let mut res = reduce_with_quasiprime_cyclotomic_polynomial_generic(a, conductor);
    res = reduce_quotient_generic(&res, &b);
    res
}


// For prime f, we reduce first with X^f - 1 and then witn (1 + X + ... + X^{f-1})
// so we end up with a polynomial of degree f-2.
pub fn reduce_with_prime_cyclotomic_polynomial(a: &RingElement32, conductor: usize)-> Vec<i128> {
    let mut quotient_1 = vec![0; conductor + 1];
    quotient_1[conductor] = 1;
    quotient_1[0] = -1;
    let mut res = reduce_quotient(&a, &quotient_1);
    let quotient_2 = vec![1; conductor];
    res = reduce_quotient(&res, &quotient_2);
    res
}

pub fn reduce_with_quasiprime_cyclotomic_polynomial(a: &RingElement32, conductor: usize)-> Vec<i128> {
    let mut res = vec![0; conductor];
    for i in 0..a.len() {
        res[i % conductor] += a[i];
    }
    res
}


pub fn reduce_with_quasiprime_cyclotomic_polynomial_generic<A: Zero + Sub<Output=A> + Add<Output=A> + Mul<Output=A> + Copy>(a: &[A], conductor: usize)-> Vec<A> {
    let mut res = vec![A::zero(); conductor];
    for i in 0..a.len() {
        res[i % conductor] = res[i % conductor] + a[i];
    }
    res
}

pub fn karatsuba_mul(a: &RingElement32, b: &RingElement32, mod_q: Option<i128>) -> Vec<i128> {
    if a.len() != b.len() {
        panic!("Karatsube algororithm accepts arrays of the same length! {:?} {:?}", a.len(), b.len());
    }

    let n = a.len();
    if n <= 4 {
        return polynomial_mul(a, b, mod_q);
    }

    let mid = n / 2;

    let low_a = &a[..mid];
    let high_a = &a[mid..];
    let low_b = &b[..mid];
    let high_b = &b[mid..];

    let z0 = karatsuba_mul(low_a, low_b, mod_q);
    let z2 = karatsuba_mul(high_a, high_b, mod_q);

    let mut sum_a = add(low_a, high_a);
    let mut sum_b = add(low_b, high_b);
    if mod_q.is_some() {
        reduce_mod(&mut sum_b, mod_q.unwrap());
        reduce_mod(&mut sum_a, mod_q.unwrap());
    }
    let z1 = sub(&karatsuba_mul(&sum_a, &sum_b, mod_q), &add(&z0, &z2));

    let mut res = vec![0; 2*n - 1];

    for i in 0..z0.len() {
        res[i] += z0[i];
    }

    for i in 0..z1.len() {
        res[i + mid] += z1[i];
    }

    for i in 0..z2.len() {
        res[i + 2*mid] += z2[i];
    }

    if mod_q.is_some() {
        reduce_mod(&mut res, mod_q.unwrap());
    }

    res
}


pub fn karatsuba_mul_generic<A: Zero + Sub<Output=A> + Add<Output=A> + Mul<Output=A> + Copy>(a: &[A], b: &[A]) -> Vec<A> {
    if a.len() != b.len() {
        panic!("Karatsuba algorithm accepts arrays of the same length! {:?} {:?}", a.len(), b.len());
    }

    let n = a.len();
    if n <= 4 {
        return polynomial_mul_generic(a, b);
    }

    let mid = n / 2;

    let low_a = &a[..mid];
    let high_a = &a[mid..];
    let low_b = &b[..mid];
    let high_b = &b[mid..];

    let z0 = crate::poly_arithmetic_i128::karatsuba_mul_generic(low_a, low_b);
    let z2 = crate::poly_arithmetic_i128::karatsuba_mul_generic(high_a, high_b);

    let mut sum_a = add(low_a, high_a);
    let mut sum_b = add(low_b, high_b);
    let z1 = sub(&crate::poly_arithmetic_i128::karatsuba_mul_generic(&sum_a, &sum_b), &add(&z0, &z2));

    let mut res = vec![A::zero(); 2*n - 1];

    for i in 0..z0.len() {
        res[i] = res[i] + z0[i];
    }

    for i in 0..z1.len() {
        res[i + mid] = res[i + mid] + z1[i];
    }

    for i in 0..z2.len() {
        res[i + 2*mid] = res[i + 2*mid] +  z2[i];
    }

    res
}

pub fn add<A: Add<Output=A> + Copy>(a: &[A], b: &[A]) -> Vec<A> {
    let mut res = Vec::new();
    let (short, long) = if a.len() < b.len() { (a, b) } else { (b, a) };
    for i in 0..short.len() {
        res.push(short[i] + long[i]);
    }
    for i in short.len()..long.len() {
        res.push(long[i]);
    }
    res
}


pub fn add_in_place(a: &mut RingElement32, b: &RingElement32) {
    for i in 0..a.len() {
        a[i] += b[i];
    }
}


pub fn hadamard(a: &RingElement32, b: &RingElement32) -> Vec<i128> {
    let (short, long) = if a.len() < b.len() { (a, b) } else { (b, a) };

    let mut res:Vec<i128> = short
        .iter()
        .zip(long.iter())
        .map(|(aa, ba)| aa * ba)
        .collect();

    res.extend(long[short.len()..].iter().cloned());
    res
}

pub fn hadamard_64(a: &[u64], b: &[u64], mod_q: u64) -> Vec<u64> {
    let (short, long) = if a.len() < b.len() { (a, b) } else { (b, a) };

    let mut res:Vec<u64> = short
        .iter()
        .zip(long.iter())
        .map(|(aa, ba)| mulmod_u64(*aa, *ba, mod_q))
        .collect();

    res.extend(long[short.len()..].iter().cloned());
    res
}

pub fn sub<A: Sub<Output=A> + Copy>(a: &[A], b: &[A]) -> Vec<A> {
    let mut res = Vec::new();
    for i in 0..a.len() {
        res.push(a[i] - b[i]);
    }
    res
}

pub fn polynomial_mul(a: &RingElement32, b: &RingElement32, mod_q: Option<i128>) -> Vec<i128> {
    let mut res = vec![0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            res[i + j] += ai * bj;
            if mod_q.is_some() {
                res[i + j] %= mod_q.unwrap();
            }
        }
    }

    if mod_q.is_some() {
        reduce_mod(&mut res, mod_q.unwrap());
    }

    res
}

pub fn polynomial_mul_generic<A: Zero + Sub<Output=A> + Add<Output=A> + Mul<Output=A> + Copy>(a: &[A], b: &[A]) -> Vec<A> {
    let mut res = vec![A::zero(); a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            res[i + j] = res[i + j] + ai * bj;
        }
    }

    res
}



pub fn choose_root_unity(n: usize, mod_q: i128) -> Option<i128> {
    for x in 1..mod_q {
        let mut y = modpow(x, n as i128, mod_q);
        if y == 1 {
            let mut is_root = true;
            for i in 1..n {
                y = modpow(x, i as i128, mod_q);
                if y == 1 {
                    is_root = false;
                    break;
                }
            }
            if is_root {
                return Some(x)
            }
        }
    }
    None
}

pub fn extended_euclidean(a: i128, b: i128) -> (i128, i128, i128) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = extended_euclidean(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

pub fn mod_inverse(a: i128, m: i128) -> i128 {
    let (_, x, _) = extended_euclidean(a, m);
    (x % m + m) % m
}

pub fn choose_root_unity_fast(n: usize, mod_q: i128) -> Option<i128> {
    for x in 2..mod_q {
        let y = mod_inverse(modpow(x, n as i128, mod_q), mod_q);
        if y == 1 {
            return Some(x);
        }
    }
    None
}



pub fn ntt_pow_of_2(a: &mut [i128], mod_q: i128, root_unity: i128) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let mut a_odd: Vec<_> = a.iter().step_by(2).cloned().collect();
    let mut a_even: Vec<_> = a.iter().skip(1).step_by(2).cloned().collect();

    if a.len() > 1000000 {
        rayon::scope(|s| {
            s.spawn(|_| {
                ntt_pow_of_2(a_odd.as_mut_slice(), mod_q, (root_unity * root_unity) % mod_q);
            });
            s.spawn(|_| {
                ntt_pow_of_2(a_even.as_mut_slice(), mod_q, (root_unity * root_unity) % mod_q);
            });
        });
    } else {
        ntt_pow_of_2(a_odd.as_mut_slice(), mod_q, (root_unity * root_unity) % mod_q);
        ntt_pow_of_2(a_even.as_mut_slice(), mod_q, (root_unity * root_unity) % mod_q);
    }



    for i in 0..n / 2 {
        a[i] = (a_odd[i] + modpow_64(root_unity as u64, i as u64, mod_q as u64) as i128 * a_even[i]) % mod_q;
        a[i + n / 2]  = (a_odd[i] - modpow_64(root_unity as u64, i as u64, mod_q as u64) as i128 * a_even[i] + mod_q) % mod_q;
    }
    //
    reduce_mod(a, mod_q);
}

pub fn ntt_pow_of_2_fast(a: &mut [u64], mod_q: u64, root_unity: u64) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let mut a_odd: Vec<_> = a.iter().step_by(2).cloned().collect();
    let mut a_even: Vec<_> = a.iter().skip(1).step_by(2).cloned().collect();

    if a.len() > 100 {
        rayon::scope(|s| {
            s.spawn(|_| {
                crate::poly_arithmetic_i128::ntt_pow_of_2_fast(a_odd.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64,mod_q as u64));
            });
            s.spawn(|_| {
                crate::poly_arithmetic_i128::ntt_pow_of_2_fast(a_even.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64,mod_q as u64));
            });
        });
    } else {
        crate::poly_arithmetic_i128::ntt_pow_of_2_fast(a_odd.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64,mod_q as u64));
        crate::poly_arithmetic_i128::ntt_pow_of_2_fast(a_even.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64,mod_q as u64));
    }

    // reduce_mod_imbalanced(a_even, mod_q);


    for i in 0..n / 2 {
        a[i] = mod_u128u64_unchecked((a_odd[i] as u128 + mod_q as u128 + mulmod_u64(powmod_u64(root_unity, i as u64, mod_q), a_even[i], mod_q) as u128), mod_q);
        a[i + n / 2]  = mod_u128u64_unchecked((a_odd[i] as u128 + mod_q as u128 - mulmod_u64(powmod_u64(root_unity, i as u64, mod_q), a_even[i], mod_q) as u128), mod_q);
    }
    //
    reduce_mod_imbalanced_u64(a, mod_q);
}

pub fn inverse_ntt_pow_of_2(a: &mut RingElement32, mod_q: i128, root_unity: i128) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let inv_n = modpow(n as i128, mod_q - 2, mod_q);
    let inv_root_unity = modpow(root_unity, mod_q - 2, mod_q);
    ntt_pow_of_2(a, mod_q, inv_root_unity);

    for i in 0..n {
        a[i] = a[i] * inv_n % mod_q;
    }

    reduce_mod(a, mod_q);
}

pub fn inverse_ntt_pow_of_2_fast(a: &mut [u64], mod_q: u64, root_unity: u64) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let inv_n = powmod_u64(n as u64, mod_q - 2, mod_q);
    let inv_root_unity = powmod_u64(root_unity, mod_q - 2, mod_q);

    ntt_pow_of_2_fast(a, mod_q, inv_root_unity);

    for i in 0..n {
        a[i] = mulmod_u64(a[i], inv_n, mod_q);
    }

    reduce_mod_imbalanced_u64(a, mod_q);
}


pub fn reduce_mod_imbalanced(a: &mut RingElement32, q: i128) -> &RingElement32 {
    for ai in a.iter_mut() {
        *ai %= q;
        while *ai < 0 {
            *ai += q;
        }
    }
    a
}

pub fn reduce_mod_imbalanced_u64(a: &mut [u64], q: u64) -> &[u64] {
    for ai in a.iter_mut() {
        *ai %= q;
        while *ai < 0 {
            *ai += q;
        }
    }
    a
}


pub fn reduce_mod(a: &mut RingElement32, q: i128) -> &RingElement32 {
    for ai in a.iter_mut() {
        *ai %= q;
        while *ai <= -q/2 {
            *ai += q;
        }

        while *ai > q/2 {
            *ai -= q;
        }
    }
    a
}




pub fn modpow(mut base: i128, mut exp: i128, modulus: i128) -> i128 {
    base %= modulus;
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mulmod_u64(result as u64, base as u64, modulus as u64) as i128;
        }
        base = mulmod_u64(base as u64, base as u64, modulus as u64) as i128;
        exp >>= 1;
    }
    result
}


pub fn modpow_64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    base %= modulus;
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mulmod_u64(result, base, modulus);
        }
        base = mulmod_u64(base, base, modulus);
        exp >>= 1;
    }
    result
}

pub fn ntt_slow(input: &RingElement32, root: i128, modulo:i128) -> Vec<i128> {
    let n = input.len();
    let mut output = vec![0; n];
    for i in 0..n {
        let mut sum = 0;
        for j in 0..n {
            let k = i * j % n;
            let tmp = (input[j] * modpow(root, k as i128, modulo) + sum) % modulo;
            sum = tmp as i128;
        }
        output[i] = sum as i128;
    }
    output
}

pub fn inverse_ntt_slow(input: &RingElement32, root: i128, modulo:i128) -> Vec<i128> {
    let output = ntt_slow(input.clone(), reciprocal(root, modulo), modulo);
    let scaler = reciprocal(input.len() as i128, modulo);
    output.iter().enumerate().map(|(i, _)| ((output[i] as i128) * scaler % modulo) as i128).collect()
}

pub fn reciprocal(n: i128, modulo: i128) -> i128 {
    let mut x = modulo;
    let mut y = n;
    let mut a = 0;
    let mut b = 1;
    while y != 0 {
        let temp = a - (x / y) * b;
        a = b;
        b = temp;
        let temp = x % y;
        x = y;
        y = temp;
    }
    if x == 1 {
        if a >= 0 { a as i128 } else { (a + modulo as i128) as i128 }
    } else {
        panic!("Arithmetic error occurred!")
    }
}



pub fn cyclic_mul_schoolbook(a: &RingElement32, b: &RingElement32) -> Vec<i128>  {
    let double_c = polynomial_mul(a, b, None);
    let c = reduce_with_prime_cyclotomic_polynomial(&double_c, a.len());
    c
}

pub fn cyclic_mul_karatsuba(a: &RingElement32, b: &RingElement32) -> Vec<i128>  {
    let double_c = karatsuba_mul(a, b, None);
    let c = reduce_with_prime_cyclotomic_polynomial(&double_c, a.len());
    c
}

pub fn cyclic_mul_ntt_pow_of_2(a: &RingElement32, b: &RingElement32, mod_q: i128) -> Vec<i128>  {
    let n = a.len().next_power_of_two()*2;
    let mut extended_a = a.to_vec();
    let mut extended_b = b.to_vec();
    extended_a.resize(n, 0);
    extended_b.resize(n, 0);
    let root_unity = choose_root_unity(n, mod_q).unwrap();
    ntt_pow_of_2(&mut extended_a, mod_q, root_unity);
    ntt_pow_of_2(&mut extended_b, mod_q, root_unity);
    let mut extended_c = hadamard(&extended_a, &extended_b);
    inverse_ntt_pow_of_2(&mut extended_c, mod_q, root_unity);
    let c = reduce_with_prime_cyclotomic_polynomial(&extended_c, a.len());
    c
}

pub fn cyclic_mul_ntt_slow(a: &RingElement32, b: &RingElement32, mod_q: i128) -> Vec<i128>  {
    let root_unity = choose_root_unity(a.len(), mod_q).unwrap();;
    let a_ntt = ntt_slow(&a, root_unity, mod_q);
    let b_ntt = ntt_slow(&b, root_unity, mod_q);
    let mut c_ntt = hadamard(&a_ntt, &b_ntt);
    reduce_mod(&mut c_ntt, mod_q);
    let mut c = inverse_ntt_slow(&c_ntt, root_unity, mod_q);
    reduce_mod_imbalanced(&mut c, mod_q);
    c
}


// a is of a length p-1, where p is prime and represents a polynomial with of degree p-2 with in Z[x]/(\phi_p(x))
// trace is a sum of all automophism in Aut(Z[x]/(\phi_p(x)))
pub fn trace(a: &RingElement32) -> i128  {
    let mut res = 0;
    let p = a.len() + 1;

    for aut in 1..p {
        let mut applied_aut = vec![0; (p-2)*(p-1) + 1];
        for i in 0..a.len() {
            applied_aut[i * aut] += a[i];
        }
        let reduced = reduce_with_prime_cyclotomic_polynomial(&applied_aut, p);
        res += reduced[0];
    }
    res
}


pub fn conjugate(a: &RingElement32) -> Vec<i128> {
    let p = a.len() + 1;
    let mut applied_aut = vec![0; (p-2)*(p-1) + 1];
    for i in 0..a.len() {
        applied_aut[i * (p - 1)] += a[i];
    }
    let reduced = reduce_with_prime_cyclotomic_polynomial(&applied_aut, p);
    reduced
}

pub fn random(len: usize, mod_q: i128) -> Vec<i128> {
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(len);
    for _ in 0..len {
        let number = rng.gen_range(0..mod_q);
        result.push(number);
    }
    result
}







