use fast_modulo::{mod_u128u64_unchecked, mulmod_u64, powmod_u64};
use crate::arithmetic::reduce_mod_vec;
use crate::custom_ring::r#static::{MOD_Q, MOD_Q_1};
use crate::slow_ntt::root_of_unity::choose_root_unity;

pub fn ntt_pow_of_2_inner(a: &mut [u64], mod_q: u64, root_unity: u64) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let mut a_odd: Vec<_> = a.iter().step_by(2).cloned().collect();
    let mut a_even: Vec<_> = a.iter().skip(1).step_by(2).cloned().collect();

    if a.len() > 100 {
        rayon::scope(|s| {
            s.spawn(|_| {
                ntt_pow_of_2_inner(a_odd.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64, mod_q as u64));
            });
            s.spawn(|_| {
                ntt_pow_of_2_inner(a_even.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64, mod_q as u64));
            });
        });
    } else {
        ntt_pow_of_2_inner(a_odd.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64, mod_q as u64));
        ntt_pow_of_2_inner(a_even.as_mut_slice(), mod_q, mulmod_u64(root_unity as u64, root_unity as u64, mod_q as u64));
    }

    // reduce_mod_imbalanced(a_even, mod_q);


    for i in 0..n / 2 {
        a[i] = mod_u128u64_unchecked((a_odd[i] as u128 + mod_q as u128 + mulmod_u64(powmod_u64(root_unity, i as u64, mod_q), a_even[i], mod_q) as u128), mod_q);
        a[i + n / 2]  = mod_u128u64_unchecked((a_odd[i] as u128 + mod_q as u128 - mulmod_u64(powmod_u64(root_unity, i as u64, mod_q), a_even[i], mod_q) as u128), mod_q);
    }
    //
    reduce_mod_vec(a, mod_q);
}
pub fn inverse_ntt_pow_of_2_inner(a: &mut [u64], mod_q: u64, root_unity: u64) {
    let n = a.len();
    if n <= 1 {
        return;
    }

    let inv_n = powmod_u64(n as u64, mod_q - 2, mod_q);
    let inv_root_unity = powmod_u64(root_unity, mod_q - 2, mod_q);

    ntt_pow_of_2_inner(a, mod_q, inv_root_unity);

    for i in 0..n {
        a[i] = mulmod_u64(a[i], inv_n, mod_q);
    }

    reduce_mod_vec(a, mod_q);
}




pub fn ntt_pow_of_2(a: &mut [u64], mod_q: u64) {
    let root = choose_root_unity(a.len(), mod_q as i128).unwrap() as u64;
    ntt_pow_of_2_inner(a, mod_q, root);
}

pub fn inverse_ntt_pow_of_2(a: &mut [u64], mod_q: u64) {
    let root = choose_root_unity(a.len(), mod_q as i128).unwrap() as u64;
    inverse_ntt_pow_of_2_inner(a, mod_q, root);
}

#[test]
fn test_slow_ntt_1() {
    let mut a = [1,2,3,4];
    let q = MOD_Q_1;
    ntt_pow_of_2(&mut a, q);
    println!("{:?}", a);
    inverse_ntt_pow_of_2(&mut a, q);
    assert_eq!(a, [1,2,3,4]);
}

#[test]
fn test_slow_ntt_2() {
    let mut a = [1,0,0,0];
    let q = 13;
    ntt_pow_of_2(&mut a, q);
    assert_eq!(a, [1,1,1,1]);
}

#[test]
fn test_slow_ntt_3() {
    let mut a = [2,3,0,0];
    let q = MOD_Q;
    ntt_pow_of_2(&mut a, q);
    inverse_ntt_pow_of_2(&mut a, q);

    assert_eq!(a, [2, 3, 0, 0]);
}

