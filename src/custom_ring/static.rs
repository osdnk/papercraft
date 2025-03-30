use lazy_static::lazy_static;
use once_cell::sync::Lazy;
use rug::Integer;
use crate::custom_ring::ring::{DPrimeRingElement, Ring, RingElement};
use crate::slow_ntt::root_of_unity::mod_inverse;

// pub const PHI: usize = 6;
// pub const PHI: usize = 2038;

pub const n_fhe: usize = 723;
pub const PHI: usize = 8;
// pub const n_fhe: usize = 5;
pub const CONDUCTOR: usize = PHI + 1;
pub const LOG_CONDUCTOR: usize = 10;
pub const TWO_PHI_MINUS_ONE: usize = 2 * PHI - 1;
pub const TEST_PHI: usize = 6;
pub const TEST_TWO_PHI_MINUS_ONE: usize = 11;
pub static NOF_THREADS_UPPED_BOUND: usize = 120;
pub static MAX_THREADS: usize = 100;

pub const NOF_PRIMES: usize = 1;
// pub const MOD_Q_1: u64 = 563843306618881;
pub const MOD_Q_1: u64 = 2305841497385205761;
pub const MOD_Q_2: u64 = 2305839985556717569;
pub const MOD_Q_3: u64 = 2305836137266020353;

// pub const FHE_Q: u64 = 562194039177217;
pub const FHE_Q: u64 = MOD_Q;

cfg_if::cfg_if! {
    if #[cfg(feature = "a0")] {
        pub static MOD_Q: u64 = 4611686078556930049;
    } else if #[cfg(feature = "a1")] {
        pub static MOD_Q: u64 = 4611686078556930049;
    } else {
        pub static MOD_Q: u64 = 4611686019232694273;
    }
}

pub const MOD_Q_1_INVERSE: &str = "20465042156472605823678476990328894094661227";
pub const MOD_Q_2_INVERSE: &str = "116086187203291140600610033409056534146448479";
pub const MOD_Q_3_INVERSE: &str = "44105694256337425998556374770667470354165048";


pub const MODS_Q: [u64; NOF_PRIMES] = [MOD_Q];

pub static MODS_Q_INVERSE: Lazy<Vec<Integer>> = Lazy::new(|| {
    [MOD_Q_1_INVERSE, MOD_Q_2_INVERSE, MOD_Q_3_INVERSE]
        .iter()
        .map(|m| m.parse::<Integer>().unwrap())
        .collect::<Vec<Integer>>()
        .try_into()
        .unwrap()
});

pub static PRODUCT_Q: Lazy<Integer> = Lazy::new(|| {
    MODS_Q.iter().map(|&p| Integer::from(p)).product()
});

pub static ONE: Lazy<DPrimeRingElement> = Lazy::new(|| {
    Ring::constant(1)
});

pub static TWO: Lazy<DPrimeRingElement> = Lazy::new(|| {
    Ring::constant(2)
});

// pub static INVERSE_FHE_Q: Lazy<DPrimeRingElement> = Lazy::new(|| {
//     PrimeRing::new_from_larger_vec_rns(
//         &MODS_Q.iter()
//             .map(|m| {
//                 let mut coeff = vec![mod_inverse(FHE_Q as i128, (*m) as i128).unwrap() as u64];
//                 coeff.resize(PHI + 2, 0);
//                 coeff
//             })
//             .collect::<Vec<Vec<u64>>>()
//     )
// });
//
// pub static INVERSE_CONDUCTOR: Lazy<DPrimeRingElement> = Lazy::new(|| {
//     PrimeRing::new_from_larger_vec_rns(
//         &MODS_Q.iter()
//             .map(|m| {
//                 let mut coeff = vec![mod_inverse(CONDUCTOR as i128, (*m) as i128).unwrap() as u64];
//                 coeff.resize(PHI + 2, 0);
//                 coeff
//             })
//             .collect::<Vec<Vec<u64>>>()
//     )
// });

// pub static DUAL_BASIS: Lazy<Vec<DPrimeRingElement>> = Lazy::new(|| {
//     let mut coeffs_v = [0u64; CONDUCTOR + 1];
//     coeffs_v[1] = 1;
//     let v = PrimeRing::new_from_larger_vec(&coeffs_v.try_into().unwrap());
//
//     (0..PHI).into_iter().map(|i| {
//         let mut coeffs_t = [0u64; CONDUCTOR + 1];
//         coeffs_t[CONDUCTOR - i] = 1;
//         let t = PrimeRing::new_from_larger_vec(&coeffs_t.try_into().unwrap());
//         (t - v) * INVERSE_CONDUCTOR.clone()
//     }).collect()
// });

pub const BASIS: Lazy<Vec<DPrimeRingElement>> = Lazy::new(|| {
    (0..PHI).into_iter().map(|i| {
        let mut coeffs_t = [0u64; CONDUCTOR];
        coeffs_t[i] = 1;
        let t = Ring::new_from_larger_vec(&coeffs_t.try_into().unwrap());
        t
    }).collect()
});

pub static  LOG_Q: usize = 62;



pub static WIT_DIM: usize = 16384;
pub static REP: usize = 9;

cfg_if::cfg_if! {
    if #[cfg(feature = "c1")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 220;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 175104;
        pub static NOF_ROUNDS: i32 = 5;
        pub static RADICES: [u64; 9] = [0, 128, 512, 128, 128, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = true;
    } else if #[cfg(feature = "c2")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 216;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 87552;
        pub static NOF_ROUNDS: i32 = 5;
        pub static RADICES: [u64; 9] = [0, 256, 512, 128, 512, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = false;
    } else if #[cfg(feature = "c3")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 210;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 43776;
        pub static NOF_ROUNDS: i32 = 4;
        pub static RADICES: [u64; 9] = [0, 512, 512, 512, 0, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = true;
    } else if #[cfg(feature = "b1")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 180;
        pub static CHUNKS: usize = 48;
        pub static TIME: usize = 196608;
        pub static NOF_ROUNDS: i32 = 5;
        pub static RADICES: [u64; 9] = [0, 64, 256, 1024, 1024, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = true;
   } else if #[cfg(feature = "b2")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 177;
        pub static CHUNKS: usize = 48;
        pub static TIME: usize = 98304;
        pub static NOF_ROUNDS: i32 = 5;
        pub static RADICES: [u64; 9] = [0, 128, 512, 1024, 512, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = false;
    } else if #[cfg(feature = "b3")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 162;
        pub static CHUNKS: usize = 48;
        pub static TIME: usize = 49152;
        pub static NOF_ROUNDS: i32 = 4;
        pub static RADICES: [u64; 9] = [0, 128, 256, 512, 0, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = true;

    } else if #[cfg(feature = "a0")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 118;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 389120;
        pub static NOF_ROUNDS: i32 = 6;
        pub static RADICES: [u64; 9] = [0, 32, 128, 512, 1024, 1024,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = false;
    } else if #[cfg(feature = "a1")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 115;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 194560;
        pub static NOF_ROUNDS: i32 = 5;
        pub static RADICES: [u64; 9] = [0, 64, 256, 512, 1024, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = true;
    } else if #[cfg(feature = "a2")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 113;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 97280;
        pub static NOF_ROUNDS: i32 = 5;
        pub static RADICES: [u64; 9] = [0, 128, 512, 512, 512, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = false;
    } else if #[cfg(feature = "a3")] {
        pub static MODULE_SIZE: usize = 14;
        pub static COMMITMENT_MODULE_SIZE: usize = 103;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 48640;
        pub static NOF_ROUNDS: i32 = 4;
        pub static RADICES: [u64; 9] = [0, 256, 256, 512, 0, 0,0,0,0];
        pub static SHOULD_REPEAT_SP_IN_LAST: bool = true;

    } else {
        pub static MODULE_SIZE: usize = 2;
        pub static COMMITMENT_MODULE_SIZE: usize = MODULE_SIZE / 2;
        pub static CHUNKS: usize = 3;
        pub static TIME: usize = 256 * CHUNKS;
        pub static RADICES: [i32; 8] = [0, 0, 0, 0, 0, 0, 0, 0];
    }
}

pub static CHUNK_SIZE: usize = TIME / CHUNKS;

pub static RADIX: u64 = 16;
