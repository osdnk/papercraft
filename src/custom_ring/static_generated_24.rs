use crate::custom_ring::r#static::MOD_Q;

type GEN_TYPE = u64;
pub static CONDUCTOR_COMPOSITE: usize = 24;
pub const PHI: usize = 8;
pub const DEGREE: usize = 4;
pub static BASIS: [[GEN_TYPE; 4];8] = [
    [MOD_Q - 1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, MOD_Q - 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, MOD_Q - 1],
    [0, 0, MOD_Q - 1, 0],
    [0, 0, 0, 0],
];

pub static INV_BASIS: [[GEN_TYPE; 8];4] = [
    [MOD_Q - 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, MOD_Q - 1, 0, 0, 0, 0],
    [0, 0, MOD_Q - 1, 0, 0, 0, MOD_Q - 1, 0],
    [0, 0, 0, 0, 0, MOD_Q - 1, 0, 0],
];

pub static MIN_POLY: [GEN_TYPE; 9] = [
    1, 0, 0, 0, MOD_Q - 1, 0, 0, 0, 1, ];

pub static TWIST: [GEN_TYPE; 8] = [
    3, 0, 0, 0, 0, 0, 0, 0, ];

pub static CONVERSION_FACTOR: [GEN_TYPE; 21] = [
    8, 0, 0, 0, 4, 0, 0, 0, MOD_Q-4, 0, 0, 0, MOD_Q-8, 0, 0, 0, MOD_Q-4, 0, 0, 0, 4, ];

pub static CIRCULANT_QUOTIENT: [GEN_TYPE; 25] = [
    MOD_Q - 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, ];

pub static TRACE_COEFFS: [GEN_TYPE; 8] = [
    8, 0, 0, 0, 4, 0, 0, 0, ];

pub static V_COEFFS: [GEN_TYPE; 8] = [
    5496693261158, 1596781416804, 1671707017376, 1596781416804, 0, MOD_Q-2224954554742, MOD_Q-835853508688, 628173137938, ];

pub static V_INV_COEFFS: [GEN_TYPE; 8] = [
    579306122153, 2039643313149, 2112559483490, 2039643313149, 0, 1650632874806, 1692499327696, 1807281950927, ];

pub static CHALLENGE_SET: [[GEN_TYPE; 8];3] = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
];

pub static CONJUGATE_MAP: [[GEN_TYPE; 8];8] = [
    [1, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, MOD_Q - 1, 0, 0, 0],
    [0, 0, 0, MOD_Q - 1, 0, 0, 0, MOD_Q - 1],
    [0, 0, MOD_Q - 1, 0, 0, 0, MOD_Q - 1, 0],
    [0, MOD_Q - 1, 0, 0, 0, MOD_Q - 1, 0, 0],
];


