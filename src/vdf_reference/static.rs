
cfg_if::cfg_if! {
    if #[cfg(feature = "c1")] {
        pub static MODULE_SIZE: usize = 42;
        pub static COMMITMENT_MODULE_SIZE: usize = 203;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 80256;
        pub static SKIP_OPENER: bool = false;
    } else if #[cfg(feature = "c2")] {
        pub static MODULE_SIZE: usize = 40;
        pub static COMMITMENT_MODULE_SIZE: usize = 149;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 40128;
        pub static SKIP_OPENER: bool = true;
    } else if #[cfg(feature = "c3")] {
        pub static MODULE_SIZE: usize = 37;
        pub static COMMITMENT_MODULE_SIZE: usize = 145;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 20064;
        pub static SKIP_OPENER: bool = true;
    } else if #[cfg(feature = "b1")] {
        pub static MODULE_SIZE: usize = 32;
        pub static COMMITMENT_MODULE_SIZE: usize = 150;
        pub static CHUNKS: usize = 48;
        pub static TIME: usize = 76800;
        pub static SKIP_OPENER: bool = false;
   } else if #[cfg(feature = "b2")] {
        pub static MODULE_SIZE: usize = 30;
        pub static COMMITMENT_MODULE_SIZE: usize = 110;
        pub static CHUNKS: usize = 48;
        pub static TIME: usize = 38400;
        pub static SKIP_OPENER: bool = true;
    } else if #[cfg(feature = "b3")] {
        pub static MODULE_SIZE: usize = 28;
        pub static COMMITMENT_MODULE_SIZE: usize = 108;
        pub static CHUNKS: usize = 48;
        pub static TIME: usize = 18432;
        pub static SKIP_OPENER: bool = true;
    } else if #[cfg(feature = "a0")] {
        pub static MODULE_SIZE: usize = 23;
        pub static COMMITMENT_MODULE_SIZE: usize = 109;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 155648;
        pub static SKIP_OPENER: bool = false;
    } else if #[cfg(feature = "a1")] {
        pub static MODULE_SIZE: usize = 21;
        pub static COMMITMENT_MODULE_SIZE: usize = 96;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 77824;
        pub static SKIP_OPENER: bool = false;
    } else if #[cfg(feature = "a2")] {
        pub static MODULE_SIZE: usize = 20;
        pub static COMMITMENT_MODULE_SIZE: usize = 71;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 38912;
        pub static SKIP_OPENER: bool = true;
    } else if #[cfg(feature = "a3")] {
        pub static MODULE_SIZE: usize = 19;
        pub static COMMITMENT_MODULE_SIZE: usize = 69;
        pub static CHUNKS: usize = 38;
        pub static TIME: usize = 19456;
        pub static SKIP_OPENER: bool = true;
    } else {
        pub static MODULE_SIZE: usize = 416;
        pub static COMMITMENT_MODULE_SIZE: usize = MODULE_SIZE / 2;
        pub static CHUNKS: usize = 57;
        pub static TIME: usize = 7296;
        pub static SKIP_OPENER: bool = false;
    }
}



pub static CONDUCTOR: usize = 257;
cfg_if::cfg_if! {
    if #[cfg(feature = "a0")] {
        pub static MOD_Q: i128 = 4611686019232694273;
    } else {
        pub static MOD_Q: i128 = 4611686078556930049;
    }
}
pub static BIG_MOD_Q: &str = "1324325423464534264434434342342342342345325346352367564534123546753";
pub static  LOG_Q: usize = 62;
pub static  LOG_BIG_Q: usize = 220;
pub static N_DIM: usize = 10;

pub type BASE_INT = i128;

// pub static MODULE_SIZE: usize = 4;
// pub static MODULE_SIZE: usize = 2;


pub static RADIX: BASE_INT = 21;


pub static CHUNK_SIZE: usize = TIME / CHUNKS;




// MOD_1 > N LOG_Q * Conductor^2
// pub(crate) static MOD_1: i128 = 69206017;
// 2^M | MOD - 1
// either MOD_1 = MOD_Q or MOD_Q so large that no overflow happens
pub(crate) static MOD_1: i128 = MOD_Q;


