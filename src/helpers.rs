
macro_rules! println_with_timestamp {
    ($($arg:tt)*) => {
        // Get the current local time
        let now = chrono::Local::now();
        // Format the time however you like. Here we use "%Y-%m-%d %H:%M:%S"
        print!("[{}] ", now.format("%Y-%m-%d %H:%M:%S"));
        println!($($arg)*);
    };
}

// pub fn assert_balanced_ntt(a_ring_el: &DPrimeRingElement) {
//     for j in 1..NOF_PRIMES {
//         for i in 0..PHI {
//             assert!(
//                 a_ring_el.coeffs[j][i] == a_ring_el.coeffs[j-1][i] ||
//                 MOD_Q - a_ring_el.coeffs[j][i] == MODS_Q[j - 1] - a_ring_el.coeffs[j-1][i]
//             );
//
//         }
//     }
// }



pub(crate) use println_with_timestamp;
use crate::custom_ring::r#static::{MODS_Q, NOF_PRIMES, PHI};
use crate::custom_ring::ring::DPrimeRingElement;
