#![feature(adt_const_params)]
#![feature(const_refs_to_static)]

use crate::arithmetic::call_sage_inverse_polynomial;
use crate::helpers::println_with_timestamp;
use crate::custom_ring::r#static::{MOD_Q, MODS_Q, MODULE_SIZE, PHI, TIME, CHUNKS, RADIX, COMMITMENT_MODULE_SIZE, RADICES};
use crate::custom_ring::ring::{Ring};
use crate::custom_ring::static_generated_24::CONDUCTOR_COMPOSITE;
use crate::protocol::protocol;

mod custom_ring;

mod slow_ntt;
mod arithmetic;
mod subroutines;
//
mod helpers;
mod protocol;
mod vdf;
// mod fhe_mock;
// mod translation_layer;

fn main() {
    cfg_if::cfg_if! {
        if #[cfg(feature = "c3")] {
                println_with_timestamp!("c3");
        } else if #[cfg(feature = "c2")] {
                println_with_timestamp!("c2");
        } else if #[cfg(feature = "c1")] {
                println_with_timestamp!("c1");
        } else if #[cfg(feature = "b3")] {
                println_with_timestamp!("b3");
        } else if #[cfg(feature = "b2")] {
                println_with_timestamp!("b2");
        } else if #[cfg(feature = "b1")] {
                println_with_timestamp!("b1");
        } else if #[cfg(feature = "a3")] {
                println_with_timestamp!("a3");
        } else if #[cfg(feature = "a2")] {
                println_with_timestamp!("a2");
        } else if #[cfg(feature = "a1")] {
                println_with_timestamp!("a1");
        } else if #[cfg(feature = "a0")] {
                println_with_timestamp!("a0");
        } else {
                println_with_timestamp!("default");
        }
    }
    println_with_timestamp!("PARAMS: MODULE: {:?}, COMMITMENT_MODULE: {:?}, TIME: {:?}, CHUNKS: {:?}, Q: {:?}, CONDUCTOR: {:?}, RADICES: {:?}", MODULE_SIZE, COMMITMENT_MODULE_SIZE, TIME, CHUNKS, MOD_Q, CONDUCTOR_COMPOSITE, RADICES);
    let a = Ring::random();
    let b = call_sage_inverse_polynomial(&a).unwrap();
    assert_eq!(a * b, Ring::constant(1));
    println_with_timestamp!("OK sage");
    protocol()
}
