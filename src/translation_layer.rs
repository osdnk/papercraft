use num_traits::{One, Zero};
use rayon::iter::IntoParallelRefIterator;
use crate::arithmetic::{decompose_by_radix, decompose_matrix_by_radix, random, compose_with_radix, ring_inner_product, sub_mod, reduce_mod, decompose_matrix_by_chunks, decompose_by_chunks, transpose, sample_random_mat};
use crate::prime_ring::r#static::{CONDUCTOR, LOG_CONDUCTOR, MOD_Q, PHI};
use crate::prime_ring::ring::{DPrimeRingElement, PrimeRing, PrimeRingElement};

pub fn prepare_translation_layer() {
    //TEST
    let q = (2^24) as u64;
    let b = PrimeRing::constant(
        random(1, q)[0]
    );

    let v = PrimeRing::new(
        random(PHI, q).try_into().unwrap()
    );
    let n = 585;
    let ell_bs = 2;
    let ell_ks = 5;
    let beta_bs = 2u64.pow(8);
    let beta_ks = 2u64.pow(2);

    let mut C = (0..n).into_iter().map(|_| {
        (0..2).map(|_| {
            (0..2 * ell_bs).map(|_| PrimeRing::random()).collect()
        }).collect()
    }).collect::<Vec<Vec<Vec<DPrimeRingElement>>>>();

    let b_hat = PrimeRing::constant(
        (b.coeffs[0] as f64  / q as f64 * CONDUCTOR as f64)
            .floor() as u64
    );
    println!("{:?}", b_hat);
    let z_b_hat = PrimeRing::constant(
        sub_mod(
            b_hat.coeffs[0] * q,
            b.coeffs[0] * CONDUCTOR as u64,
            MOD_Q
        )
    );
    let b_to_b_hat = vec![
        PrimeRing::constant(CONDUCTOR as u64),
        PrimeRing::constant(MOD_Q - q),
        PrimeRing::constant(1u64),
    ];

    let witness = vec![b, b_hat, z_b_hat];

    assert_eq!(ring_inner_product(&b_to_b_hat, &witness), PrimeRingElement::zero());

    let mut b_hat_decomposed = decompose_by_chunks(&b_hat, LOG_CONDUCTOR);
    println!("{:?}", b_hat);

    b_hat_decomposed.resize(LOG_CONDUCTOR, PrimeRingElement::zero());

    println!("{:?}", b_hat_decomposed);

    assert_eq!(
        compose_with_radix(
            &vec![b_hat_decomposed.clone()],
            2, b_hat_decomposed.len()
        )[0][0],
        b_hat
    );


    let zeta_to_b_hat_decomposed =
        b_hat_decomposed
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let mut coeffs = [0u64; CONDUCTOR];
                if b.coeffs[0] == 1 {
                    coeffs[2usize.pow(i as u32)] = 1;
                } else {
                    coeffs[0] = 1;
                }
                PrimeRing::new_from_larger_vec(&coeffs.to_vec())
            })
            .collect::<Vec<DPrimeRingElement>>();

    println!("{:?}", zeta_to_b_hat_decomposed);




    let zeta_to_b_hat = (||{
        let mut coeffs = [0u64; CONDUCTOR];
        coeffs[b_hat.coeffs[0] as usize] = 1;
        PrimeRing::new_from_larger_vec(&coeffs.to_vec())
    })();

    let zeta_to_b_hat_conj = zeta_to_b_hat.conjugate();
    let mut t_0 = Vec::with_capacity(n + 1);
    let mut t_1 = Vec::with_capacity(n + 1);

    t_0.push(PrimeRing::constant(0));
    t_1.push(zeta_to_b_hat_conj * v);




    assert_eq!((|| {
        let mut prod = PrimeRing::constant(1);
        zeta_to_b_hat_decomposed
            .iter()
            .for_each(|e| { prod = prod * *e });
        prod
    })(), zeta_to_b_hat);



    let a = (0.. n).into_iter()
        .map(|_| {
            PrimeRing::constant(
                random(1, q)[0]
            )
        }).collect::<Vec<DPrimeRingElement>>();

    let a_hat = a.iter().map(|ai| {
        PrimeRing::constant(
            (ai.coeffs[0] as f64  / q as f64 * CONDUCTOR as f64)
                .floor() as u64
        )
    }).collect::<Vec<DPrimeRingElement>>();

    let z_a_hat = a.iter().zip(a_hat.iter()).map(|(ai, a_hati)|{
        PrimeRing::constant(
            sub_mod(
                a_hati.coeffs[0] * q,
                ai.coeffs[0] * CONDUCTOR as u64,
                MOD_Q
            )
        )
    }).collect::<Vec<DPrimeRingElement>>();

    let (a_hat_decomposed, _) =
        decompose_matrix_by_chunks(&transpose(&vec![a_hat.clone()]), LOG_CONDUCTOR);

    println!("A_HAT, {:?}", a_hat);
    println!("A_HAT_DEC, {:?}", a_hat_decomposed);
    let zeta_to_a_hat_decomposed =
        a_hat_decomposed
            .iter()
            .map(|a_hat_decomposed_el| {
                a_hat_decomposed_el
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| {
                        let mut coeffs = [0u64; CONDUCTOR];
                        if b.coeffs[0] == 1 {
                            coeffs[2usize.pow(i as u32)] = 1;
                        } else {
                            coeffs[0] = 1;
                        }
                        PrimeRing::new_from_larger_vec(&coeffs.to_vec())
                    })
                    .collect()
            }).collect::<Vec<Vec<DPrimeRingElement>>>();

    let zeta_to_a_hat = a_hat
        .iter()
        .map(|a_hati| {
            let mut coeffs = [0u64; CONDUCTOR];
            coeffs[a_hati.coeffs[0] as usize] = 1;
            PrimeRing::new_from_larger_vec(&coeffs.to_vec())
        })
        .collect::<Vec<DPrimeRingElement>>();

    zeta_to_a_hat.iter().zip(zeta_to_a_hat_decomposed.iter()).for_each(
      |(zeta_to_a_hati, zeta_to_a_hati_decomposed)| {
          assert_eq!((|| {
              let mut prod = PrimeRing::constant(1);
              zeta_to_a_hati_decomposed
                  .iter()
                  .for_each(|e| { prod = prod * *e });
              prod
          })(), *zeta_to_a_hati);
      }
    );

    let zeta_to_a_hat_minus_one = zeta_to_a_hat.iter().map(|el| {
        *el - PrimeRing::constant(1)
    }).collect::<Vec<DPrimeRingElement>>();

    let mut m_0 = Vec::with_capacity(n);
    let mut m_0_ddot = Vec::with_capacity(n);
    let mut m_1 = Vec::with_capacity(n);
    let mut m_1_ddot = Vec::with_capacity(n);

    for i in 0..n {
        m_0.push(zeta_to_a_hat_minus_one[i] * *t_0.last().unwrap());
        m_1.push(zeta_to_a_hat_minus_one[i] * *t_1.last().unwrap());
        m_0_ddot.push(decompose_by_radix(&m_0.last().unwrap(), beta_bs));
        m_1_ddot.push(decompose_by_radix(&m_1.last().unwrap(), beta_bs));
        // let s_m_0_ddot = m_0_ddot
        //     .last()
        //     .unwrap()
        //     .iter()
        //     .rev()
        //     .take(ell_bs)
        //     .collect::<Vec<DPrimeRingElement>>();
        //
        // let s_m_1_ddot = m_1_ddot
        //     .last()
        //     .unwrap()
        //     .iter()
        //     .rev()
        //     .take(ell_bs)
        //     .collect::<Vec<DPrimeRingElement>>();
        // let C_i_0 = &C[i][0];
        // let C_i_1 = &C[i][1];





    }


}

#[test]
fn test_prepare_translation_layer() {
    prepare_translation_layer();
}
