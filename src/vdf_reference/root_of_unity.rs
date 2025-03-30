use crate::r#static::{BASE_INT, LOG_BIG_Q, LOG_Q, MOD_1, MOD_Q, N_DIM};

/// Extended Euclidean Algorithm
///
/// Given two integers `a` and `b`, returns a tuple `(g, x, y)` where `g` is
/// the greatest common divisor (gcd) of `a` and `b`, and `x` and `y` are
/// integers such that `a * x + b * y = g`.
///
/// # Arguments
///
/// * `a` - An integer.
/// * `b` - An integer.
///
/// # Returns
///
/// A tuple `(g, x, y)` such that `g` is the gcd of `a` and `b`, and
/// `a * x + b * y = g`.
fn extended_gcd(a: BASE_INT, b: BASE_INT) -> (BASE_INT, BASE_INT, BASE_INT) {
    if a == 0 {
        (b, 0, 1)
    } else {
        let (g, x, y) = extended_gcd(b % a, a);
        (g, y - (b / a) * x, x)
    }
}

/// Modular Inverse
///
/// Computes the modular inverse of `a` modulo `m` using the Extended Euclidean Algorithm.
/// The modular inverse of `a` is an integer `x` such that `(a * x) % m == 1`.
///
/// # Arguments
///
/// * `a` - An integer.
/// * `m` - The modulus.
///
/// # Returns
///
/// An `Option<BASE_INT>`, which is `Some(x)` if the inverse exists, and `None` otherwise.
fn mod_inverse(a: BASE_INT, m: BASE_INT) -> Option<BASE_INT> {
    let (g, x, _) = extended_gcd(a, m);
    if g != 1 {
        None
    } else {
        Some((x % m + m) % m)
    }
}

/// Modular Exponentiation
///
/// Computes `(base^exp) % modulus` efficiently using the method of exponentiation by squaring.
///
/// # Arguments
///
/// * `base` - The base integer.
/// * `exp` - The exponent integer.
/// * `modulus` - The modulus.
///
/// # Returns
///
/// `base` raised to the power `exp` modulo `modulus`.
fn mod_pow(base: BASE_INT, exp: BASE_INT, modulus: BASE_INT) -> BASE_INT {
    let mut base = base % modulus;
    let mut result = 1;
    let mut exp = exp;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result * base) % modulus;
        }
        exp >>= 1;
        base = (base * base) % modulus;
    }
    result
}

/// Chooses a Root of Unity
///
/// Chooses a primitive `n`-th root of unity modulo `mod_q`.
///
/// # Arguments
///
/// * `n` - The order of the root of unity.
/// * `mod_q` - The modulus.
///
/// # Returns
///
/// An `Option<BASE_INT>`, which is `Some(root)` if the root exists, and `None` otherwise.
pub fn choose_root_unity(n: usize, mod_q: BASE_INT) -> Option<BASE_INT> {
    // Ensure `n` divides `mod_q - 1`
    if !(n > 0 && (mod_q - 1) % (n as BASE_INT) == 0) {
        return None;
    }

    let a: BASE_INT = 2;
    let t = 16;
    let inverse_n = mod_inverse((n * t) as BASE_INT, mod_q);
    match inverse_n {
        None => None,
        Some(inverse_n) => {
            let root = mod_pow(a, ((mod_q - 1) / ((n * t) as BASE_INT)) * inverse_n % (mod_q - 1), mod_q);
            Some(root)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_root_unity_valid() {
        let q = 17;
        let n = 4;
        let root = choose_root_unity(n, q).unwrap();
        assert_eq!(mod_pow(root, n as BASE_INT, q), 1);
    }

    #[test]
    fn test_choose_root_unity_valid_large() {
        let q = 257;
        let n = 8;
        let root = choose_root_unity(n, q).unwrap();
        assert_eq!(mod_pow(root, n as BASE_INT, q), 1);
    }

    #[test]
    fn test_choose_root_unity_valid_very_large() {
        let q = MOD_1;
        let n = 2_usize.pow(28);
        let root = choose_root_unity(n, q).unwrap();
        assert_eq!(mod_pow(root, n as BASE_INT, q), 1);
        assert_ne!(mod_pow(root, (n / 2) as BASE_INT, q), 1);
    }

    #[test]
    fn test_choose_root_unity_invalid_n() {
        let q = 17;
        let n_invalid = 3;
        assert_eq!(choose_root_unity(n_invalid, q), None);
    }

    #[test]
    fn test_mod_inverse_valid() {
        assert_eq!(mod_inverse(3, 11), Some(4)); // 3 * 4 ≡ 1 (mod 11)
        assert_eq!(mod_inverse(10, 17), Some(12)); // 10 * 12 ≡ 1 (mod 17)
        assert_eq!(mod_inverse(5, 13), Some(8)); // 5 * 8 ≡ 1 (mod 13)
    }
}
