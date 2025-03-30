use std::iter::Sum;
use rayon::prelude::*;
use std::ops::{Mul, Add, Sub};
use std::process::Command;
use std::sync::Mutex;
use num_traits::{One, Zero};
use rand::Rng;
use crate::poly_arithmetic_i128::{hadamard, inverse_ntt_pow_of_2, ntt_pow_of_2, reduce_mod, reduce_quotient, RingElement32};
use crate::r#static::{BASE_INT, MOD_1, MOD_Q};
use crate::ring_helpers::transpose;
use crate::ring_i128::{Ring, ring_inner_product, RingElement};
use crate::static_rings::static_generated::{CHALLENGE_SET, CONDUCTOR_COMPOSITE, MIN_POLY, PHI};

/// Computes the dot product of each row of a matrix with a vector in parallel.
///
/// # Arguments
///
/// * `matrix` - A reference to a 2D vector (vector of vectors), where each element is a row of the matrix.
/// * `vector` - A reference to a 1D vector representing the vector to multiply with each row of the matrix.
///
/// # Returns
///
/// A vector containing the results of the dot products of each row of the matrix with the given vector.
///
/// # Panics
///
/// This function will panic if any row of the matrix does not have the same number of columns as the length of the vector.
pub fn parallel_dot_matrix_vector<T>(matrix: &[Vec<T>], vector: &[T]) -> Vec<T>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
{
    let nrows = matrix.len();
    // Ensure the matrix and vector dimensions are compatible for multiplication
    if nrows > 0 {
        assert_eq!(matrix[0].len(), vector.len(), "Matrix columns must be equal to vector length");
    }

    // Allocate output vector, initially default values
    let mut result = vec![T::zero(); nrows];

    // Parallel iteration over the rows of the matrix
    result
        .par_iter_mut()  // Parallel mutable iterator
        .enumerate()
        .for_each(|(i, res)| {
            // Obtaining a row slice from the matrix
            let row = &matrix[i];
            // Computing the dot product of the row and the vector
            let mut sum = T::zero();
            // Compute the dot product manually
            for (a, b) in row.iter().zip(vector.iter()) {
                // TODO fix no cloning
                let product = a.clone() * b.clone();
                sum = sum + product;
            }
            *res = sum
        });

    result
}

/// Computes the dot product of a vector and a matrix in parallel.
///
/// # Arguments
///
/// * `matrix` - A reference to a matrix represented as a slice of vectors.
/// * `vector` - A reference to a vector.
///
/// # Returns
///
/// A new vector containing the result of the vector-matrix multiplication.
///
/// # Type Parameters
///
/// * `T` - The type of the elements in the matrix and the vector. It must implement the `Mul`, `Zero`, `Copy`, `Send`, `Sync`,
/// and `Add` traits.
///
/// # Panics
///
/// This function will panic if the number of rows in the matrix does not match the length of the vector.
///
/// # Examples
///
/// ```
/// # fn main() {
/// let matrix = vec![
///     vec![1, 2],
///     vec![3, 4],
///     vec![5, 6],
/// ];
/// let vector = vec![7, 8, 9];
/// let result = parallel_dot_vector_matrix(&matrix, &vector);
/// assert_eq!(result, vec![76, 100]);
/// # }
/// ```
pub fn parallel_dot_vector_matrix<T>(vector: &[T], matrix: &[Vec<T>]) -> Vec<T>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
{
    assert!(
        matrix.len() == vector.len(),
        "Number of rows in the matrix must match the length of the vector"
    );

    (0..matrix[0].len())
        .into_par_iter()
        .map(|col| {
            matrix
                .iter()
                .zip(vector.iter())
                .map(|(row, &v)| row[col] * v)
                .fold(T::zero(), |acc, x| acc + x)
        })
        .collect()
}

#[test]
fn test_parallel_dot_vector_matrix_integers() {
    let matrix = vec![
        vec![1, 2],
        vec![3, 4],
        vec![5, 6],
    ];
    let vector = vec![7, 8, 9];
    let result = parallel_dot_vector_matrix(&vector, &matrix);
    assert_eq!(result, vec![76, 100]);
}


/// Multiplies each element in the given vector by a given scalar.
///
/// # Arguments
///
/// * `vector` - A reference to a vector of elements to be multiplied.
/// * `ell` - A reference to a scalar value by which each element of the vector will be multiplied.
///
/// # Returns
///
/// A new vector containing the result of element-wise multiplication.
///
/// # Type Parameters
///
/// * `T` - The type of elements in the vector and the scalar. It must implement the `Mul`, `Zero`, `Copy`, `Send`, `Sync`,
/// and `Add` traits.
///
/// # Examples
///
/// ```
/// # fn main() {
/// let vector = vec![1, 2, 3, 4];
/// let scalar = 2;
/// let result = vector_element_product(&vector, &scalar);
/// assert_eq!(result, vec![2, 4, 6, 8]);
/// # }
/// ```
pub fn vector_element_product<T>(vector: &Vec<T>, ell: &T) -> Vec<T>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
{
    vector.par_iter().map(|x| *x * *ell).collect()
}

#[test]
fn test_vector_element_product_integers() {
    let vector = vec![1, 2, 3, 4];
    let scalar = 2;
    let result = vector_element_product(&vector, &scalar);
    assert_eq!(result, vec![2, 4, 6, 8]);
}

#[test]
fn test_vector_element_product_floats() {
    let vector = vec![1.0, 2.0, 3.0, 4.0];
    let scalar = 0.5;
    let result = vector_element_product(&vector, &scalar);
    assert_eq!(result, vec![0.5, 1.0, 1.5, 2.0]);
}

#[test]
fn test_vector_element_product_zeros() {
    let vector = vec![0, 0, 0];
    let scalar = 999;
    let result = vector_element_product(&vector, &scalar);
    assert_eq!(result, vec![0, 0, 0]);
}

/// Computes the dot product of two matrices in parallel.
///
/// # Arguments
///
/// * `matrix_a` - A reference to a slice of vectors representing the first matrix.
/// * `matrix_b` - A reference to a slice of vectors representing the second matrix.
///
/// # Returns
///
/// A matrix (vector of vectors) containing the result of the dot product.
pub fn parallel_dot_matrix_matrix<T>(matrix_a: &[Vec<T>], matrix_b: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
{
    let nrows = matrix_a.len();
    let ncols = matrix_b[0].len();
    let inner_dim = matrix_b.len();

    // Ensure the matrices dimensions are compatible for multiplication
    assert!(matrix_a.first().map_or(true, |row| row.len() == inner_dim));

    // Transpose the second matrix for easier column access
    let matrix_b_t: Vec<Vec<T>> = transpose(&matrix_b.to_vec());

    // Allocate the resulting matrix
    let mut result = vec![vec![T::zero(); ncols]; nrows];

    // Parallel iteration over the rows of the first matrix
    result.par_iter_mut().enumerate().for_each(|(i, res_row)| {
        let row_a = &matrix_a[i];

        // Iterate over the columns of the second matrix (transposed)
        for j in 0..ncols {
            let col_b = &matrix_b_t[j];
            // Compute the dot product for element (i, j)
            // res_row[j] = inner_product(&row_a, &col_b);
            res_row[j] = row_a.iter().zip(col_b.iter()).fold(T::zero(), |acc, (&a, &b)| acc + a * b);
        }
    });

    result
}

pub fn ring_very_parallel_dot_matrix_matrix(matrix_a: &[Vec<RingElement>], matrix_b: &[Vec<RingElement>]) -> Vec<Vec<RingElement>>
{
    let nrows = matrix_a.len();
    let ncols = matrix_b[0].len();
    let inner_dim = matrix_b.len();

    // Ensure the matrices dimensions are compatible for multiplication
    assert!(matrix_a.first().map_or(true, |row| row.len() == inner_dim));

    // Transpose the second matrix for easier column access
    let matrix_b_t: Vec<Vec<RingElement>> = transpose(&matrix_b.to_vec());

    // Allocate the resulting matrix
    let mut result = vec![vec![RingElement::zero(); ncols]; nrows];

    // Parallel iteration over the rows of the first matrix
    result.par_iter_mut().enumerate().for_each(|(i, res_row)| {
        let row_a = &matrix_a[i];

        // Iterate over the columns of the second matrix (transposed)
        for j in 0..ncols {
            let col_b = &matrix_b_t[j];
            // Compute the dot product for element (i, j)
            res_row[j] = ring_inner_product(&row_a, &col_b);
        }
    });

    result
}



#[test]
fn test_parallel_dot_matrix_matrix() {
    let matrix_a = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let matrix_b = vec![
        vec![1, 4, 7],
        vec![2, 5, 8],
        vec![3, 6, 9],
    ];
    let result = parallel_dot_matrix_matrix(&matrix_a, &matrix_b);
    let expected = vec![
        vec![14, 32, 50],
        vec![32, 77, 122],
        vec![50, 122, 194],
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_parallel_dot() {
    let matrix: Vec<Vec<i32>> = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let vector: Vec<i32> = vec![1, 2, 3];
    let result = parallel_dot_matrix_vector(&matrix, &vector);
    assert_eq!(result, vec![14, 32, 50]);
}

pub fn parallel_dot_matrix_matrix_many_columns<T>(matrix_a: &Vec<Vec<T>>, matrix_b: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Mul<Output = T> + Zero + Copy + Send + Sync + Add<Output = T>,
{
    let ta = transpose(&matrix_a);
    let tb = transpose(&matrix_b);
    let res = parallel_dot_matrix_matrix(&tb, &ta);
    transpose(&res)
}


/// Samples a random matrix of size n x m where each element is a random RingElement.
///
/// # Arguments
///
/// * `n` - The number of rows in the matrix.
/// * `m` - The number of columns in the matrix.
///
/// # Returns
///
/// A vector of vectors (matrix) where each element is a random RingElement.
pub fn sample_random_mat(n: usize, m: usize) -> Vec<Vec<RingElement>> {
    // Create a matrix of size n x m
    (0..n).map(|_| {
        (0..m).map(|_| Ring::random()).collect()
    }).collect()
}


/// Samples a random bin matrix of size n x m where each element is a random RingElement.
///
/// # Arguments
///
/// * `n` - The number of rows in the matrix.
/// * `m` - The number of columns in the matrix.
///
/// # Returns
///
/// A vector of vectors (matrix) where each element is a random binary RingElement.
pub fn sample_random_bin_mat(n: usize, m: usize) -> Vec<Vec<RingElement>> {
    // Create a matrix of size n x m
    (0..n).map(|_| {
        (0..m).map(|_| Ring::random_bin()).collect()
    }).collect()
}

pub fn sample_random_constant_bin_mat(n: usize, m: usize) -> Vec<Vec<RingElement>> {
    // Create a matrix of size n x m
    (0..n).map(|_| {
        (0..m).map(|_| Ring::random_constant_bin()).collect()
    }).collect()
}


pub fn sample_random_constant_ss_mat(n: usize, m: usize) -> Vec<Vec<RingElement>> {
    // Create a matrix of size n x m
    let mut rng = rand::thread_rng();
    (0..n).map(|_| {
        (0..m).map(|_| {
            let number = rng.gen_range(0..CHALLENGE_SET.len());
            RingElement {
                coeffs: CHALLENGE_SET[number].clone()
            }
        }).collect()
    }).collect()
}


pub fn sample_random_bin_vec(n: usize) -> Vec<RingElement> {
    // Create a matrix of size n x m
    (0..n).map(|_| Ring::random_bin()).collect()
}


/// Returns a zero matrix of size n x m where each element is a zero RingElement.
///
/// # Arguments
///
/// * `n` - The number of rows in the matrix.
/// * `m` - The number of columns in the matrix.
///
/// # Returns
///
/// A vector of vectors (matrix) where each element is a zero RingElement.
pub fn zero_mat(n: usize, m: usize) -> Vec<Vec<RingElement>> {
    // Create a matrix of size n x m
    (0..n).map(|_| {
        (0..m).map(|_| Ring::zero()).collect()
    }).collect()
}


/// Samples a random vector of the given size where each element is a random RingElement.
///
/// # Arguments
///
/// * `size` - The number of elements in the vector.
///
/// # Returns
///
/// A vector where each element is a random RingElement.
pub fn sample_random_vector(size: usize) -> Vec<RingElement> {
    // Create a vector of the given size with random RingElement values
    (0..size).map(|_| Ring::random()).collect()
}

pub fn sample_random_vector_non_real(size: usize) -> Vec<RingElement> {
    // Create a vector of the given size with random RingElement values
    (0..size).map(|_| Ring::random_non_real()).collect()
}

/// Produces a zero of the given size where each element is a zero RingElement.
///
/// # Arguments
///
/// * `size` - The number of elements in the vector.
///
/// # Returns
///
/// A vector where each element is a zero RingElement.
pub fn zero_vector(size: usize) -> Vec<RingElement> {
    // Create a vector of the given size with random RingElement values
    (0..size).map(|_| Ring::zero()).collect()
}

/// Multiplies two polynomials using the Karatsuba algorithm.
///
/// # Arguments
///
/// * `a` - A slice representing the coefficients of the first polynomial.
/// * `b` - A slice representing the coefficients of the second polynomial.
/// * `mod_q` - An optional modulus for the coefficients.
///
/// # Returns
///
/// A vector representing the coefficients of the resulting polynomial.
///
/// # Panics
///
/// This function will panic if the input slices are not of the same length.
pub fn karatsuba_mul(a: &[BASE_INT], b: &[BASE_INT], mod_q: Option<BASE_INT>) -> Vec<BASE_INT> {
    if a.len() != b.len() {
        panic!("Karatsuba algorithm accepts arrays of the same length! {:?} {:?}", a.len(), b.len());
    }

    let n = a.len();
    if n <= 30 {
        return polynomial_mul(a, b, mod_q);
    }

    let mid = n / 2;

    let low_a = &a[..mid];
    let high_a = &a[mid..];
    let low_b = &b[..mid];
    let high_b = &b[mid..];

    let z0 = karatsuba_mul(low_a, low_b, mod_q);      // z0 = low_a * low_b
    let z2 = karatsuba_mul(high_a, high_b, mod_q);    // z2 = high_a * high_b

    let mut sum_a: Vec<BASE_INT> = low_a.iter().copied().zip(high_a.iter().copied()).map(|(x, y)| x + y).collect();
    let mut sum_b: Vec<BASE_INT> = low_b.iter().copied().zip(high_b.iter().copied()).map(|(x, y)| x + y).collect();

    if let Some(q) = mod_q {
        reduce_mod(&mut sum_a, q);
        reduce_mod(&mut sum_b, q);
    }

    let z1 = karatsuba_mul(&sum_a, &sum_b, mod_q);    // z1 = (low_a + high_a) * (low_b + high_b)

    let z1: Vec<BASE_INT> = z1.iter().copied().zip(z0.iter().copied().zip(z2.iter().copied()))
        .map(|(z1_val, (z0_val, z2_val))| z1_val - z0_val - z2_val)
        .collect();

    let mut res = vec![BASE_INT::zero(); 2 * n - 1];

    for (i, &val) in z0.iter().enumerate() {
        res[i] = res[i] + val;
    }

    for (i, &val) in z1.iter().enumerate() {
        res[i + mid] = res[i + mid] + val;
    }

    for (i, &val) in z2.iter().enumerate() {
        res[i + 2 * mid] = res[i + 2 * mid] + val;
    }

    if mod_q.is_some() {
        reduce_mod(&mut res, mod_q.unwrap());
    }
    res
}

pub fn karatsuba_mul_par(a: &[BASE_INT], b: &[BASE_INT], mod_q: Option<BASE_INT>) -> Vec<BASE_INT> {
    if a.len() != b.len() {
        panic!("Karatsuba algorithm accepts arrays of the same length! {:?} {:?}", a.len(), b.len());
    }

    let n = a.len();
    if n <= 60 {
        return karatsuba_mul(a, b, mod_q);
    }

    let mid = n / 2;

    let (low_a, high_a) = a.split_at(mid);
    let (low_b, high_b) = b.split_at(mid);

    // Parallel computation of z0 and z2
    let (z0, z2) = rayon::join(
        || karatsuba_mul_par(low_a, low_b, mod_q),
        || karatsuba_mul_par(high_a, high_b, mod_q),
    );

    let mut sum_a: Vec<BASE_INT> = low_a.iter().copied().zip(high_a.iter().copied()).map(|(x, y)| x + y).collect();
    let mut sum_b: Vec<BASE_INT> = low_b.iter().copied().zip(high_b.iter().copied()).map(|(x, y)| x + y).collect();

    if let Some(q) = mod_q {
        reduce_mod(&mut sum_a, q);
        reduce_mod(&mut sum_b, q);
    }

    let z1 = karatsuba_mul_par(&sum_a, &sum_b, mod_q);  // z1 = (low_a + high_a) * (low_b + high_b)

    let z1: Vec<BASE_INT> = z1.iter().copied().zip(z0.iter().copied().zip(z2.iter().copied()))
        .map(|(z1_val, (z0_val, z2_val))| z1_val - z0_val - z2_val)
        .collect();

    let mut res = vec![BASE_INT::zero(); 2 * n - 1];

    for (i, &val) in z0.iter().enumerate() {
        res[i] = res[i] + val;
    }

    for (i, &val) in z1.iter().enumerate() {
        res[i + mid] = res[i + mid] + val;
    }

    for (i, &val) in z2.iter().enumerate() {
        res[i + 2 * mid] = res[i + 2 * mid] + val;
    }

    if mod_q.is_some() {
        reduce_mod(&mut res, mod_q.unwrap());
    }
    res
}

/// Multiplies two polynomials using the naive method.
///
/// # Arguments
///
/// * `a` - A slice representing the coefficients of the first polynomial.
/// * `b` - A slice representing the coefficients of the second polynomial.
/// * `mod_q` - An optional modulus for the coefficients.
///
/// # Returns
///
/// A vector representing the coefficients of the resulting polynomial.
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


pub fn ntt_mul(a: &RingElement, b: &RingElement) -> RingElement {
    let len = a.coeffs.len().next_power_of_two() * 2;
    let root_unity = crate::root_of_unity::choose_root_unity(len, MOD_Q).unwrap();
    let mut a_extended = a.coeffs.to_vec();
    a_extended.resize(len, 0);
    ntt_pow_of_2(&mut a_extended, MOD_Q, root_unity);
    let mut b_extended = b.coeffs.to_vec();
    b_extended.resize(len, 0);
    ntt_pow_of_2(&mut b_extended, MOD_Q, root_unity);

    let mut c_extended = hadamard(&b_extended, &a_extended);

    inverse_ntt_pow_of_2(&mut c_extended, MOD_Q, root_unity);
    let mut v_intt = reduce_quotient(&c_extended, &MIN_POLY);
    reduce_mod(&mut v_intt, MOD_1);
    RingElement{ coeffs: <[BASE_INT; PHI]>::try_from(v_intt).unwrap() }
}


/// Multiplies two polynomials using the naive method with parallelization.
///
/// # Arguments
///
/// * `a` - A slice representing the coefficients of the first polynomial.
/// * `b` - A slice representing the coefficients of the second polynomial.
/// * `mod_q` - An optional modulus for the coefficients.
///
/// # Returns
///
/// A vector representing the coefficients of the resulting polynomial.
pub fn polynomial_mul_parallel(a: &[i128], b: &[i128], mod_q: Option<i128>) -> Vec<i128> {
    let res_len = a.len() + b.len() - 1;
    let res = Mutex::new(vec![0; res_len]);

    (0..res_len).into_par_iter().for_each(|k| {
        let mut sum = 0_i128;
        for i in 0..=k {
            if i < a.len() && (k - i) < b.len() {
                let ai = a[i];
                let bj = b[k - i];
                sum += ai * bj;
            }
        }
        if let Some(q) = mod_q {
            sum %= q;
        }
        res.lock().unwrap()[k] = sum;
    });

    let mut final_res = res.into_inner().unwrap();

    if let Some(q) = mod_q {
        reduce_mod(&mut final_res, q);
    }

    final_res
}

#[test]
fn test_polynomial_mul() {
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    let result = polynomial_mul(&a, &b, None);
    assert_eq!(result, vec![4, 13, 28, 27, 18]);
}

#[test]
fn test_polynomial_mul_parallel() {
    let a = vec![1, 2, 3];
    let b = vec![4, 5, 6];
    let result = polynomial_mul_parallel(&a, &b, None);
    assert_eq!(result, vec![4, 13, 28, 27, 18]);
}

#[test]
fn test_karatsuba_mul() {
    let a = vec![1, 2, 3, 4];
    let b = vec![5, 6, 7, 8];
    let result = karatsuba_mul(&a, &b, None);
    assert_eq!(result, vec![5, 16, 34, 60, 61, 52, 32]);
}

#[test]
fn test_karatsuba_mul_with_mod() {
    let a = vec![1, 2, 3, 4];
    let b = vec![5, 6, 7, 8];
    let mod_q = 10;
    let result = karatsuba_mul(&a, &b, Some(mod_q));
    assert_eq!(result, vec![5, -4, 4, 0, 1, 2, 2]); // Result mod 10
}


#[test]
fn test_ntt_mul() {
    let a = Ring::random();
    let b = Ring::random();
    let result = ntt_mul(&a, &b);
    assert_eq!(result, a * b);
}

/// Subtracts two slices element-wise.
///
/// This function takes two slices and returns a new vector containing the
/// element-wise subtraction of the second slice from the first slice.
///
/// # Arguments
///
/// * `a` - A slice of elements to be subtracted from.
/// * `b` - A slice of elements to subtract.
///
/// # Returns
///
/// A vector where each element is the result of subtracting elements of `b` from `a`.
///
/// # Panics
///
/// This function will panic if the lengths of the input slices are not equal.
///
/// # Example
///
/// ```rust
/// let a = vec![5, 6, 7];
/// let b = vec![1, 2, 3];
/// let result = sub(&a, &b);
/// assert_eq!(result, vec![4, 4, 4]);
/// ```
pub fn sub<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Sub<Output = T> + Copy,
{
    assert_eq!(a.len(), b.len(), "Slices must be the same length");

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x - y)
        .collect()
}

/// Adds two slices element-wise.
///
/// This function takes two slices and returns a new vector containing the
/// element-wise addition of the two slices.
///
/// # Arguments
///
/// * `a` - A slice of elements to be added.
/// * `b` - A slice of elements to add with.
///
/// # Returns
///
/// A vector where each element is the result of adding elements of `a` with `b`.
///
/// # Panics
///
/// This function will panic if the lengths of the input slices are not equal.
///
/// # Example
///
/// ```rust
/// let a = vec![1, 2, 3];
/// let b = vec![4, 5, 6];
/// let result = add(&a, &b);
/// assert_eq!(result, vec![5, 7, 9]);
/// ```
pub fn add<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: Add<Output = T> + Copy,
{
    assert_eq!(a.len(), b.len(), "Slices must be the same length");

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x + y)
        .collect()
}

/// Adds two matrices element-wise.
///
/// # Arguments
///
/// * `matrix_a` - A reference to the first matrix.
/// * `matrix_b` - A reference to the second matrix.
///
/// # Returns
///
/// A new matrix which is the element-wise sum of `matrix_a` and `matrix_b`.
///
/// # Panics
///
/// This function will panic if the dimensions of the two matrices are not the same.
///
/// # Example
///
/// ```rust
/// let matrix_a = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
/// ];
/// let matrix_b = vec![
///     vec![7, 8, 9],
///     vec![10, 11, 12],
/// ];
/// let result = add_matrices(&matrix_a, &matrix_b);
/// assert_eq!(result, vec![
///     vec![8, 10, 12],
///     vec![14, 16, 18],
/// ]);
/// ```
pub fn add_matrices<T>(matrix_a: &[Vec<T>], matrix_b: &[Vec<T>]) -> Vec<Vec<T>>
where
    T: Add<Output = T> + Copy + Zero + Send + Sync,
{
    let nrows = matrix_a.len();
    let ncols = matrix_a[0].len();

    // Ensure the dimensions of the two matrices are the same
    assert_eq!(nrows, matrix_b.len(), "The number of rows in the matrices must be the same");
    assert_eq!(ncols, matrix_b[0].len(), "The number of columns in the matrices must be the same");

    // Allocate the resulting matrix with the required size
    let mut result = vec![vec![T::zero(); ncols]; nrows];

    result.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in 0..ncols {
            row[j] = matrix_a[i][j] + matrix_b[i][j];
        }
    });

    result
}

/// Adds corresponding elements of two vectors and returns the result as a new vector.
///
/// # Arguments
///
/// * `vector_a` - A reference to the first vector.
/// * `vector_b` - A reference to the second vector.
///
/// # Returns
///
/// A new vector containing the result of element-wise addition.
///
/// # Type Parameters
///
/// * `T` - The type of the elements in the vectors. It must implement the `Add`, `Zero`, `Copy`, `Send`, and `Sync` traits.
///
/// # Panics
///
/// This function will panic if the input vectors are not of the same length.
///
/// # Examples
///
/// ```
/// # fn main() {
/// let vector_a = vec![1, 2, 3, 4];
/// let vector_b = vec![5, 6, 7, 8];
/// let result = add_vectors(&vector_a, &vector_b);
/// assert_eq!(result, vec![6, 8, 10, 12]);
/// # }
/// ```
pub fn add_vectors<T>(vector_a: &Vec<T>, vector_b: &Vec<T>) -> Vec<T>
where
    T: Add<Output = T> + Copy + Zero + Send + Sync,
{
    assert_eq!(vector_a.len(), vector_b.len(), "Vectors must be of the same length");

    vector_a.iter().zip(vector_b).map(|(&a, &b)| a + b).collect()
}

#[test]
fn test_add_vectors_integers() {
    let vector_a = vec![1, 2, 3, 4];
    let vector_b = vec![5, 6, 7, 8];
    let result = add_vectors(&vector_a, &vector_b);
    assert_eq!(result, vec![6, 8, 10, 12]);
}

#[test]
fn test_add_vectors_floats() {
    let vector_a = vec![1.0, 2.0, 3.0, 4.0];
    let vector_b = vec![0.5, 1.5, 2.5, 3.5];
    let result = add_vectors(&vector_a, &vector_b);
    assert_eq!(result, vec![1.5, 3.5, 5.5, 7.5]);
}

#[test]
#[should_panic(expected = "Vectors must be of the same length")]
fn test_add_vectors_different_lengths() {
    let vector_a = vec![1, 2, 3];
    let vector_b = vec![1, 2, 3, 4];
    add_vectors(&vector_a, &vector_b); // This should panic
}

#[test]
fn test_add_vectors_zeros() {
    let vector_a = vec![0, 0, 0];
    let vector_b = vec![1, 2, 3];
    let result = add_vectors(&vector_a, &vector_b);
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_add_matrices_basic() {
    let matrix_a = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
    ];
    let matrix_b = vec![
        vec![7, 8, 9],
        vec![10, 11, 12],
    ];
    let result = add_matrices(&matrix_a, &matrix_b);
    let expected = vec![
        vec![8, 10, 12],
        vec![14, 16, 18],
    ];
    assert_eq!(result, expected);
}

/// Reshapes a flat vector into a vector of vectors, with each inner vector having a specified chunk size.
///
/// # Arguments
///
/// * `vec` - The flat vector to be reshaped.
/// * `chunk_size` - The size of each chunk (inner vector).
///
/// # Returns
///
/// A vector of vectors, where each inner vector has a length of `chunk_size`.
/// The last inner vector may have fewer elements if the length of `vec` is not a multiple of `chunk_size`.
///
/// # Example
///
/// ```rust
/// let vec = vec![1, 2, 3, 4, 5, 6, 7];
/// let chunk_size = 3;
/// let result = reshape(vec, chunk_size);
/// assert_eq!(result, vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7],
/// ]);
/// ```
pub fn reshape<T>(vec: &Vec<T>, chunk_size: usize) -> Vec<Vec<T>>
where
    T: Clone,
{
    vec.chunks(chunk_size).map(|chunk| chunk.to_vec()).collect()
}


/// Extracts a submatrix with columns ranging from `from` to `to` (inclusive).
///
/// # Arguments
///
/// * `mat` - A 2D vector representing the input matrix.
/// * `from` - The starting column index (inclusive).
/// * `to` - The ending column index (inclusive).
///
/// # Returns
///
/// A new 2D vector (submatrix) containing the specified range of columns.
///
/// # Panics
///
/// Panics if the matrix is empty, if `from` is greater than `to`, or if `to` is out of bounds.
///
/// # Examples
///
/// ```
/// let mat = vec![
///     vec![1, 2, 3, 4],
///     vec![5, 6, 7, 8],
///     vec![9, 10, 11, 12],
/// ];
///
/// let sub_mat = columns(mat, 1, 3);
/// assert_eq!(sub_mat, vec![
///     vec![2, 3, 4],
///     vec![6, 7, 8],
///     vec![10, 11, 12]
/// ]);
/// ```
pub fn columns<T: Clone>(mat: &Vec<Vec<T>>, from: usize, to: usize) -> Vec<Vec<T>> {
    if mat.is_empty() {
        panic!("Matrix is empty");
    }
    if !from < to {
        panic!("Invalid range: `from` cannot be greater than `to`");
    }
    if to >= mat[0].len() {
        panic!("Invalid range: `to` is out of bounds");
    }

    let mut submatrix = Vec::new();
    for row in mat.iter() {
        let sub_row = row[from..=to].to_vec();
        submatrix.push(sub_row);
    }

    submatrix
}

#[test]
fn test_columns_normal_case() {
    let mat = vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
        vec![9, 10, 11, 12],
    ];

    let sub_mat = columns(&mat, 1, 3);
    assert_eq!(sub_mat, vec![
        vec![2, 3, 4],
        vec![6, 7, 8],
        vec![10, 11, 12]
    ]);
}

/// Extracts the last `n` columns from the input matrix.
///
/// # Arguments
///
/// * `mat` - A 2D vector representing the input matrix.
/// * `n` - The number of columns to extract from the end.
///
/// # Returns
///
/// A new 2D vector (submatrix) containing the last `n` columns.
///
/// # Panics
///
/// Panics if the matrix is empty or if `n` is greater than the number of columns in the matrix.
///
/// # Examples
///
/// ```
/// let mat = vec![
///     vec![1, 2, 3, 4],
///     vec![5, 6, 7, 8],
///     vec![9, 10, 11, 12],
/// ];
///
/// let sub_mat = last_n_columns(mat, 2);
/// assert_eq!(sub_mat, vec![
///     vec![3, 4],
///     vec![7, 8],
///     vec![11, 12]
/// ]);
/// ```
pub fn last_n_columns<T: Clone>(mat: &Vec<Vec<T>>, n: usize) -> Vec<Vec<T>> {
    if mat.is_empty() {
        panic!("Matrix is empty");
    }
    let num_columns = mat[0].len();
    if n > num_columns {
        panic!("Invalid range: `n` is greater than the number of columns in the matrix");
    }

    let from = num_columns - n;
    let to = num_columns - 1;

    let mut submatrix = Vec::new();
    for row in mat.iter() {
        let sub_row = row[from..=to].to_vec();
        submatrix.push(sub_row);
    }

    submatrix
}


/// Extracts the first `n` columns from the input matrix.
///
/// # Arguments
///
/// * `matrix` - A 2D vector representing the input matrix.
/// * `n` - The number of columns to extract from the start.
///
/// # Returns
///
/// A new 2D vector (submatrix) containing the first `n` columns.
///
/// # Panics
///
/// Panics if the matrix is empty or if `n` is greater than the number of columns in the matrix.
///
/// # Examples
///
/// ```
/// let mat = vec![
///     vec![1, 2, 3, 4],
///     vec![5, 6, 7, 8],
///     vec![9, 10, 11, 12],
/// ];
///
/// let sub_mat = first_n_columns(mat, 2);
/// assert_eq!(sub_mat, vec![
///     vec![1, 2],
///     vec![5, 6],
///     vec![9, 10]
/// ]);
/// ```
pub fn first_n_columns<T: Clone>(matrix: &Vec<Vec<T>>, n: usize) -> Vec<Vec<T>> {
    if matrix.is_empty() {
        panic!("Matrix is empty");
    }
    let num_columns = matrix[0].len();
    if n > num_columns {
        panic!("Invalid range: `n` is greater than the number of columns in the matrix");
    }

    let mut submatrix = Vec::new();
    for row in matrix.iter() {
        let sub_row = row[0..n].to_vec();
        submatrix.push(sub_row);
    }

    submatrix
}


/// Computes the row-wise tensor product of two matrices `a` and `b`.
///
/// # Arguments
///
/// * `a` - A matrix represented as a vector of vectors of generic type `T`.
/// * `b` - A matrix represented as a vector of vectors of generic type `T`.
///
/// # Returns
///
/// A matrix represented as a vector of vectors of generic type `T`, where each row of the result
/// is the tensor product of the corresponding rows of `a` and `b`.
///
/// # Panics
///
/// Panics if the number of rows in `a` and `b` do not match.
///
/// # Examples
///
/// ```
/// let a = vec![
///     vec![1, 2],
///     vec![3, 4],
/// ];
///
/// let b = vec![
///     vec![5, 6],
///     vec![7, 8],
/// ];
///
/// let result = row_wise_tensor(a, b);
/// assert_eq!(result, vec![
///     vec![5, 6, 10, 12],
///     vec![21, 24, 28, 32],
/// ]);
/// ```
pub fn row_wise_tensor<T>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Copy + Mul<Output = T> + Send + Sync,
{
    if a.len() != b.len() {
        panic!("Matrices `a` and `b` must have the same number of rows");
    }

    // Use the same length from vector `a` as they must be equal
    let num_rows = a.len();

    // Preallocate the result vector with the required size
    let mut result = Vec::with_capacity(num_rows);

    // Use par_iter to parallelize the outer loop
    result.par_extend(
        a.par_iter()
            .zip(b.par_iter())
            .map(|(row_a, row_b)| {
                let mut tensor_row = Vec::with_capacity(row_a.len() * row_b.len());
                for &elem_a in row_a {
                    for &elem_b in row_b {
                        tensor_row.push(elem_a * elem_b);
                    }
                }
                tensor_row
            })
    );

    result
}
#[test]
fn test_row_wise_tensor_normal_case() {
    let a = vec![
        vec![1, 2],
        vec![3, 4],
    ];

    let b = vec![
        vec![5, 6],
        vec![7, 8],
    ];

    let result = row_wise_tensor(&a, &b);
    assert_eq!(result, vec![
        vec![5, 6, 10, 12],
        vec![21, 24, 28, 32],
    ]);
}

#[test]
#[should_panic(expected = "Matrices `a` and `b` must have the same number of rows")]
fn test_row_wise_tensor_mismatched_rows() {
    let a = vec![
        vec![1, 2],
    ];

    let b = vec![
        vec![5, 6],
        vec![7, 8],
    ];

    row_wise_tensor(&a, &b);
}

#[test]
fn test_row_wise_tensor_single_element_rows() {
    let a = vec![
        vec![1],
        vec![3],
    ];

    let b = vec![
        vec![2],
        vec![4],
    ];

    let result = row_wise_tensor(&a, &b);
    assert_eq!(result, vec![
        vec![2],
        vec![12],
    ]);
}

#[test]
fn test_row_wise_tensor_empty() {
    let a: Vec<Vec<i32>> = vec![];
    let b: Vec<Vec<i32>> = vec![];

    let result = row_wise_tensor(&a, &b);
    assert!(result.is_empty());
}

/// Zips multiple 2D matrices horizontally by alternating their columns.
///
/// # Arguments
///
/// * `matrices` - A reference to a vector of 2D matrices (each represented as `Vec<Vec<T>>`).
///
/// # Returns
///
/// A 2D matrix containing the columns of the input matrices alternated.
///
/// # Panics
///
/// Panics if the input matrices do not have the same number of rows or columns.
///
/// # Examples
///
/// ```
/// let mat1 = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6]
/// ];
///
/// let mat2 = vec![
///     vec![7, 8, 9],
///     vec![10, 11, 12]
/// ];
///
/// let result = zip_columns_horizontally(&vec![mat1, mat2]);
///
/// assert_eq!(result, vec![
///     vec![1, 7, 2, 8, 3, 9],
///     vec![4, 10, 5, 11, 6, 12]
/// ]);
/// ```
pub fn zip_columns_horizontally<T: Copy>(matrices: &Vec<Vec<Vec<T>>>) -> Vec<Vec<T>> {
    if matrices.is_empty() {
        return vec![];
    }

    let nrows = matrices[0].len();
    let ncols = matrices[0][0].len();

    // Check that all matrices have the same number of rows and columns
    for matrix in matrices {
        if matrix.len() != nrows {
            panic!("All matrices must have the same number of rows");
        }
        for row in matrix.iter() {
            if row.len() != ncols {
                panic!("All matrices must have the same number of columns");
            }
        }
    }

    // Initialize the resulting matrix with the appropriate number of rows
    let mut result = vec![Vec::with_capacity(ncols * matrices.len()); nrows];

    // Concatenate matrix columns horizontally by alternating columns
    for i in 0..ncols {
        for matrix in matrices {
            for (j, row) in matrix.iter().enumerate() {
                result[j].push(row[i]);
            }
        }
    }

    result
}

#[test]
fn test_zip_columns_horizontally_basic() {
    let mat1 = vec![
        vec![1, 2, 3],
        vec![4, 5, 6]
    ];

    let mat2 = vec![
        vec![7, 8, 9],
        vec![10, 11, 12]
    ];

    let result = zip_columns_horizontally(&vec![mat1, mat2]);
    assert_eq!(result, vec![
        vec![1, 7, 2, 8, 3, 9],
        vec![4, 10, 5, 11, 6, 12]
    ]);
}


/// Splits a given matrix into submatrices by columns, based on a specified chunk size.
///
/// # Arguments
///
/// * `matrix` - The original 2D matrix to be split, represented as `Vec<Vec<T>>`.
/// * `chunk_size` - The number of columns in each submatrix.
///
/// # Returns
///
/// A vector of 2D submatrices where each submatrix contains `chunk_size` columns from the original matrix.
///
/// # Panics
///
/// Panics if `chunk_size` does not evenly divide the number of columns in the matrix.
///
/// # Examples
///
/// ```
/// let matrix = vec![
///     vec![1, 2, 3, 4],
///     vec![5, 6, 7, 8]
/// ];
///
/// let result = split_into_submatrices_by_columns(&matrix, 2);
///
/// assert_eq!(result, vec![
///     vec![
///         vec![1, 2],
///         vec![5, 6]
///     ],
///     vec![
///         vec![3, 4],
///         vec![7, 8]
///     ]
/// ]);
/// ```
pub fn split_into_submatrices_by_columns<T: Clone>(matrix: &Vec<Vec<T>>, chunk_size: usize) -> Vec<Vec<Vec<T>>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return vec![];
    }

    let ncols = matrix[0].len();
    if ncols % chunk_size != 0 {
        panic!("The number of columns in the matrix must be divisible by the chunk size for splitting.");
    }

    let nrows = matrix.len();
    let n_chunks = ncols / chunk_size;
    let mut submatrices = vec![vec![vec![]; nrows]; n_chunks];

    // Split the matrix into submatrices by columns
    for i in 0..n_chunks {
        for j in 0..nrows {
            submatrices[i][j] = matrix[j][i * chunk_size..(i + 1) * chunk_size].to_vec();
        }
    }

    submatrices
}

#[test]
fn test_split_into_submatrices_by_columns_basic() {
    let matrix = vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
    ];

    let result = split_into_submatrices_by_columns(&matrix, 2);
    assert_eq!(result, vec![
        vec![
            vec![1, 2],
            vec![5, 6],
        ],
        vec![
            vec![3, 4],
            vec![7, 8],
        ],
    ]);
}

/// Decomposes a vector of coefficients into their binary representations.
///
/// This function takes a vector of coefficients and a logarithmic base `log_q`.
/// It computes the binary representation of each coefficient up to `log_q` bits
/// and returns a vector of vectors, where each inner vector contains the binary
/// digits of the corresponding coefficient.
///
/// # Arguments
///
/// * `coeffs` - A vector of coefficients to be decomposed.
/// * `log_q` - The logarithmic base for binary decomposition.
///
/// # Returns
///
/// A vector of vectors, where each inner vector contains the binary digits of
/// the corresponding coefficient from the input vector. The outer vector's length
/// is equal to `log_q`, and each inner vector's length is equal to the length of
/// the input `coeffs` vector.
///
/// # Example
///
/// ```rust
/// let coeffs = vec![3, 7, 10];
/// let log_q = 4;
/// let result = binary_decomposition(&coeffs, log_q);
/// assert_eq!(result, vec![
///     vec![0, 0, 0],
///     vec![0, 1, 0],
///     vec![1, 1, 1],
///     vec![1, 1, 0],
/// ]);
/// // Explanation:
/// // 3 (base 10) -> 0011 (base 2)
/// // 7 (base 10) -> 0111 (base 2)
/// // 10 (base 10) -> 1010 (base 2)
/// // Result transposed:
/// // [
/// //   [0, 0, 0],
/// //   [0, 1, 0],
/// //   [1, 1, 1],
/// //   [1, 1, 0]
/// // ]
/// ```
pub fn binary_decomposition(coeffs: &Vec<i128>, log_q: usize) -> Vec<Vec<i128>> {
    // Clone and reduce coeffs modulo 2^log_q
    let mut cloned = coeffs.clone();
    cloned.iter_mut().for_each(|x| *x %= 1 << log_q);

    // Convert each coefficient to its binary representation
    let binary_coeffs: Vec<Vec<i128>> = cloned.into_iter().map(|x| {
        let mut binary = Vec::with_capacity(log_q);
        for i in (0..log_q).rev() {
            binary.push((x >> i) & 1);
        }
        binary
    }).collect();

    // Initialize the result vector with zeros
    let mut result: Vec<Vec<i128>> = vec![vec![0; binary_coeffs.len()]; log_q];

    // Transpose the binary_coeffs matrix
    for (i, row) in binary_coeffs.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            result[j][i] = value;
        }
    }

    result
}

// Unit testing
#[cfg(test)]
mod binary_decomposition_tests {
    use super::*;

    #[test]
    fn test_binary_decomposition_basic() {
        let coeffs = vec![3, 7, 10];
        let log_q = 4;
        let result = binary_decomposition(&coeffs, log_q);
        let expected = vec![
            vec![0, 0, 1],
            vec![0, 1, 0],
            vec![1, 1, 1],
            vec![1, 1, 0],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_decomposition_single_bit() {
        let coeffs = vec![1, 0, 1, 1];
        let log_q = 1;
        let result = binary_decomposition(&coeffs, log_q);
        let expected = vec![
            vec![1, 0, 1, 1],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_decomposition_mixed_values() {
        let coeffs = vec![15, 8, 12, 4];
        let log_q = 4;
        let result = binary_decomposition(&coeffs, log_q);
        let expected = vec![
            vec![1, 1, 1, 0],
            vec![1, 0, 1, 1],
            vec![1, 0, 0, 0],
            vec![1, 0, 0, 0],
        ];
        assert_eq!(result, expected);
    }
}

/// Decomposes a vector of coefficients into their representations with a specified radix in inverse order.
///
/// This function takes a vector of coefficients and a radix.
/// It computes the representation of each coefficient up to the maximum value
/// representable by the radix and returns a vector of vectors, where each inner vector
/// contains the digits of the corresponding coefficient in inverse order.
///
/// # Arguments
///
/// * `coeffs` - A vector of coefficients to be decomposed.
/// * `radix` - The base for the decomposition.
/// * `log_q` - The number of digits in the specified radix.
///
/// # Returns
///
/// A vector of vectors, where each inner vector contains the digits of
/// the corresponding coefficient from the input vector in inverse order. The outer vector's length
/// is equal to `log_q`, and each inner vector's length is equal to the length of
/// the input `coeffs` vector.
///
/// # Example
///
/// ```rust
/// let coeffs = vec![23, 45, 67];
/// let radix = 10;
/// let log_q = 3;
/// let result = binary_decomposition_radix(&coeffs, radix, log_q);
/// assert_eq!(result, vec![
///     vec![0, 0, 0],
///     vec![2, 4, 6],
///     vec![3, 5, 7],
/// ]);
/// // Explanation:
/// // 23 (base 10) -> [3, 2, 0] (base 10 representation as [units, tens, hundreds], in inverse order)
/// // 45 (base 10) -> [5, 4, 0]
/// // 67 (base 10) -> [7, 6, 0]
/// // Result transposed:
/// // [
/// //   [0, 0, 0],
/// //   [2, 4, 6],
/// //   [3, 5, 7]
/// // ]
/// ```
pub fn binary_decomposition_radix(coeffs: &Vec<i128>, radix: i128, log_q: usize) -> Vec<Vec<i128>> {
    // Clone the coefficients to avoid modifying the original vector.
    let mut cloned = coeffs.clone();

    // Convert each coefficient to its representation in the specified radix.
    let radix_coeffs: Vec<Vec<i128>> = cloned.into_iter().map(|mut x| {
        let mut base_repr = Vec::with_capacity(log_q);
        for _ in 0..log_q {
            base_repr.push(x % radix);
            x /= radix;
        }
        base_repr.reverse(); // Ensure the representation is in inverse order.
        base_repr
    }).collect();

    // Initialize the result vector with zeros.
    let mut result: Vec<Vec<i128>> = vec![vec![0; radix_coeffs.len()]; log_q];

    // Transpose the radix_coeffs matrix.
    for (i, row) in radix_coeffs.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            result[j][i] = value;
        }
    }

    result
}

// Unit testing
#[cfg(test)]
mod tests_binary_decomposition_custom_radix {
    use super::*;

    #[test]
    fn test_binary_decomposition_binary_radix() {
        let coeffs = vec![15, 8, 12, 4];
        let radix = 2;
        let log_q = 6;
        let result = binary_decomposition_radix(&coeffs, radix, log_q);
        let expected = binary_decomposition(&coeffs, log_q);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_decomposition_radix_basic() {
        let coeffs = vec![23, 45, 67];
        let radix = 10;
        let log_q = 3;
        let result = binary_decomposition_radix(&coeffs, radix, log_q);
        let expected = vec![
            vec![0, 0, 0],
            vec![2, 4, 6],
            vec![3, 5, 7],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_decomposition_radix_single_digit() {
        let coeffs = vec![1, 0, 2, 3];
        let radix = 2;
        let log_q = 1;
        let result = binary_decomposition_radix(&coeffs, radix, log_q);
        let expected = vec![
            vec![1, 0, 0, 1],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_decomposition_radix_mixed_values() {
        let coeffs = vec![15, 8, 12, 4];
        let radix = 10;
        let log_q = 2;
        let result = binary_decomposition_radix(&coeffs, radix, log_q);
        let expected = vec![
            vec![1, 0, 1, 0],
            vec![5, 8, 2, 4],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_binary_decomposition_radix_large_radix() {
        let coeffs = vec![255, 128];
        let radix = 16;
        let log_q = 2;
        let result = binary_decomposition_radix(&coeffs, radix, log_q);
        let expected = vec![
            vec![15, 8],
            vec![15, 0],
        ];
        assert_eq!(result, expected);
    }
}

/// Joins two matrices horizontally.
///
/// This function takes two matrices and joins them horizontally, i.e., concatenates them column-wise.
///
/// # Arguments
///
/// * `mat1` - A reference to the first matrix.
/// * `mat2` - A reference to the second matrix.
///
/// # Returns
///
/// A single matrix that is the result of concatenating the input matrices column-wise.
///
/// # Panics
///
/// This function will panic if the matrices do not have the same number of rows.
///
/// # Example
///
/// ```rust
/// let mat1 = vec![
///     vec![1, 2],
///     vec![3, 4],
/// ];
/// let mat2 = vec![
///     vec![5, 6],
///     vec![7, 8],
/// ];
/// let result = join_matrices_horizontally(&mat1, &mat2);
/// assert_eq!(result, vec![
///     vec![1, 2, 5, 6],
///     vec![3, 4, 7, 8],
/// ]);
/// ```
pub fn join_matrices_horizontally<T>(mat1: &Vec<Vec<T>>, mat2: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    // Ensure both matrices have the same number of rows
    assert_eq!(mat1.len(), mat2.len(), "Both matrices must have the same number of rows");

    // Initialize the result matrix
    let mut result = Vec::with_capacity(mat1.len());

    // Extend each row of the result with rows from both matrices
    for (row1, row2) in mat1.iter().zip(mat2) {
        let mut new_row = row1.clone();
        new_row.extend(row2.clone());
        result.push(new_row);
    }

    result
}

// Unit testing
#[cfg(test)]
mod join_matrices_horizontally_tests {
    use super::*;

    #[test]
    fn test_join_matrices_horizontally_basic() {
        let mat1 = vec![
            vec![1, 2],
            vec![3, 4],
        ];
        let mat2 = vec![
            vec![5, 6],
            vec![7, 8],
        ];
        let result = join_matrices_horizontally(&mat1, &mat2);
        let expected = vec![
            vec![1, 2, 5, 6],
            vec![3, 4, 7, 8],
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_join_matrices_horizontally_single_row() {
        let mat1 = vec![vec![1, 2, 3]];
        let mat2 = vec![vec![4, 5, 6]];
        let result = join_matrices_horizontally(&mat1, &mat2);
        let expected = vec![vec![1, 2, 3, 4, 5, 6]];
        assert_eq!(result, expected);
    }

    #[test]
    #[should_panic(expected = "Both matrices must have the same number of rows")]
    fn test_join_matrices_horizontally_diff_row_count() {
        let mat1 = vec![
            vec![1, 2],
            vec![3, 4],
        ];
        let mat2 = vec![
            vec![5, 6],
        ];
        join_matrices_horizontally(&mat1, &mat2);
    }

    #[test]
    fn test_join_matrices_horizontally_empty() {
        let mat1: Vec<Vec<i32>> = vec![];
        let mat2: Vec<Vec<i32>> = vec![];
        let result = join_matrices_horizontally(&mat1, &mat2);
        let expected: Vec<Vec<i32>> = vec![];
        assert_eq!(result, expected);
    }
}


/// Computes a power series for the given element, prefixing the series with a `1`.
///
/// # Arguments
///
/// * `element` - A `RingElement` which serves as the base for the power series.
/// * `len` - The length of the series.
///
/// # Returns
///
/// A vector containing a single vector of `RingElement`s, representing a power series prefixed with `1`.
pub fn compute_one_prefixed_power_series(element: &RingElement, len: usize) -> Vec<RingElement> {
    let mut series = Vec::with_capacity(len);
    series.push(Ring::constant(1));
    series.push(element.clone());

    let mut power = element.clone();
    for _ in 2..len {
        power = power.clone() * element.clone();
        series.push(power.clone());
    }

    series
}


#[test]
fn test_compute_one_prefixed_power_series() {
    let element = Ring::constant(2);
    let result = compute_one_prefixed_power_series(&element, 4);
    assert_eq!(
        result,
        vec![Ring::constant(1), Ring::constant(2), Ring::constant(4), Ring::constant(8)]
    );
}

/// Generates a series of `RingElement` prefixed with a `1` and followed by zero elements.
///
/// # Arguments
///
/// * `len` - The length of vector.
///
/// # Returns
///
/// A vector containing  `RingElement`s. Vector has a `1` followed by zero elements.
pub fn compute_one_prefixed_zero_series(len: usize) -> Vec<RingElement> {
    let mut series = Vec::with_capacity(len);
    series.push(Ring::constant(1));

    for _ in 1..len {
        series.push(Ring::zero());
    }

    series
}

#[test]
fn test_compute_one_prefixed_zero_series() {
    let result = compute_one_prefixed_zero_series(4);
    assert_eq!(
        result,
        vec![Ring::constant(1), Ring::zero(), Ring::zero(), Ring::zero()],
    );
}




/// Generates an n x m matrix with all elements set to `1`.
///
/// # Arguments
///
/// * `n` - The number of rows in the matrix.
/// * `m` - The number of columns in the matrix.
///
/// # Returns
///
/// A vector of vectors representing the matrix full of `1`s.
pub fn one_mat(rows: usize, cols: usize) -> Vec<Vec<RingElement>> {
    (0..rows).map(|_| {
        (0..cols).map(|_| Ring::constant(1)).collect()
    }).collect()
}
#[test]
fn test_one_mat() {
    let result = one_mat(2, 3);
    assert_eq!(
        result,
        vec![
            vec![Ring::constant(1), Ring::constant(1), Ring::constant(1)],
            vec![Ring::constant(1), Ring::constant(1), Ring::constant(1)]
        ]
    );
}

/// Computes the conjugate of each element in a 2D `RingElement` vector.
///
/// # Arguments
///
/// * `series` - A 2D vector of `RingElement`s.
///
/// # Returns
///
/// A new 2D vector where each `RingElement` in the input has been replaced by its conjugate.
pub fn conjugate_matrix(series: &Vec<Vec<RingElement>>) -> Vec<Vec<RingElement>> {
    series.iter().map(|row|
    row.iter().map(RingElement::conjugate).collect()
    ).collect()
}

/// Computes the conjugate of each element in a `RingElement` vector.
///
/// # Arguments
///
/// * `series` - A vector of `RingElement`s.
///
/// # Returns
///
/// A new vector where each `RingElement` in the input has been replaced by its conjugate.
pub fn conjugate_vector(row: &Vec<RingElement>) -> Vec<RingElement> {
    row.iter().map(RingElement::conjugate).collect()
}

/// Computes the inverse of each element in a `RingElement` vector.
///
/// # Arguments
///
/// * `series` - A vector of `RingElement`s.
///
/// # Returns
///
/// A new vector where each `RingElement` in the input has been replaced by its inverse.
pub fn inverse_vector(row: &Vec<RingElement>) -> Vec<RingElement> {
    row.iter().map(RingElement::inverse).collect()
}

/// Calls a Sage script to compute the inverse of a polynomial.
///
/// # Arguments
///
/// * `a` - A `RingElement` to find the inverse for.
///
/// # Returns
///
/// * `Option<RingElement>` - The inverse of the given polynomial if it exists, otherwise `None`.
pub fn call_sage_inverse_polynomial(a: &RingElement) -> Option<RingElement> {
    // Prepare the polynomial coefficients as a string.
    let coeffs_a_str = format!("{:?}", a.coeffs).replace(" ", "");

    // Call the Sage script with the required arguments.
    let output = Command::new("sage")
        .arg("inverse.sage")
        .arg(&coeffs_a_str)
        .arg(format!("{:?}", CONDUCTOR_COMPOSITE).replace(" ", ""))
        .arg(MOD_Q.to_string())
        .output()
        .expect("Failed to execute Sage script");

    // Check if the Sage script execution was successful.
    if !output.status.success() {
        eprintln!("Error running Sage script: {:?}", output);
        return None;
    }

    // Process the output from the Sage script.
    let stdout = String::from_utf8_lossy(&output.stdout);
    if stdout.trim() == "None" {
        None // Inverse does not exist
    } else {
        let output = stdout.trim();
        let result: Vec<i128> = output
            .trim_matches(&['[', ']'] as &[_])
            .split(',')
            .map(|s| s.trim().parse().expect("Invalid number"))
            .collect();

        Some(RingElement { coeffs: result.try_into().unwrap() })
    }
}

#[test]
pub fn test_inverse_sage() {
    let a = Ring::random();
    let b = call_sage_inverse_polynomial(&a).unwrap();
    assert_eq!(a * b, Ring::constant(1));
}

/// Multiplies each element of a matrix by a given constant.
///
/// # Arguments
///
/// * `m` - A mutable reference to a matrix represented as a vector of vectors.
/// * `a` - A constant by which each element of the matrix will be multiplied.
///
/// # Examples
///
/// ```
/// let mut matrix = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
/// ];
/// multiply_matrix_constant(&mut matrix, 2);
/// assert_eq!(matrix, vec![
///     vec![2, 4, 6],
///     vec![8, 10, 12],
///     vec![14, 16, 18],
/// ]);
/// ```
pub fn multiply_matrix_constant_in_place<T>(m: &mut Vec<Vec<T>>, a: &T)
where
    T: Mul<Output = T> + Copy,
{
    for row in m.iter_mut() {
        for elem in row.iter_mut() {
            *elem = *elem * *a;
        }
    }
}

#[test]
fn test_multiply_matrix_constant() {
    let mut matrix = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let constant = 2;

    multiply_matrix_constant_in_place(&mut matrix, &constant);

    let expected = vec![
        vec![2, 4, 6],
        vec![8, 10, 12],
        vec![14, 16, 18],
    ];

    assert_eq!(matrix, expected);
}

// Computes `a` raised to the power of `pow` using exponentiation by squaring.
///
/// This method works for `pow` being non-negative. If `pow` is 0, the result is 1.
///
/// # Arguments
///
/// * `a` - The base value of type `T`.
/// * `pow` - The exponent value of type `u32`.
///
/// # Returns
///
/// A value of type `T` representing `a` raised to the power of `pow`.
///
/// # Example
///
/// ```
/// let result = fast_power(2, 10); // result should be 1024
/// ```
pub fn fast_power<T>(a: T, pow: u32) -> T
where
    T: Mul<Output = T> + Copy + One,
{
    if pow == 0 {
        return T::one();
    }

    let mut base = a;
    let mut exponent = pow;
    let mut result = T::one();

    while exponent > 0 {
        if exponent % 2 != 0 {
            result = result * base;
        }
        base = base * base;
        exponent /= 2;
    }

    result
}

#[test]
fn test_fast_power() {
    assert_eq!(fast_power(2, 10), 1024);
    assert_eq!(fast_power(3, 0), 1); // a^0 should be 1
    assert_eq!(fast_power(5, 3), 125); // 5^3 = 5 * 5 * 5
    assert_eq!(fast_power(2, 1), 2); // 2^1 should be 2
}


pub struct PowerSeries {
    pub power_series: Vec<RingElement>,
    pub multiplier:  RingElement
}



