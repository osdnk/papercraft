use num_traits::Zero;
use crate::arithmetic::parallel_dot_matrix_vector;
use crate::r#static::{LOG_Q, MODULE_SIZE};
use crate::ring_i128::{Ring, ring_inner_product, RingElement};
use rayon::prelude::*;

/// Generates a vector of `RingElement` values, each being a power of 2.
///
/// This function creates a vector of `RingElement` values of the specified size,
/// where each element in the vector is `2^i` and `i` decreases from `size-1` to 0.
///
/// # Arguments
///
/// * `size` - The number of `RingElement` values to generate.
///
/// # Returns
///
/// A vector of `RingElement` values where each element is `2^i`, starting from `2^(size-1)` down to `2^0`.
///
/// # Example
///
/// ```
/// let g_values = get_g(5);
/// // g_values will be a vector with elements [16, 8, 4, 2, 1]
/// ```
pub fn get_g(size: usize) -> Vec<RingElement> {
    let mut result = Vec::with_capacity(size);
    for i in (0..size).rev() {
        let number = Ring::constant(i128::pow(2, i as u32));
        result.push(number);
    }
    result
}

#[test]
pub fn test_g_decompose() {
    let ell = Ring::random();
    let decomposed = ell.g_decompose();
    let g = get_g(LOG_Q);
    let ell2 = ring_inner_product(&decomposed, &g);
    assert_eq!(ell, ell2);
}

/// Tensors (also known as the Kronecker product) an identity matrix of specified rank with a given vector.
///
/// # Arguments
///
/// * `vector` - The vector to be tensored with the identity matrix.
/// * `rank` - The rank (size) of the identity matrix.
///
/// # Returns
///
/// A vector of vectors representing the tensored identity matrix with the given vector.
///
/// # Example
///
/// ```
/// let vector = vec![1, 2, 3];
/// let rank = 2;
/// let result = tensor_identity_matrix_with_vector(vector, rank);
/// // result will be:
/// // [
/// //   [1, 2, 3, 0, 0, 0],
/// //   [0, 0, 0, 1, 2, 3]
/// // ]
/// ```
pub fn tensor_identity_matrix_with_vector<T>(vector: &Vec<T>, rank: usize) -> Vec<Vec<T>>
where
    T: Copy + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    // The length of the original vector
    let vec_len = vector.len();
    // Allocate the resulting matrix
    let mut result = vec![vec![T::zero(); vec_len * rank]; rank];

    for i in 0..rank {
        for j in 0..vec_len {
            result[i][i * vec_len + j] = vector[j];
        }
    }

    result
}

#[test]
fn test_tensor_identity_matrix_with_vector() {
    let vector = vec![1, 2, 3];
    let rank = 2;
    let result = tensor_identity_matrix_with_vector(&vector, rank);
    let expected = vec![
        vec![1, 2, 3, 0, 0, 0],
        vec![0, 0, 0, 1, 2, 3]
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_identity_matrix_with_vector_large_rank() {
    let vector = vec![1, 2];
    let rank = 3;
    let result = tensor_identity_matrix_with_vector(&vector, rank);
    let expected = vec![
        vec![1, 2, 0, 0, 0, 0],
        vec![0, 0, 1, 2, 0, 0],
        vec![0, 0, 0, 0, 1, 2]
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_identity_matrix_with_vector_single_element_vector() {
    let vector = vec![4];
    let rank = 3;
    let result = tensor_identity_matrix_with_vector(&vector, rank);
    let expected = vec![
        vec![4, 0, 0],
        vec![0, 4, 0],
        vec![0, 0, 4]
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_tensor_identity_matrix_with_vector_rank_1() {
    let vector = vec![5, 6, 7];
    let rank = 1;
    let result = tensor_identity_matrix_with_vector(&vector, rank);
    let expected = vec![
        vec![5, 6, 7]
    ];
    assert_eq!(result, expected);
}

/// Tensors (also known as the Kronecker product) an identity matrix of specified rank with a given matrix.
///
/// # Arguments
///
/// * `matrix` - The matrix to be tensored with the identity matrix.
/// * `rank` - The rank (size) of the identity matrix.
///
/// # Returns
///
/// A vector of vectors representing the tensored identity matrix with the given matrix.
///
/// # Example
///
/// ```rust
/// let matrix = vec![
///     vec![1, 2],
///     vec![3, 4],
/// ];
/// let rank = 2;
/// let result = tensor_identity_matrix_with_matrix(&matrix, rank);
/// assert_eq!(result, vec![
///     vec![1, 2, 0, 0],
///     vec![3, 4, 0, 0],
///     vec![0, 0, 1, 2],
///     vec![0, 0, 3, 4],
/// ]);
/// ```
pub fn tensor_identity_matrix_with_matrix<T>(matrix: &Vec<Vec<T>>, rank: usize) -> Vec<Vec<T>>
where
    T: Copy + Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    // The dimensions of the original matrix
    let rows = matrix.len();
    let cols = matrix[0].len();

    // Allocate the resulting matrix
    let mut result = vec![vec![T::zero(); cols * rank]; rows * rank];

    for i in 0..rank {
        for j in 0..rank {
            for r in 0..rows {
                for c in 0..cols {
                    result[i * rows + r][j * cols + c] = (if i == j { matrix[r][c] } else { T::zero() });
                }
            }
        }
    }

    result
}

#[test]
fn test_tensor_identity_matrix_with_matrix() {
    let matrix = vec![
        vec![1, 2],
        vec![3, 4],
    ];
    let rank = 2;
    let result = tensor_identity_matrix_with_matrix(&matrix, rank);
    let expected = vec![
        vec![1, 2, 0, 0],
        vec![3, 4, 0, 0],
        vec![0, 0, 1, 2],
        vec![0, 0, 3, 4],
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_g() {
    let vector = vec![5, 6, 7];
    let rank = 1;
    let result = tensor_identity_matrix_with_vector(&vector, rank);
    let expected = vec![
        vec![5, 6, 7]
    ];
    assert_eq!(result, expected);
}

#[test]
pub fn test_g_decompose_rank_2() {
    let ell =  vec![Ring::random(), Ring::random()];
    let decomposed: Vec<Vec<RingElement>> = ell.iter().map(|r| r.g_decompose()).collect();
    let decompose_flattened: Vec<RingElement> = decomposed.iter().flatten().cloned().collect();
    let G = tensor_identity_matrix_with_vector(&get_g(LOG_Q), 2);
    let ell2 = parallel_dot_matrix_vector(&G, &decompose_flattened);
    assert_eq!(ell, ell2);
}

#[test]
pub fn test_g_decompose_rank_2_easy() {
    let ell = vec![7, 5];
    let decompose_flattened = vec![1, 1, 1, 1, 0, 1];
    let G = tensor_identity_matrix_with_vector(&vec![4, 2, 1], 2);
    let ell2 = parallel_dot_matrix_vector(&G, &decompose_flattened);
    assert_eq!(ell, ell2);
}

/// Chunks a vector into `n` chunks such that the last element of chunk `i` is the first element
/// of chunk `i+1`.
///
/// # Arguments
///
/// * `vec` - The vector to be chunked.
/// * `n` - The number of chunks.
///
/// # Returns
///
/// A vector of vectors, where each inner vector is a chunk.
///
/// # Examples
///
/// ```
/// let vec = vec![1, 2, 3, 4, 5, 6, 7];
/// let n = 3;
/// let result = chunk_witness(vec, n);
/// // Result: [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
/// ```
pub fn chunk_witness<T: Clone>(vec: &Vec<T>, n: usize) -> Vec<Vec<T>> {
    if n == 0 {
        panic!("Number of chunks must be greater than 0");
    }

    let len = vec.len();
    assert_eq!((len - 1) % n, 0, "T is not compatible with the number of repetitions");
    let chunk_size = (len - 1) / n + 1; // Calculate chunk size

    let mut result = Vec::with_capacity(n);

    let mut start = 0;
    while start < len {
        let end = std::cmp::min(start + chunk_size, len);
        result.push(vec[start..end].to_vec());
        if end == len {
            break;
        }
        start = end - 1; // Make sure the last element of the current chunk is the first element of the next chunk
    }

    if result.len() != n {
        panic!("Cannot chunk vector into exactly {} chunks with the given conditions", n);
    }

    result
}

// Unit tests
#[test]
fn test_chunk_witness_basic() {
    let vec = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
    let n = 3;
    let result = chunk_witness(&vec, n);
    let expected = vec![
        vec![1, 2, 3, 4, 5],
        vec![5, 6, 7, 8, 9],
        vec![9, 10, 11, 12, 13]
    ];
    assert_eq!(result, expected);
}

fn test_chunk_witness_large_vector() {
    let vec = (1..99).collect::<Vec<_>>();
    let n = 10;
    let result = chunk_witness(&vec, n);
    assert_eq!(result.len(), 10);
}

/// Transposes a matrix represented as a vector of vectors.
///
/// # Arguments
///
/// * `matrix` - A vector of vectors representing the matrix to be transposed.
///
/// # Returns
///
/// A new vector of vectors representing the transposed matrix.
///
/// # Example
///
/// ```rust
/// let matrix = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
/// ];
/// let result = transpose(matrix);
/// assert_eq!(result, vec![
///     vec![1, 4, 7],
///     vec![2, 5, 8],
///     vec![3, 6, 9],
/// ]);
/// ```
pub fn transpose<T>(matrix: &Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Copy + Zero + Send + Sync,
{
    if matrix.is_empty() {
        return vec![];
    }

    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut transposed = vec![vec![T::zero(); rows]; cols];

    // Transpose using parallel iteration
    transposed.par_iter_mut().enumerate().for_each(|(i, row)| {
        for (j, value) in row.iter_mut().enumerate() {
            *value = matrix[j][i];
        }
    });

    transposed
}

#[test]
fn test_transpose_square_matrix() {
    let matrix = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let result = transpose(&matrix);
    let expected = vec![
        vec![1, 4, 7],
        vec![2, 5, 8],
        vec![3, 6, 9],
    ];
    assert_eq!(result, expected);
}
