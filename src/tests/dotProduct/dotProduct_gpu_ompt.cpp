#include "dotProduct.hpp"
#include "unifyBackends.hpp"
#include <cassert>

namespace Kokkidio::gpu
{

namespace kernel
{

#pragma omp declare target
void dot_colwise(const scalar* m1, const scalar* m2, scalar* v, int rows, int cols, int idx){
	/* Map raw data to const MatrixXs objects to perform .dot() multiplication */ 
	Eigen::Map<const MatrixXs>
		matA{m1, rows, cols},
		matB{m2, rows, cols};

	/* Perform calculation */
	v[idx] = matA.col(idx).dot(matB.col(idx));
}

void dot_coalesced(const scalar* m12, scalar* v, int rows, int cols, int idx){
	/* Map matrix consisting of 2 input matrices to const MatrixXs object */
	Eigen::Map<const MatrixXs> matAB(m12, 2 * rows, cols);

	/* "split" into first (top) and second (bottom) matrix */
	auto matA{matAB.topRows(rows)};
	auto matB{matAB.bottomRows(rows)};
	
	/* Perform calculation */
	v[idx] = matA.col(idx).dot(matB.col(idx));
}
#pragma omp end declare target

} // namespace kernel

template<bool isManual> 
scalar dot_impl( const MatrixXs& m1, const MatrixXs& m2, int iterations ){
	/* Dimensions check */
	assert(m1.rows() == m2.rows());
	assert(m1.cols() == m2.cols());

	const int rows = m1.rows();
	const int cols = m1.cols();

	/* Vector of dot products */
	ArrayXs h_v(cols);

	/* Allocate memory on device and copy data */
	// printf("* Allocate memory on the device, and copy data\n");
	scalar *d_m1, *d_m2, *d_v;
	gpuAllocAndCopy(d_m1, m1);
	gpuAllocAndCopy(d_m2, m2);
	gpuAlloc(d_v, h_v);

	/* Run calculation multiple times */
	// printf("* Compute on the device\n");
	for (int iter = 0; iter < iterations; ++iter){
		// #pragma omp target teams distribute parallel for simd // warning: loop not vectorized
		#pragma omp target teams distribute parallel for
		for (int i = 0; i < cols; ++i){
			if constexpr (isManual){
				d_v[i] = 0;
				for (int r = 0; r < rows; ++r){
					d_v[i] += d_m1[rows * i + r] * d_m2[rows * i + r];
				}
			} else {
				kernel::dot_colwise(d_m1, d_m2, d_v, rows, cols, i);
			}
		}
	}

	// /* Copy vector of dot products to host */
	gpuMemcpyDeviceToHost(d_v, h_v);

	/* Calculate the sum of dot products */
	scalar finalResult_gpu = h_v.sum();

	/* Deallocate device memory */
	gpuFree(d_m1);
	gpuFree(d_m2);
	gpuFree(d_v);

	/* Return the result */
	return finalResult_gpu;
}

scalar dot_manual(const MatrixXs& m1, const MatrixXs& m2, int iterations){
	return dot_impl<true>(m1, m2, iterations);
}
scalar dot_colwise(const MatrixXs& m1, const MatrixXs& m2, int iterations){
	return dot_impl<false>(m1, m2, iterations);
}

scalar dot_coalesced(const MatrixXs& m1, const MatrixXs& m2, int iterations){
	/* Dimensions check */
	assert(m1.rows() == m2.rows());
	assert(m1.cols() == m2.cols());

	const int rows = m1.rows();
	const int cols = m1.cols();

	/* Vector of dot products */
	ArrayXs h_v(cols);

	/* Create matrix m12 to store both m1 and m2 */
	MatrixXs m12(rows + rows, cols);

	/* Fill m12 with values of m1 and m2, m1 on top amd m2 on bottom */
	m12 << m1, m2;

	/* Allocate memory on the device for input matrices and vector of dot products */
	// printf("* Allocate memory on the device, and copy data\n");
	scalar *d_m12, *d_v;

	gpuAllocAndCopy(d_m12, m12);
	gpuAlloc(d_v, h_v);

	/* Run calculation multiple times */
	// printf("* Compute on the device\n");
	for (int i = 0; i < iterations; ++i){
		#pragma omp target teams distribute parallel for
		for (int i = 0; i < cols; ++i){
			kernel::dot_coalesced(d_m12, d_v, rows, cols, i);
		}
	}

	/* Copy vector of dot products to host */
	gpuMemcpyDeviceToHost(d_v, h_v);

	/* Calculate the sum of dot products */
	scalar finalResult_gpu = h_v.sum();

	/* Deallocate device memory */
	gpuFree(d_m12);
	gpuFree(d_v);

	/* Return the result */
	return finalResult_gpu;
}

} // namespace Kokkidio::gpu
