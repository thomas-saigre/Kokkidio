#include <iostream>
#include <cassert>

#include "dotProduct.hpp"
#include "unifyBackends.hpp"

namespace Kokkidio::gpu
{

namespace dot
{

/* CUDA kernel with Eigen's .dot() function */
__global__ void colwise_eigen(const scalar* m1, const scalar* m2, scalar* v, int rows, int cols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < cols){
		/* Map raw data to const MatrixXs objects to perform .dot() multiplication */ 
		Eigen::Map<const MatrixXs>
			matA{m1, rows, cols},
			matB{m2, rows, cols};

		/* Perform calculation */
		v[idx] = matA.col(idx).dot(matB.col(idx));
		// scalar res = matA.col(idx).dot(matB.col(idx));
		// v[idx] = res;
		// v[idx] = ( matA.col(idx).array() * matB.col(idx).array() ).sum();
	}
}

/* CUDA kernel with Eigen's .dot() function (memory coalescing pattern) */
__global__ void merged_eigen(const scalar* m12, scalar* v, int rows, int cols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < cols){
		/* Map matrix consisting of 2 input matrices to const MatrixXs object */
		Eigen::Map<const MatrixXs> matAB(m12, 2 * rows, cols);

		/* "split" into first (top) and second (bottom) matrix */
		auto matA{matAB.topRows(rows)};
		auto matB{matAB.bottomRows(rows)};
		
		/* Perform calculation */
		v[idx] = matA.col(idx).dot(matB.col(idx));
	}
}

/* CUDA kernel (manual implementation of dot product) */
__global__ void colwise_manual(const scalar* m1, const scalar* m2, scalar* v, int rows, int cols){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < cols) {
		/* Every thread calculates it's column */
		int startIdx = idx * rows;
		scalar dot_product = 0;
		for (int i = 0; i < rows; ++i)
		{
			dot_product += m1[startIdx + i] * m2[startIdx + i];
		}
		v[idx] = dot_product;
	}
}

} // namespace dot

using K = Kernel;
constexpr Target dev { Target::device };

template<>
scalar dotProduct<dev, K::colwise_eigen>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
	/* Dimensions check */
	assert(m1.rows() == m2.rows());
	assert(m1.cols() == m2.cols());

	const int rows = m1.rows();
	const int cols = m1.cols();

	/* Vector of dot products */
	ArrayXs h_v(cols);

	/* Allocate memory on device and copy data */
	scalar *d_m1, *d_m2, *d_v;
	gpuAllocAndCopy(d_m1, m1);
	gpuAllocAndCopy(d_m2, m2);
	gpuAlloc(d_v, h_v);

	/* Run calculation multiple times */
	for (int i = 0; i < iterations; ++i){
		/* Define block and grid dimensions */
		dim3 dimGrid((cols + 1023) / 1024, 1, 1);
		dim3 dimBlock(1024, 1, 1);

		/* Call the kernel function */
		chLaunchKernel(
			dot::colwise_eigen,
			dimGrid, dimBlock, 0, 0,
			d_m1, d_m2, d_v, rows, cols
		);
	}

	/* Copy vector of dot products to host */
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

template<>
scalar dotProduct<dev, K::merged_eigen>(
	const MatrixXs& m1, const MatrixXs& m2, int iterations
){
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
	scalar *d_m12, *d_v;

	gpuAllocAndCopy(d_m12, m12);
	gpuAlloc(d_v, h_v);

	/* Run calculation multiple times */
	for (int i = 0; i < iterations; ++i){
		/* Define block and grid dimensions */
		dim3 dimGrid((cols + 1023) / 1024, 1, 1);
		dim3 dimBlock(1024, 1, 1);

		/* Call the kernel function */
		chLaunchKernel(
			dot::merged_eigen,
			dimGrid, dimBlock, 0, 0,
			d_m12, d_v, rows, cols
		);
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

// template<>
// scalar dotProduct<dev, K::colwise_manual>(
// 	const MatrixXs& m1, const MatrixXs& m2, int iterations
// ){
// 	/* Dimensions check */
// 	assert(m1.rows() == m2.rows());
// 	assert(m1.cols() == m2.cols());

// 	int rows = m1.rows();
// 	int cols = m1.cols();

// 	/* Vector of dot products */
// 	ArrayXs h_v(cols);

// 	/* Allocate memory on device and copy data */
// 	scalar *d_m1, *d_m2, *d_v;
// 	gpuAllocAndCopy(d_m1, m1);
// 	gpuAllocAndCopy(d_m2, m2);
// 	gpuAlloc(d_v, h_v);

// 	/* Run calculation multiple times */
// 	for (int i = 0; i < iterations; ++i){
// 		/* Define block and grid dimensions */
// 		dim3 dimGrid((cols + 1023) / 1024, 1, 1);
// 		dim3 dimBlock(1024, 1, 1);

// 		/* Call the kernel function */
// 		chLaunchKernel(
// 			dot::colwise_manual,
// 			dimGrid, dimBlock, 0, 0,
// 			d_m1, d_m2, d_v, rows, cols
// 		);
// 	}

// 	/* Copy vector of dot products to host */
// 	gpuMemcpyDeviceToHost(d_v, h_v);

// 	/* Calculate the sum of dot products */
// 	scalar finalResult_gpu = h_v.sum();

// 	/* Deallocate device memory */
// 	gpuFree(d_m1);
// 	gpuFree(d_m2);
// 	gpuFree(d_v);

// 	/* Return the result */
// 	return finalResult_gpu;
// }

} // namespace Kokkidio::gpu
