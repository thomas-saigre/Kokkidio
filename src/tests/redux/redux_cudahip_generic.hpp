#ifndef KOKKIDIO_REDUX_CUDAHIP_GENERIC_HPP
#define KOKKIDIO_REDUX_CUDAHIP_GENERIC_HPP

#include "redux.hpp"

namespace Kokkidio::gpu::redux
{


template<Index blockSize, typename Scalar, typename Op>
__device__ void warpReduce(volatile Scalar *sdata, unsigned int tid, Op&& op){
	#define KOKKIDIO_WR_UNROLL(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		sdata[tid] = op( sdata[tid], sdata[tid + BLOCKSIZE_MAX / 2] ); \
	}
	KOKKIDIO_WR_UNROLL(64)
	KOKKIDIO_WR_UNROLL(32)
	KOKKIDIO_WR_UNROLL(16)
	KOKKIDIO_WR_UNROLL( 8)
	KOKKIDIO_WR_UNROLL( 4)
	KOKKIDIO_WR_UNROLL( 2)
	#undef KOKKIDIO_WR_UNROLL
}

template<Index blockSize, typename Scalar, typename Op>
KOKKIDIO_INLINE __device__ void reduce(const Scalar* input, Scalar* const output, Index n, Op&& op, Scalar neutralElement){
	static __shared__ Scalar sdata[blockSize];
	/* Every thread processes two values now,
	 * so i and gridSize get doubled.
	 * gridSize is the number of elements that are skipped in the while-loop */
	unsigned int
		tid = threadIdx.x,
		i = 2 * blockSize * blockIdx.x + tid,
		gridSize = 2 * blockSize * gridDim.x;
	sdata[tid] = neutralElement;

	while (i < n){
		sdata[tid] = op( sdata[tid], input[i] );
		if (i+blockDim.x < n){
			sdata[tid] = op( sdata[tid], input[i+blockDim.x] );
		}
		i += gridSize;
	}
	__syncthreads();

	#define R6_REDUX(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		if (tid < BLOCKSIZE_MAX / 2){ \
			sdata[tid] = op( sdata[tid], sdata[tid + BLOCKSIZE_MAX / 2] ); \
		} \
		__syncthreads(); \
	}
	R6_REDUX(1024)
	R6_REDUX( 512)
	R6_REDUX( 256)
	R6_REDUX( 128)
	#undef R6_REDUX

	if (tid < 32) warpReduce<blockSize, Scalar, Op>( sdata, tid, std::forward<Op>(op) );
	if (tid == 0) output[blockIdx.x] = sdata[0];
}


template<Index blockSize, typename Scalar>
__global__ void sum(const Scalar* input, Scalar* const output, Index n){
	reduce<blockSize, Scalar>(
		input, output, n,
		[](Scalar a, Scalar b){ return a + b; },
		0
	);
}


template<Index blockSize, typename Scalar>
__global__ void prod(const Scalar* input, Scalar* const output, Index n){
	reduce<blockSize, Scalar>(
		input, output, n,
		[](Scalar a, Scalar b){ return a * b; },
		1
	);
}


template<Index blockSize, typename Scalar>
__global__ void min(const Scalar* input, Scalar* const output, Index n){
	reduce<blockSize, Scalar>(
		input, output, n,
		[](Scalar a, Scalar b){ return b < a ? b : a; },
		std::numeric_limits<Scalar>::max()
	);
}


template<Index blockSize, typename Scalar>
__global__ void max(const Scalar* input, Scalar* const output, Index n){
	reduce<blockSize, Scalar>(
		input, output, n,
		[](Scalar a, Scalar b){ return b > a ? b : a; },
		std::numeric_limits<Scalar>::lowest()
	);
}


} // namespace Kokkidio::gpu::redux

#endif
