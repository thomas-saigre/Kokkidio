#ifdef KOKKIDIO_REDUX_SUM6_CONSTEXPR_BRANCH
template <Index blockSize, bool branchWhile, typename Scalar>
#else
template <Index blockSize, typename Scalar>
#endif

__global__ void KOKKIDIO_REDUX_SUM_NAME(const Scalar* input, Scalar* const output, Index n){
	static __shared__ Scalar sdata[blockSize];
	/* Every thread processes two values now,
	 * so i and gridSize get doubled.
	 * gridSize is the number of elements that are skipped in the while-loop */
	unsigned int
		tid = threadIdx.x,
		i = 2 * blockSize * blockIdx.x + tid,
		gridSize = 2 * blockSize * gridDim.x;
		// i = 2 * blockDim.x * blockIdx.x + tid,
		// gridSize = 2 * blockDim.x * gridDim.x;
	sdata[tid] = 0;

	while (i < n){
		#ifdef KOKKIDIO_REDUX_SUM6_CONSTEXPR_BRANCH
		if constexpr (!branchWhile){
			sdata[tid] += input[i] + input[i+blockDim.x];
		} else
		#endif
		{
			sdata[tid] += input[i];
			if (i+blockDim.x < n){
				sdata[tid] += input[i+blockDim.x];
			}
		}
		i += gridSize;
	}
	__syncthreads();

	#define R6_SUM(BLOCKSIZE_MAX) \
	if (blockSize >= BLOCKSIZE_MAX){ \
		if (tid < BLOCKSIZE_MAX / 2){ \
			sdata[tid] += sdata[tid + BLOCKSIZE_MAX / 2]; \
		} \
		__syncthreads(); \
	}
	R6_SUM(1024)
	R6_SUM( 512)
	R6_SUM( 256)
	R6_SUM( 128)
	#undef R6_SUM

	if (tid < 32) warpSum<blockSize, Scalar>(sdata, tid);
	if (tid == 0) output[blockIdx.x] = sdata[0];

	// if (blockIdx.x == 0 && tid == 0){
	// // if (tid == 0){
	// 	printf( "Current reduction result: %f\n", sdata[0]);
	// }
	// __syncthreads();
	// output[0] = 10;
}

#undef KOKKIDIO_REDUX_SUM_NAME
#undef KOKKIDIO_REDUX_SUM6_CONSTEXPR_BRANCH
