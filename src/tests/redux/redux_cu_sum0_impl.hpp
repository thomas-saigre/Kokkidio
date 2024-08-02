
#ifdef KOKKIDIO_REDUX_SUM0_CONSTEXPR_BLOCKSIZE
template <Index blockSize>
#endif

__global__ void KOKKIDIO_REDUX_SUM_NAME(scalar* input, scalar* output, int n) {
	#ifdef KOKKIDIO_REDUX_SUM0_CONSTEXPR_BLOCKSIZE
	static __shared__ scalar shared_data[blockSize];
	#else
	extern __shared__ scalar shared_data[];
	#endif

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	/* Make sure this is a neutral element for the type of reduction,
	 * e.g. 0 for a sum, numeric_limits<...>::max for min, etc. */
	scalar thread_sum = 0;

	/* this loop iterates across the whole input array */
	while (i < n) {
		thread_sum += input[i];
		/* here, we skip ahead to the next location 
		 * that is not assigned to any other thread */
		i += blockDim.x * gridDim.x;
	}

	/* If the number of elements is larger than the grid,
	 * then this fills up the shared memory.
	 * If not, e.g. in the later iterations,
	 * then this sets everything but the processed values to the initial value
	 * of thread_sum. 
	 * So that initial value MUST be a neutral element for the reduction. */
	shared_data[tid] = thread_sum;
	__syncthreads();

	// int s_init = ( n >= blockSize ? blockDim.x : getNextPowerOf2(n) ) / 2;

	/* This loop performs the reduction within the block */
	for (int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			shared_data[tid] += shared_data[tid + s];
		}
		__syncthreads();
	}

	/* After the block-level reduction,
	 * the data is written to a global output array.
	 * Therefore, this output array must have at least as many elements
	 * as there are blocks. */
	if (tid == 0) {
		output[blockIdx.x] = shared_data[0];
	}
}

#undef KOKKIDIO_REDUX_SUM_NAME
#undef KOKKIDIO_REDUX_SUM0_CONSTEXPR_BLOCKSIZE