#include <iostream>

#include "friction.hpp"
#include "unifyBackends.hpp"

namespace Kokkidio::gpu
{

namespace kernel {

__global__ void friction_manual(
	scalar*,
	scalar* flux_out_d,
	const scalar* flux_in_d,
	const scalar* d_d,
	const scalar* v_d,
	const scalar* n_d,
	Index nCols
){
	auto i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
	if ( i < nCols ){
		#include "impl/friction_cstyle.hpp"
	}
}


__global__ void friction_buf3(
	scalar* buf_d,
	scalar* flux_out_d,
	const scalar* flux_in_d,
	const scalar* d_d,
	const scalar* v_d,
	const scalar* n_d,
	Index nCols
){
	auto i = static_cast<Index>(blockIdx.x * blockDim.x + threadIdx.x);
	if ( i < nCols ){
		/* nCols was zero when a long was used */
		// printf("nCols in kernel::friction_buf3: %li\n", nCols);
		/* it wasn't about Eigen::Map, it was about Eigen::pow */
		#include "impl/friction_buf3.hpp"
	}
}

} // namespace kernel

// Wrapper for the CUDA kernel

template<Target target, Kernel k>
void friction(
	ArrayXXs& flux_out,
	const ArrayXXs& flux_in,
	const ArrayXXs& d,
	const ArrayXXs& v,
	const ArrayXXs& n,
	int nRuns
){
	using K = Kernel;

	#ifndef NDEBUG
	auto assertCols = [&](const auto& arr){
		assert( flux_out.cols() == arr.cols() );
	};
	assertCols(flux_in);
	assertCols(d);
	assertCols(v);
	assertCols(n);
	#endif

	Index nColsTotal { flux_out.cols() };

	Index bufRows;
	if (k == K::cstyle){
		bufRows = 0;
	} else
	if (k == K::eigen_colwise_fullbuf){
		bufRows = 3;
	}
	ArrayXXs buf {bufRows, nColsTotal};

	// Allocate device arrays
	scalar
		*dev_buf,
		*dev_flux_out,
		*dev_flux_in,
		*dev_d,
		*dev_v,
		*dev_n;

	gpuAlloc(dev_buf, buf);
	gpuAlloc(dev_flux_out, flux_out);

	gpuAllocAndCopy(dev_flux_in, flux_in);
	gpuAllocAndCopy(dev_d, d);
	gpuAllocAndCopy(dev_v, v);
	gpuAllocAndCopy(dev_n, n);


	for (int i{0}; i<nRuns; ++i){
		dim3 dimGrid((nColsTotal + 1023) / 1024, 1, 1);
		dim3 dimBlock(1024, 1, 1);

		#define KOKKIDIO_KERNEL_CALL(NAME) chLaunchKernel( NAME, \
				dimGrid, dimBlock, 0, 0, \
				dev_buf, dev_flux_out, dev_flux_in, dev_d, dev_v, dev_n, \
				nColsTotal \
			);

		if constexpr ( k == K::cstyle ){
			KOKKIDIO_KERNEL_CALL(kernel::friction_manual);
		} else if constexpr ( k == K::eigen_colwise_fullbuf ){
			KOKKIDIO_KERNEL_CALL(kernel::friction_buf3);
		}
	}

	#undef KOKKIDIO_KERNEL_CALL

	gpuMemcpyDeviceToHost(dev_flux_out, flux_out);

	// Cleanup
	auto freeArgs = [](std::vector<scalar*>&& arrs){
		for (auto ptr : arrs){
			gpuFree(ptr);
		}
	};
	freeArgs( {dev_flux_in, dev_buf, dev_flux_out, dev_d, dev_v, dev_n} );
}

constexpr Target dev {Target::device};
using K = Kernel;
template void friction<dev, K::cstyle >(KOKKIDIO_FRICTION_ARGS);
template void friction<dev, K::eigen_colwise_fullbuf>(KOKKIDIO_FRICTION_ARGS);

} // namespace Kokkidio::gpu
