#include "axpy.hpp"

namespace Kokkidio::gpu
{

template<>
void axpy<Target::device, Kernel::cstyle>(KOKKIDIO_AXPY_ARGS){

	Index nRows {z.rows()};
	sycl::queue queue(sycl::gpu_selector{});
	auto x_d = sycl::malloc_device<scalar>(nRows, queue);
	auto y_d = sycl::malloc_device<scalar>(nRows, queue);
	auto z_d = sycl::malloc_device<scalar>(nRows, queue);

	queue.memcpy(x_d, x.data(), nRows * sizeof(scalar));
	queue.memcpy(y_d, y.data(), nRows * sizeof(scalar));

	auto uRows { static_cast<std::size_t>(nRows) };
	for (int iter = 0; iter < nRuns; ++iter){
		queue.parallel_for( sycl::range<1>{uRows}, [=](sycl::id<1> idx){
			auto i = idx[0];
			z_d[i] = a * x_d[i] + y_d[i];
		} ).wait();
	}
	/* Copy results back to host */
	queue.memcpy(z.data(), z_d, nRows * sizeof(scalar)).wait();

	/* Deallocate device memory */
	sycl::free(x_d, queue);
	sycl::free(y_d, queue);
	sycl::free(z_d, queue);
}

} // namespace Kokkidio::gpu
