#include "friction.hpp"

namespace Kokkidio::gpu
{

template<>
void axpy<Target::device, Kernel::cstyle>(
	ArrayXXs& flux_out,
	const ArrayXXs& flux_in,
	const ArrayXXs& d,
	const ArrayXXs& v,
	const ArrayXXs& n,
	int nRuns
){
	Index nCols { flux_out.cols() };
	sycl::queue queue(sycl::gpu_selector{});
	auto alloc = [&](const auto& obj_h){
		return sycl::malloc_device<scalar>(obj_h.size(), queue);
	};
	auto allocAndCopy(const auto& obj_h){
		auto data_d = alloc(obj_h);
		queue.memcpy( data_d, obj_h.data(), obj_h.size() * sizeof(scalar) );
	};
	auto flux_out_d { alloc(flux_out) };
	auto flux_in_d  { allocAndCopy(flux_in) };
	auto d_d        { allocAndCopy(d) };
	auto v_d        { allocAndCopy(v) };
	auto n_d        { allocAndCopy(n) };

	auto uCols { static_cast<std::size_t>(nCols) };
	for (int iter = 0; iter < nRuns; ++iter){
		queue.parallel_for( sycl::range<1>{uCols}, [=](sycl::id<1> idx){
			auto i = idx[0];
			scalar
				vNorm { sqrt( pow(v_d[2 * idx], 2) + pow(v_d[2 * idx + 1], 2) ) },
				chezyFac = phys::g * pow(n_d[idx], 2) / pow(d_d[idx], 1./3),
				fricFac = chezyFac * vNorm;
			chezyFac /= d_d[idx];

			for (Index row = 0; row<2; ++row){
				flux_out_d[3 * idx + row + 1] =
				(flux_in_d[3 * idx + row + 1] - fricFac * v_d[2 * idx + row] ) /
				( 1 + chezyFac * ( vNorm + pow(v_d[2 * idx + row], 2) / vNorm ) );
			}
		} ).wait();
	}
	/* Copy results back to host */
	queue.memcpy(
		flux_out.data(), //dst
		flux_out_d, //src
		flux_out.size() * sizeof(scalar)
	).wait();

	/* Deallocate device memory */
	auto freeM = [&](auto&& ptrs){
		for ( auto& ptr : ptrs ){
			sycl::free(ptr, queue);
		};
	};
	freeM(flux_out_d, flux_in_d, d_d, v_d, n_d);
}

} // namespace Kokkidio::gpu
