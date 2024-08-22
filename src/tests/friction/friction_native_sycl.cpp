#include "friction.hpp"

namespace Kokkidio::gpu
{

template<Target target, Kernel k>
void friction(
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
	auto allocAndCopy = [&](const auto& obj_h){
		auto data_d = alloc(obj_h);
		queue.memcpy( data_d, obj_h.data(), obj_h.size() * sizeof(scalar) );
		return data_d;
	};
	auto flux_out_d { alloc(flux_out) };
	auto flux_in_d  { allocAndCopy(flux_in) };
	auto d_d        { allocAndCopy(d) };
	auto v_d        { allocAndCopy(v) };
	auto n_d        { allocAndCopy(n) };

	auto uCols { static_cast<std::size_t>(nCols) };
	auto run = [&](auto&& func){
		for (int iter = 0; iter < nRuns; ++iter){
			queue.parallel_for( sycl::range<1>{uCols}, func ).wait();
		}
	};

	using K = Kernel;
	if constexpr ( k == K::cstyle ){
		run( [=](sycl::id<1> idx){
			auto i = idx[0];
			#include "impl/friction_cstyle.hpp"
		} );
	} else if constexpr ( k == K::eigen_colwise_fullbuf ){
		auto buf_d = sycl::malloc_device<scalar>(3 * nCols, queue);
		run( [=](sycl::id<1> idx){
			auto i = idx[0];
			#include "impl/friction_buf3.hpp"
		} );
	}
	/* Copy results back to host */
	queue.memcpy(
		flux_out.data(), //dst
		flux_out_d, //src
		flux_out.size() * sizeof(scalar)
	).wait();

	/* Deallocate device memory */
	auto freeM = [&]( std::vector<scalar*>&& ptrs ){
		for ( auto& ptr : ptrs ){
			sycl::free(ptr, queue);
		};
	};
	freeM( {flux_out_d, flux_in_d, d_d, v_d, n_d} );
}

constexpr Target dev {Target::device};
using K = Kernel;
template void friction<dev, K::cstyle >(KOKKIDIO_FRICTION_ARGS);
template void friction<dev, K::eigen_colwise_fullbuf>(KOKKIDIO_FRICTION_ARGS);

} // namespace Kokkidio::gpu
