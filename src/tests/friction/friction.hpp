#ifndef KOKKIDIO_FRICTION_HPP
#define KOKKIDIO_FRICTION_HPP

#include <Kokkidio.hpp>

// #include <type_traits>
// #include <cassert>
// #include <chrono>

namespace Kokkidio
{

namespace phys {
	KOKKIDIO_CONSTANT( constexpr scalar g {9.81}; )
	// KOKKIDIO_CONSTANT(g, 9.81);
}

namespace fric
{

enum class Kernel {
	cstyle,
	eigen_colwise_stackbuf,
	eigen_colwise_fullbuf,
	eigen_ranged_fullbuf,
	eigen_ranged_chunkbuf,
	context_ranged,
};

} // namespace fric


#define KOKKIDIO_FRICTION_ARGS \
	ArrayXXs& flux_out, \
	const ArrayXXs& flux_in, \
	const ArrayXXs& d, \
	const ArrayXXs& v, \
	const ArrayXXs& n, \
	int nRuns


namespace cpu
{
using fric::Kernel;

template<Target target, fric::Kernel k>
void friction( KOKKIDIO_FRICTION_ARGS );

} // namespace cpu

namespace gpu
{
using fric::Kernel;

template<Target target, fric::Kernel k>
void friction( KOKKIDIO_FRICTION_ARGS );

} // namespace gpu

namespace unif
{

enum class Kernel {
	cstyle,
	kokkidio_index_stackbuf,
	kokkidio_index_fullbuf,
	kokkidio_range_fullbuf,
	kokkidio_range_chunkbuf,
	context_ranged,
};

template<Target target, Kernel k>
void friction( KOKKIDIO_FRICTION_ARGS );

} // namespace unif


namespace detail
{

// template<typename T>
// KOKKOS_FUNCTION auto rowOrRef( T&& buf, Index row ) -> decltype(auto) {
// 	if constexpr ( T::IsVectorAtCompileTime ){
// 		using Scalar = typename T::Scalar;
// 		Scalar& ret { buf[row] };
// 		return ret;
// 	} else {
// 		return buf.row(row);
// 	}
// }


/* version with three buffer values per computation -> no register spillover on GPU */
template<typename T_buf, typename T_fout, typename T_fin, typename T_dn, typename T_v>
KOKKOS_FUNCTION void friction_buf3(
	T_buf && buf,
	T_fout&& flux3s_out, // we pass three rows, but friction only affects the bottom 2
	const T_fin& flux3s_in,
	const T_dn& d,
	const T_v & v,
	const T_dn& n
){
	assert(buf.rows() == 3);

	auto repl = [&](const auto& arr){
		return arr.template replicate<2,1>();
	};

	auto vNorm    { buf.row(0) };
	auto chezyFac { buf.row(1) };
	auto  fricFac { buf.row(2) };
	vNorm = v.matrix().colwise().norm().array();
	chezyFac = phys::g * Kokkidio::pow(n, 2) / Kokkidio::pow(d, 1./3);
	fricFac = chezyFac * vNorm;
	chezyFac /= d;

	auto flux_in  { flux3s_in .template bottomRows<2>() };
	auto flux_out { flux3s_out.template bottomRows<2>() };

	flux_out =
		( flux_in - repl(fricFac) * v ) /
		( 1 + repl(chezyFac) * ( repl(vNorm) + Kokkidio::pow(v, 2) / repl(vNorm) ) );
}

} // namespace detail

} // namespace Kokkidio

#endif