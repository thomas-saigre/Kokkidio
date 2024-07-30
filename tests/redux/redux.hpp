#ifndef KOKKIDIO_REDUX_HPP
#define KOKKIDIO_REDUX_HPP

#include <type_traits>
#include <cassert>

#include <Kokkidio.hpp>

namespace Kokkidio
{

namespace gpu
{
enum class Kernel {
	sum0_dyn,
	sum0_fixed,
	sum6,
	sum6_cbranch,
	sum_generic,
};

template<Target target, Kernel k>
scalar sum(const ArrayXXs& values, int nRuns);


enum class Reduction {
	sum,
	prod,
	min,
	max,
};

template<Target target, Reduction reduction>
scalar reduce(const ArrayXXs& values, int nRuns);

} // namespace gpu


namespace cpu
{

enum class Kernel {
	seq,
	manual_global_var_only,
	manual_with_local_var,
	eigen_global_var_only,
	eigen_with_local_var,
};

template<Target target, Kernel k>
scalar sum(const ArrayXXs& values, int nRuns);

enum class Reduction {
	sum,
	prod,
	min,
	max,
};

template<Target target, Reduction reduction>
scalar reduce(const ArrayXXs& values, int nRuns);

} // namespace cpu


} // namespace Kokkidio

#endif
